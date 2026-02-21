"""MARISA trie with minimal perfect hashing for word-to-index mapping."""

import gzip
import struct
from functools import lru_cache
from typing import BinaryIO, Iterable, Optional
from collections import deque

from bitarray import bitarray


class MarisaTrie:
    """Compact read-only word-to-index mapping backed by a radix trie.

    Ingests an iterable of strings, deduplicates, and builds an in-memory
    path-compressed radix trie with subtree word counts for minimal perfect
    hashing.

    Supports:

    * ``index(word)``      - dense unique index in [0, N)
    * ``restore_key(idx)`` - word from index
    * Serialize / deserialize to disk (compact binary format)

    At query time, navigation uses plain Python parallel lists
    (``_node_labels``, ``_node_children``, ``_node_counts``,
    ``_node_terminal``) with plain list-indexing — no rank/select overhead.

    The binary format uses CSR (child_count + flat_children) arrays, so
    deserialization is a direct ``frombytes`` with no rank/select computation.

    ``cache_size`` controls the ``lru_cache`` capacity installed on each
    instance's ``index()`` method.  When ``None``, defaults to the full
    vocabulary size (all words fit in cache with zero evictions).  Pass an
    integer to cap memory use at the cost of occasional re-traversals.
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        words: Iterable[str],
        *,
        cache_size: Optional[int] = None,
    ) -> None:
        """Build a MarisaTrie from an iterable of strings.

        Args:
            words: Iterable of strings.  Duplicates are silently removed.
            cache_size: ``lru_cache`` capacity for ``index()``.  Defaults to
                the full vocabulary size (unbounded cache hits).
        """
        unique = list(dict.fromkeys(words))
        self._n = len(unique)
        self._cache_size: Optional[int] = cache_size

        if self._n == 0:
            self._root_is_terminal: bool            = False
            self._root_children:    list[int]       = []
            self._node_labels:      list[str]       = []
            self._node_label_lens:  list[int]       = []
            self._node_terminal:    list[bool]      = []
            self._node_children:    list[list[int]] = []
            self._node_counts:      list[int]       = []
            self._first_char_maps:  list[dict]      = [{}]
            self._prefix_counts:    list[list[int]] = [[0]]
            self._attach_index_cache(cache_size)
            return

        root = self._build_intermediate_trie(unique)
        self._root_is_terminal = ("" in root)
        self._build_arrays(root)
        self._attach_index_cache(cache_size)

    # ------------------------------------------------------------------ #
    #  Build helpers                                                       #
    # ------------------------------------------------------------------ #

    def _build_intermediate_trie(self, words: list[str]) -> dict:
        """Build a character-level dict-of-dicts trie from unique words."""
        root: dict = {}
        for word in words:
            node = root
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node[""] = True
        return root

    def _build_arrays(self, root: dict) -> None:
        """Path-compress the intermediate trie into parallel arrays.

        Populates ``_root_children``, ``_node_labels``, ``_node_terminal``,
        ``_node_children``, and ``_node_counts``.  All node arrays are in
        BFS order so that ``to_bytes()`` can emit CSR arrays without any
        additional traversal.
        """
        node_labels:   list[str]       = []
        node_terminal: list[bool]      = []
        node_children: list[list[int]] = []

        # children_map[-1] = root's children; children_map[i] = node i's children
        children_map: dict[int, list[int]] = {}
        node_queue: deque = deque()
        next_idx = 0

        def _emit_children(inode: dict, parent_idx: int) -> None:
            nonlocal next_idx
            child_indices: list[int] = []
            for char, child_node in inode.items():
                if char == "":
                    continue
                # Path compression: collapse single-child non-terminal chains
                label   = char
                current = child_node
                while True:
                    has_term = "" in current
                    if has_term or (len(current) - has_term) != 1:
                        break
                    for k, v in current.items():
                        if k != "":
                            label  += k
                            current = v
                            break

                idx = next_idx
                next_idx += 1
                child_indices.append(idx)
                node_labels.append(label)
                node_terminal.append("" in current)
                node_children.append([])   # placeholder; filled after BFS
                node_queue.append((current, idx))

            if child_indices:
                children_map[parent_idx] = child_indices

        _emit_children(root, -1)
        while node_queue:
            inode, parent_idx = node_queue.popleft()
            _emit_children(inode, parent_idx)

        # Wire up _node_children from children_map
        root_children = children_map.get(-1, [])
        for parent_idx, child_idxs in children_map.items():
            if parent_idx >= 0:
                node_children[parent_idx] = child_idxs

        self._root_children = root_children
        self._node_labels   = node_labels
        self._node_terminal = node_terminal
        self._node_children = node_children
        self._node_counts   = self._compute_counts(children_map)
        self._node_label_lens = [len(lb) for lb in node_labels]
        self._build_navigation_tables()

        # Build {word: idx} cheaply from the just-populated structures;
        # consumed and freed on the first to_dict() call.
        self._word_to_idx: dict[str, int] = self._build_word_index(children_map)

    def _build_navigation_tables(self) -> None:
        """Build O(1) child-dispatch and prefix-count structures.

        ``_first_char_maps[node+1]``  maps the first character of each child's
        label to ``(position_in_children_list, child_node_id)``.

        ``_prefix_counts[node+1][i]`` is the cumulative subtree-word count of
        all siblings *before* position ``i``, enabling O(1) MPH index
        accumulation without an inner loop.

        Index 0 is the virtual root (``node == -1``); index ``i+1`` is node ``i``.
        """
        _labels  = self._node_labels
        _counts  = self._node_counts
        _children = self._node_children

        all_children: list[list[int]] = [self._root_children] + _children
        first_char_maps: list[dict[str, tuple[int, int]]] = []
        prefix_counts:   list[list[int]] = []

        for children in all_children:
            fc_map: dict[str, tuple[int, int]] = {}
            pc: list[int] = [0]
            running = 0
            for i, child in enumerate(children):
                fc_map[_labels[child][0]] = (i, child)
                running += _counts[child]
                pc.append(running)
            first_char_maps.append(fc_map)
            prefix_counts.append(pc)

        self._first_char_maps = first_char_maps
        self._prefix_counts   = prefix_counts

    def _compute_counts(self, children_map: dict[int, list[int]]) -> list[int]:
        """Compute subtree word counts bottom-up."""
        n        = len(self._node_labels)
        counts   = [0] * n
        _get     = children_map.get
        terminal = self._node_terminal
        for i in range(n - 1, -1, -1):
            c = 1 if terminal[i] else 0
            kids = _get(i)
            if kids:
                for ch in kids:
                    c += counts[ch]
            counts[i] = c
        return counts

    def _build_word_index(
        self,
        children_map: dict[int, list[int]],
    ) -> "dict[str, int]":
        """Build ``{word: index}`` via DFS over the just-built arrays.

        Uses the same pre-order assignment as ``index()``: each node's
        terminal (if any) is numbered before its children.
        """
        result: dict[str, int] = {}
        counter = 0

        if self._root_is_terminal:
            result[""] = 0
            counter = 1

        _get      = children_map.get
        labels    = self._node_labels
        terminal  = self._node_terminal
        stack: list[tuple[int, str]] = []
        for child_idx in reversed(_get(-1, [])):
            stack.append((child_idx, labels[child_idx]))

        while stack:
            node_idx, prefix = stack.pop()
            if terminal[node_idx]:
                result[prefix] = counter
                counter += 1
            kids = _get(node_idx)
            if kids:
                for child_idx in reversed(kids):
                    stack.append((child_idx, prefix + labels[child_idx]))

        return result

    # ------------------------------------------------------------------ #
    #  Forward lookup: word -> index                                       #
    # ------------------------------------------------------------------ #

    def _attach_index_cache(self, cache_size: Optional[int]) -> None:
        """Install per-instance lru_caches on ``index()`` and ``restore_key()``.

        Called at the end of both ``__init__`` and ``from_bytes`` so the caches
        are always correctly sized.  Binding the caches to the instance (rather
        than the class) keeps each trie's cache independent.
        """
        maxsize = cache_size if cache_size is not None else self._n or 1
        self.index = lru_cache(maxsize=maxsize)(self._index_uncached)
        self.restore_key = lru_cache(maxsize=self._n or 1)(self._restore_key_uncached)

    def _index_uncached(self, word: str) -> int:
        """Return the dense unique index for *word* in ``[0, N)``.

        Navigates the in-memory radix trie using O(1) first-char dispatch
        and precomputed prefix counts — no linear scan, no rank/select.

        Raises:
            KeyError: If *word* is not in the trie.
        """
        if self._n == 0:
            raise KeyError(word)

        if word == "":
            if self._root_is_terminal:
                return 0
            raise KeyError(word)

        idx       = 1 if self._root_is_terminal else 0
        remaining = word
        node      = -1          # -1 = virtual root

        _labels        = self._node_labels
        _label_lens    = self._node_label_lens
        _counts        = self._node_counts
        _terminal      = self._node_terminal
        _first_char_maps = self._first_char_maps
        _prefix_counts   = self._prefix_counts

        while remaining:
            entry = _first_char_maps[node + 1].get(remaining[0])
            if entry is None:
                raise KeyError(word)
            i, child = entry
            label_len = _label_lens[child]
            if remaining[:label_len] != _labels[child]:
                raise KeyError(word)
            # Accumulate subtree counts of preceding siblings (O(1) lookup)
            idx += _prefix_counts[node + 1][i]
            # If the node we are descending *from* is terminal, count it
            if node >= 0 and _terminal[node]:
                idx += 1
            remaining = remaining[label_len:]
            node      = child

        if not _terminal[node]:
            raise KeyError(word)
        return idx

    def __getitem__(self, word: str) -> int:
        """Return the index for a word."""
        return self.index(word)

    def __contains__(self, word: str) -> bool:
        """Check if a word is in the trie."""
        try:
            self.index(word)
            return True
        except KeyError:
            return False

    def __len__(self) -> int:
        """Return the number of unique words."""
        return self._n

    def __repr__(self) -> str:
        return f"MarisaTrie({self._n} words)"

    # ------------------------------------------------------------------ #
    #  Reverse lookup: index -> word                                       #
    # ------------------------------------------------------------------ #

    def _restore_key_uncached(self, idx: int) -> str:
        """Return the word corresponding to *idx*.

        Raises:
            IndexError: If *idx* is out of ``[0, N)``.
        """
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of range [0, {self._n})")

        if self._root_is_terminal and idx == 0:
            return ""

        target   = idx - (1 if self._root_is_terminal else 0)
        path: list[str] = []
        node     = -1
        children = self._root_children

        _labels   = self._node_labels
        _counts   = self._node_counts
        _terminal = self._node_terminal
        _children = self._node_children

        while True:
            if node >= 0 and _terminal[node]:
                if target == 0:
                    return "".join(path)
                target -= 1

            for child in children:
                count = _counts[child]
                if target < count:
                    path.append(_labels[child])
                    node     = child
                    children = _children[child]
                    break
                target -= count
            else:
                raise IndexError(f"Index {idx} out of range")

    # ------------------------------------------------------------------ #
    #  Bulk enumeration                                                    #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, int]:
        """Return ``{word: index}`` for every word in O(N).

        The first call after construction returns the zero-cost mapping
        built during ``__init__`` and frees it immediately.  Subsequent
        calls (or calls on a deserialized trie) fall back to a DFS over
        the runtime arrays.
        """
        cached: Optional[dict[str, int]] = self.__dict__.pop("_word_to_idx", None)
        if cached is not None:
            return cached

        # Fallback: DFS over runtime arrays (used after from_bytes)
        result: dict[str, int] = {}
        if self._n == 0:
            return result

        idx = 0
        if self._root_is_terminal:
            result[""] = 0
            idx = 1

        _labels   = self._node_labels
        _terminal = self._node_terminal
        _children = self._node_children

        stack: list[tuple[int, str]] = []
        for child in reversed(self._root_children):
            stack.append((child, _labels[child]))

        while stack:
            node, prefix = stack.pop()
            if _terminal[node]:
                result[prefix] = idx
                idx += 1
            for child in reversed(_children[node]):
                stack.append((child, prefix + _labels[child]))

        return result

    # ------------------------------------------------------------------ #
    #  Serialization helpers                                               #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _wrap_read_stream(stream: BinaryIO, compression: Optional[str]) -> BinaryIO:
        if compression == "gzip":
            return gzip.open(stream, "rb")  # type: ignore[return-value]
        elif compression is None:
            return stream
        else:
            raise ValueError(f"Unsupported compression: {compression}")

    @staticmethod
    def _wrap_write_stream(stream: BinaryIO, compression: Optional[str]) -> BinaryIO:
        if compression == "gzip":
            return gzip.open(stream, "wb", compresslevel=9)  # type: ignore[return-value]
        elif compression is None:
            return stream
        else:
            raise ValueError(f"Unsupported compression: {compression}")

    # ------------------------------------------------------------------ #
    #  Serialization: CSR arrays, never stored on the instance           #
    # ------------------------------------------------------------------ #

    def to_bytes(self) -> bytes:
        """Serialize the trie to bytes (CSR arrays).

        Result is cached on the instance after the first call — the trie is
        immutable so the bytes never change.

        CSR arrays (``child_count`` + ``flat_children``) are built on the
        fly from the in-memory arrays and are **not** stored back on the
        instance.  Deserialization reconstructs the parallel lists with a
        single ``frombytes`` pass at O(N) with no rank/select computation.

        Returns:
            Binary representation of the trie.
        """
        cached = self.__dict__.get("_cached_bytes")
        if cached is not None:
            return cached
        import io

        num_nodes = len(self._node_labels)

        # Labels, terminal bits, counts in BFS order (= array order)
        labels_buf    = bytearray()
        terminal_bits = bitarray()
        for i in range(num_nodes):
            lb = self._node_labels[i].encode("utf-8")
            labels_buf.extend(struct.pack("<I", len(lb)))
            labels_buf.extend(lb)
            terminal_bits.append(self._node_terminal[i])

        counts_bytes   = struct.pack(f"<{num_nodes}I", *self._node_counts) if num_nodes else b""
        labels_bytes   = bytes(labels_buf)
        terminal_bytes = terminal_bits.tobytes() if num_nodes else b""

        # Build CSR arrays.
        # child_count[0]   = len(_root_children)
        # child_count[i+1] = len(_node_children[i])  for i in 0..num_nodes-1
        # flat_children    = _root_children ++ concat(_node_children)
        import array as _array
        import sys as _sys
        all_counts: list[int] = [len(self._root_children)]
        flat:       list[int] = list(self._root_children)
        for ch in self._node_children:
            all_counts.append(len(ch))
            flat.extend(ch)

        cc_arr = _array.array('I', all_counts)
        fc_arr = _array.array('I', flat) if flat else _array.array('I')
        if _sys.byteorder != 'little':
            cc_arr.byteswap()
            fc_arr.byteswap()
        child_count_bytes   = cc_arr.tobytes()
        flat_children_bytes = fc_arr.tobytes()

        buf = io.BytesIO()
        buf.write(b"MTrie")
        buf.write(struct.pack("<Q", 2))          # version 2
        buf.write(struct.pack("<Q", self._n))
        buf.write(struct.pack("<?", self._root_is_terminal))
        buf.write(struct.pack("<QQQQQ",
                              len(labels_bytes),
                              len(terminal_bytes),
                              len(counts_bytes),
                              len(child_count_bytes),
                              len(flat_children_bytes)))
        buf.write(labels_bytes)
        buf.write(terminal_bytes)
        buf.write(counts_bytes)
        buf.write(child_count_bytes)
        buf.write(flat_children_bytes)
        result = buf.getvalue()
        self._cached_bytes = result
        return result

    @classmethod
    def from_bytes(
        cls,
        data: bytes,
        *,
        cache_size: Optional[int] = None,
    ) -> "MarisaTrie":
        """Deserialize a trie from bytes, reconstructing the runtime arrays.

        Reconstruction is a direct ``frombytes`` over CSR arrays with no
        rank/select computation.

        Args:
            data: Binary representation produced by ``to_bytes()``.
            cache_size: ``lru_cache`` capacity for ``index()``.  Defaults to
                the full vocabulary size.

        Returns:
            Reconstructed ``MarisaTrie``.
        """
        import io
        import array as _array
        import sys as _sys

        f = io.BytesIO(data)
        magic, ver = struct.unpack("<5sQ", f.read(13))
        assert magic == b"MTrie" and ver == 2, f"Invalid format: {magic!r} v{ver} (expected 2)"

        n                = struct.unpack("<Q", f.read(8))[0]
        root_is_terminal = struct.unpack("<?", f.read(1))[0]

        # layout: labels | terminal | counts | child_count | flat_children
        labels_len, terminal_len, counts_len, child_count_len, flat_children_len = (
            struct.unpack("<QQQQQ", f.read(40))
        )
        labels_bytes        = f.read(labels_len)
        terminal_bytes      = f.read(terminal_len)
        counts_bytes        = f.read(counts_len)
        child_count_bytes   = f.read(child_count_len)
        flat_children_bytes = f.read(flat_children_len)

        num_nodes = counts_len // 4

        # Parse labels, terminal bits, counts
        node_labels:   list[str]  = []
        node_terminal: list[bool] = []
        node_counts:   list[int]  = []

        terminal_ba = bitarray()
        terminal_ba.frombytes(terminal_bytes)

        offset = 0
        for i in range(num_nodes):
            lb_len = struct.unpack("<I", labels_bytes[offset:offset + 4])[0]
            node_labels.append(
                labels_bytes[offset + 4:offset + 4 + lb_len].decode("utf-8")
            )
            offset += 4 + lb_len
            node_terminal.append(bool(terminal_ba[i]))
            node_counts.append(
                struct.unpack("<I", counts_bytes[i * 4:(i + 1) * 4])[0]
            )

        # Reconstruct children directly from CSR arrays — no rank/select.
        cc = _array.array('I'); cc.frombytes(child_count_bytes)
        fc = _array.array('I'); fc.frombytes(flat_children_bytes)
        if _sys.byteorder != 'little':
            cc.byteswap()
            fc.byteswap()
        fc_off = cc[0]
        root_children: list[int]       = list(fc[0:fc_off])
        node_children: list[list[int]] = []
        for i in range(num_nodes):
            cnt = cc[i + 1]
            node_children.append(list(fc[fc_off:fc_off + cnt]))
            fc_off += cnt

        trie = cls.__new__(cls)
        trie._n                = n
        trie._root_is_terminal = root_is_terminal
        trie._root_children    = root_children
        trie._node_labels      = node_labels
        trie._node_label_lens  = [len(lb) for lb in node_labels]
        trie._node_terminal    = node_terminal
        trie._node_children    = node_children
        trie._node_counts      = node_counts
        trie._cache_size       = cache_size
        trie._build_navigation_tables()
        trie._attach_index_cache(cache_size)
        return trie

    def serialize(self, url: str, storage_options: Optional[dict] = None) -> None:
        """Write the trie to a file.

        Args:
            url: Path or URL where to save.
            storage_options: fsspec options.  Set ``compression='gzip'`` for gzip.
        """
        from fsspec.core import url_to_fs
        opts        = storage_options or {}
        compression = opts.get("compression")
        fs, path    = url_to_fs(url, **opts)
        data        = self.to_bytes()
        with fs.open(path, "wb") as raw_stream:
            with self._wrap_write_stream(raw_stream, compression) as f:
                f.write(data)

    @classmethod
    def load(cls, url: str, storage_options: Optional[dict] = None) -> "MarisaTrie":
        """Load a trie from a file.

        Args:
            url: Path or URL to load from.
            storage_options: fsspec options.  Set ``compression='gzip'`` if compressed.

        Returns:
            Loaded ``MarisaTrie``.
        """
        from fsspec.core import url_to_fs
        opts        = storage_options or {}
        compression = opts.get("compression")
        fs, path    = url_to_fs(url, **opts)
        with fs.open(path, "rb") as raw_stream:
            with cls._wrap_read_stream(raw_stream, compression) as f:
                data = f.read()
        return cls.from_bytes(data)

    def __reduce__(self) -> tuple:
        """Support pickle by serializing to bytes."""
        return (self.from_bytes, (self.to_bytes(),))