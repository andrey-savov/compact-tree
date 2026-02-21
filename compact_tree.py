import array
import bisect
import gzip
import struct
import sys
from typing import Any, BinaryIO, Iterator, Optional

from marisa_trie import MarisaTrie

_INTERNAL = 0xFFFFFFFF  # vcol sentinel for internal (non-leaf) nodes


class CompactTree:
    """Compact read-only nested dict backed by CSR arrays + DAWG + edge labels.

    Two ways to create:

    * ``CompactTree.from_dict(d)`` - build in memory from a Python dict.
    * ``CompactTree(url)``         - deserialize from storage.

    Persist with ``tree.serialize(url)``.
    """

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _walk_dict(d: dict[str, Any], keys_out: set[str],
                   values_out: set[str],
                   _seen_ids: Optional[set] = None) -> None:
        """Recursively collect all keys and unique leaf values.

        Two shortcuts avoid redundant work at scale:

        * **id-based skip**: if the exact same dict *object* is reachable from
          multiple places in the tree, it is walked only once.
        * **key-subset skip**: if every key in *d* is already present in
          *keys_out* (detected via a single C-level ``d.keys() <= keys_out``
          call), the per-key ``keys_out.add`` loop is skipped entirely and
          only leaf values are collected.  In a 3-level dict where the 36
          inner dicts each carry the same 173K key set, this eliminates
          35 × 173K = 6 M redundant Python-level ``set.add`` calls.
        """
        if _seen_ids is None:
            _seen_ids = set()
        did = id(d)
        if did in _seen_ids:
            return
        _seen_ids.add(did)

        # C-level subset check: True when all keys are already in keys_out.
        # Replaces 173K individual set.add no-ops with a single C loop.
        keys_new = not (d.keys() <= keys_out)
        for k, v in d.items():
            if keys_new:
                keys_out.add(k)
            if type(v) is dict:
                CompactTree._walk_dict(v, keys_out, values_out, _seen_ids)
            else:
                values_out.add(v if type(v) is str else str(v))

    @staticmethod
    def _pack_strings(strings: list[str]) -> bytearray:
        """Encode a list of strings as length-prefixed UTF-8."""
        buf = bytearray()
        for s in strings:
            enc = s.encode("utf-8")
            buf.extend(struct.pack("<I", len(enc)))
            buf.extend(enc)
        return buf

    @staticmethod
    def _unpack_strings(data: bytes) -> list[str]:
        """Decode length-prefixed UTF-8 strings from bytes."""
        out: list[str] = []
        off = 0
        while off < len(data):
            ln = struct.unpack("<I", data[off:off + 4])[0]
            off += 4
            out.append(data[off:off + ln].decode("utf-8"))
            off += ln
        return out

    @staticmethod
    def _wrap_read_stream(stream: BinaryIO, compression: Optional[str]) -> BinaryIO:
        """Wrap a file stream with decompression if needed."""
        if compression == "gzip":
            return gzip.open(stream, "rb")  # type: ignore[return-value]
        elif compression is None:
            return stream
        else:
            raise ValueError(f"Unsupported compression: {compression}")

    @staticmethod
    def _wrap_write_stream(stream: BinaryIO, compression: Optional[str]) -> BinaryIO:
        """Wrap a file stream with compression if needed."""
        if compression == "gzip":
            return gzip.open(stream, "wb", compresslevel=9)  # type: ignore[return-value]
        elif compression is None:
            return stream
        else:
            raise ValueError(f"Unsupported compression: {compression}")

    # ------------------------------------------------------------------ #
    #  Factory: from Python dict                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict[str, Any], *,
                  vocabulary_size: Optional[int] = None) -> "CompactTree":
        """Build a *CompactTree* entirely in memory from a nested Python dict.

        Keys must be strings.  Leaf values are stored as strings (non-string
        values are converted via ``str()``).

        Args:
            vocabulary_size: Optional hint for the total number of unique words
                in the source dict (keys + values combined).  Used to size the
                LRU cache on each ``MarisaTrie`` so that all vocabulary entries
                fit in cache with zero evictions.  When ``None`` (default) the
                actual vocabulary sizes are computed automatically from the data
                and used as the cache size for each trie independently.
        """
        # 1. Collect vocabulary and leaf values
        all_keys: set[str] = set()
        unique_values: set[str] = set()
        cls._walk_dict(data, all_keys, unique_values)

        # Determine per-trie cache sizes: honour the caller's hint (applied to
        # both tries) or fall back to the exact vocabulary size for each trie.
        if vocabulary_size is not None:
            key_cache_size: int = vocabulary_size
            val_cache_size: int = vocabulary_size
        else:
            key_cache_size = len(all_keys)
            val_cache_size = len(unique_values)

        # 2. Build MarisaTrie for keys (deduplicated)
        key_trie = MarisaTrie(all_keys, cache_size=key_cache_size)

        # 3. Build MarisaTrie for values (deduplicated)
        val_trie = MarisaTrie(unique_values, cache_size=val_cache_size)

        # Build plain {word: idx} dicts via a single DFS over each trie.
        # This replaces the old pre-warm loop: instead of N individual trie
        # traversals (one per unique word), we do exactly one O(N) DFS pass
        # per trie, eliminating all rank/select overhead for the warm-up phase.
        key_id: dict[str, int] = key_trie.to_dict()
        val_id: dict[str, int] = val_trie.to_dict()

        # 4. BFS -> CSR arrays + vcol + elbl
        import array as _array
        elbl_list: list[int] = []
        vcol_list: list[int] = []

        # List-based BFS queue with a read-head index.
        # Entries are either a dict (internal node) or a positive int (the
        # number of consecutive leaf-rows to emit in bulk).
        # This replaces deque + None sentinels, eliminating:
        #   - 6.2 M deque.append(None) / popleft() calls
        #   - 6.2 M individual leaf-row processing calls
        # Leaf runs between dict children are queued as int counts inline.
        queue: list = []
        qi: int = 0

        # Zero-byte buffer cache for _child_start_arr/_child_count_arr leaf fills.
        # Batches sharing the same size reuse one allocation (avoids repeated
        # 692 KB bytes() allocations for the 36 same-schema L2 leaf batches).
        _zero_bytes_cache: dict[int, bytes] = {}

        # Build _child_start and _child_count directly during BFS.
        # _next_cid tracks the 1-based node id assigned to the NEXT group of
        # children; it advances only when _emit_children registers degree>0
        # children for an internal node.  Leaf batches (int queue items) get
        # zero-filled entries via array.frombytes(bytes(4*n)) — a C-level
        # memory fill far cheaper than n Python-level .append(0) calls.
        _next_cid: int = 1
        _child_start_arr = _array.array('I')
        _child_count_arr = _array.array('I')

        # Cache (sorted_keys, sorted_ids, key_indices) by frozenset of key
        # names so that sibling dicts with the same key set pay the lookup
        # cost only once.  sorted_ids is a plain list[int] so subsequent
        # elbl_list.extend(sorted_ids) calls are C-level list-to-list copies.
        _key_order_cache: dict[frozenset, tuple[list[str], list[int], dict[str, int]]] = {}

        def _emit_children(node: dict[str, Any]) -> None:
            nonlocal _next_cid
            keyset = frozenset(node.keys())
            cached = _key_order_cache.get(keyset)
            if cached is None:
                indices = {k: key_id[k] for k in node.keys()}
                sorted_keys = sorted(node.keys(), key=lambda k: indices[k])
                sorted_ids  = list(map(indices.__getitem__, sorted_keys))
                _key_order_cache[keyset] = (sorted_keys, sorted_ids, indices)
            else:
                sorted_keys, sorted_ids, indices = cached
            degree = len(sorted_keys)
            # Record this node's child range before any early return so both
            # degree==0 and degree>0 paths are covered.
            _child_start_arr.append(_next_cid)
            _child_count_arr.append(degree)
            _next_cid += degree
            if degree == 0:
                return
            # All-leaf fast-path detection: set(map(type, ...)) is a pure
            # C-level pass — no Python frame per item, unlike any(genexpr).
            value_types = set(map(type, node.values()))
            if dict not in value_types:
                # All children are leaves.  sorted_ids is a cached list[int],
                # so extend is a C memory copy.  For vcol, build vals as a
                # concrete list first so the second map() walks a C array
                # (direct PyList_GET_ITEM) rather than chaining two iterator
                # layers (which adds tp_iternext overhead per element).
                elbl_list.extend(sorted_ids)
                vals = list(map(node.__getitem__, sorted_keys))
                if value_types == {str}:
                    vcol_list.extend(map(val_id.__getitem__, vals))
                else:
                    vcol_list.extend(
                        val_id[v if type(v) is str else str(v)] for v in vals
                    )
                queue.append(degree)
                return

            _elbl = elbl_list.append
            _vcol = vcol_list.append
            _queue = queue.append
            pending_leaves: int = 0
            for key in sorted_keys:
                _elbl(indices[key])
                child = node[key]
                if type(child) is dict:
                    _vcol(_INTERNAL)
                    if pending_leaves:
                        _queue(pending_leaves)
                        pending_leaves = 0
                    _queue(child)
                else:
                    _vcol(val_id[child if type(child) is str else str(child)])
                    pending_leaves += 1
            if pending_leaves:
                _queue(pending_leaves)

        _emit_children(data)
        while qi < len(queue):
            item = queue[qi]; qi += 1
            if type(item) is not dict:
                # Batch-emit `item` False bits for a run of leaf-rows.
                n = item
                # Leaf nodes have degree=0; fill their array slots with zeros
                # via C-level frombytes — faster than n×append(0) calls.
                # Cache the zero buffer so repeated same-size batches
                # (e.g. 36 L2 nodes all with the same leaf count) reuse one
                # allocation instead of each creating a fresh bytes(4*n).
                _bz = _zero_bytes_cache.get(n)
                if _bz is None:
                    _bz = bytes(4 * n)
                    _zero_bytes_cache[n] = _bz
                _child_start_arr.frombytes(_bz)
                _child_count_arr.frombytes(_bz)
            else:
                _emit_children(item)

        # Convert accumulated int lists into uint32 arrays in native byte order
        _elbl_arr = _array.array('I', elbl_list)
        _vcol_arr = _array.array('I', vcol_list)

        # 5. Assemble the CompactTree object
        tree = cls.__new__(cls)
        tree.fs = None
        tree.f = None
        tree.mm = None
        tree._key_trie = key_trie
        tree._val_trie = val_trie
        tree._key_vocab_size = key_cache_size
        tree._val_vocab_size = val_cache_size
        tree._child_start = _child_start_arr
        tree._child_count = _child_count_arr
        tree.vcol = _vcol_arr
        tree.elbl = _elbl_arr
        tree._root_list = tree._list_children(0)
        return tree

    # ------------------------------------------------------------------ #
    #  Factory: from file / deserialize                                  #
    # ------------------------------------------------------------------ #

    def __init__(self, url: str, storage_options: Optional[dict] = None):
        """Deserialize a *CompactTree* from storage.
        
        Args:
            url: Path or URL to the CompactTree file.
            storage_options: fsspec options. Set compression='gzip' to read
                           gzip-compressed files.
        """
        from fsspec.core import url_to_fs

        opts = storage_options or {}
        compression = opts.get("compression")
        fs, path = url_to_fs(url, **opts)
        
        with fs.open(path, "rb") as raw_stream:
            with self._wrap_read_stream(raw_stream, compression) as f:
                # Read header
                magic, ver = struct.unpack("<5sQ", f.read(13))
                assert magic == b"CTree" and ver == 5, (
                    f"Unsupported CompactTree format version {ver} (expected 5)"
                )

                (
                    keys_len, val_len, child_count_len, vcol_len, elbl_len,
                    _key_vocab_size, _val_vocab_size,
                ) = struct.unpack("<QQQQQQQ", f.read(56))

                # Read and parse keys MarisaTrie
                keys_bytes = f.read(keys_len)
                self._key_trie = MarisaTrie.from_bytes(
                    keys_bytes,
                    cache_size=_key_vocab_size or None,
                )

                # Read values MarisaTrie
                val_bytes = f.read(val_len)
                self._val_trie = MarisaTrie.from_bytes(
                    val_bytes,
                    cache_size=_val_vocab_size or None,
                )

                self._key_vocab_size = _key_vocab_size
                self._val_vocab_size = _val_vocab_size

                # Reconstruct child_start from stored child_count array
                _cc = array.array('I'); _cc.frombytes(f.read(child_count_len))
                if sys.byteorder != 'little': _cc.byteswap()
                from itertools import accumulate as _acc
                n_nodes = len(_cc)
                self._child_start = array.array('I', list(_acc(_cc, initial=1))[:n_nodes])
                self._child_count = _cc

                # Read vcol and elbl sections
                _vcol = array.array('I'); _vcol.frombytes(f.read(vcol_len))
                if sys.byteorder != 'little': _vcol.byteswap()
                self.vcol = _vcol
                _elbl = array.array('I'); _elbl.frombytes(f.read(elbl_len))
                if sys.byteorder != 'little': _elbl.byteswap()
                self.elbl = _elbl

        # No file handles kept open
        self.fs = None
        self.f = None
        self.mm = None

        self._root_list = self._list_children(0)

    # ------------------------------------------------------------------ #
    #  serialize                                                           #
    # ------------------------------------------------------------------ #

    def serialize(self, url: str, storage_options: Optional[dict] = None) -> None:
        """Write the tree to *url* in binary format.
        
        Args:
            url: Path or URL where to save the CompactTree.
            storage_options: fsspec options. Set compression='gzip' to write
                           gzip-compressed output (level 9).
        """
        from fsspec.core import url_to_fs

        opts = storage_options or {}
        compression = opts.get("compression")
        fs, path = url_to_fs(url, **opts)
        
        keys_bytes = self._key_trie.to_bytes()
        val_bytes = self._val_trie.to_bytes()
        _cc = self._child_count
        if sys.byteorder != 'little':
            _cc = array.array('I', _cc); _cc.byteswap()
        child_count_bytes = _cc.tobytes()
        _vcol = self.vcol
        if sys.byteorder != 'little':
            _vcol = array.array('I', _vcol); _vcol.byteswap()
        vcol_bytes = _vcol.tobytes()
        _elbl = self.elbl
        if sys.byteorder != 'little':
            _elbl = array.array('I', _elbl); _elbl.byteswap()
        elbl_bytes = _elbl.tobytes()
        
        with fs.open(path, "wb") as raw_stream:
            with self._wrap_write_stream(raw_stream, compression) as f:
                f.write(b"CTree")
                f.write(struct.pack("<Q", 5))
                f.write(struct.pack(
                    "<QQQQQQQ",
                    len(keys_bytes), len(val_bytes), len(child_count_bytes),
                    len(vcol_bytes), len(elbl_bytes),
                    self._key_vocab_size,
                    self._val_vocab_size,
                ))
                f.write(keys_bytes)
                f.write(val_bytes)
                f.write(child_count_bytes)
                f.write(vcol_bytes)
                f.write(elbl_bytes)

    # ------------------------------------------------------------------ #
    #  to_dict                                                             #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict:
        """Materialise the tree back into a plain nested Python dict."""
        def _build(kids: list[int]) -> dict:
            out: dict[str, object] = {}
            for kid in kids:
                kv = self.elbl[kid - 1]
                key = self._key_trie.restore_key(kv)
                vv = self.vcol[kid - 1]
                if vv == _INTERNAL:
                    out[key] = _build(self._list_children(kid))
                else:
                    out[key] = self._val_trie.restore_key(vv)
            return out
        return _build(self._root_list)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _vid_to_str(self, vid: int) -> str:
        """Convert a value ID to the corresponding string."""
        return self._val_trie.restore_key(vid)

    def _list_children(self, pos: int) -> list[int]:
        """List all children of a node at a given position."""
        start = self._child_start[pos]
        count = self._child_count[pos]
        return list(range(start, start + count))

    def _find_child(self, v: int, key_vid: int) -> Optional[int]:
        """Binary search among children of node *v* for edge label == *key_vid*.

        Valid because children are emitted sorted by key_id during from_dict.
        Avoids materialising the children list entirely.
        """
        count = self._child_count[v]
        if count == 0:
            return None
        lo = self._child_start[v] - 1  # 0-based elbl index of first child
        hi = lo + count                # exclusive upper bound
        pos = bisect.bisect_left(self.elbl, key_vid, lo, hi)
        if pos < hi and self.elbl[pos] == key_vid:
            return pos + 1             # back to 1-indexed node position
        return None

    def _resolve(self, child_pos: int) -> "str | CompactTree._Node":
        """Resolve a child position to either a leaf string or a ``_Node``."""
        vv = self.vcol[child_pos - 1]
        if vv == _INTERNAL:
            return CompactTree._Node(self, child_pos)
        return self._vid_to_str(vv)

    # ------------------------------------------------------------------ #
    #  _Node  (nested dict interface for sub-trees)                      #
    # ------------------------------------------------------------------ #

    class _Node:
        def __init__(self, tree: "CompactTree", pos: int):
            self.tree = tree
            self.pos = pos

        def _children(self) -> list[int]:
            """List all children of the current node."""
            return self.tree._list_children(self.pos)

        def __getitem__(self, key: str) -> Any:
            try:
                key_vid = self.tree._key_trie[key]
            except KeyError:
                raise KeyError(key)
            child_pos = self.tree._find_child(self.pos, key_vid)
            if child_pos is None:
                raise KeyError(key)
            return self.tree._resolve(child_pos)

        def __iter__(self) -> Iterator[str]:
            for kid in self._children():
                yield self.tree._key_trie.restore_key(self.tree.elbl[kid - 1])

        def __contains__(self, key: str) -> bool:
            try:
                key_vid = self.tree._key_trie[key]
            except KeyError:
                return False
            return self.tree._find_child(self.pos, key_vid) is not None

        def __len__(self) -> int:
            return len(self._children())

        def __repr__(self) -> str:
            """Return an interpretable representation of the Node."""
            # Convert node to dict for repr
            node_dict = {}
            for key in self:
                val = self[key]
                node_dict[key] = val
            return repr(node_dict)

        def __str__(self) -> str:
            """Return a string representation like a Python dict."""
            node_dict = {}
            for key in self:
                val = self[key]
                node_dict[key] = val
            return str(node_dict)

    # ------------------------------------------------------------------ #
    #  Mapping-like interface (root level)                               #
    # ------------------------------------------------------------------ #

    def __getitem__(self, key: str) -> Any:
        try:
            key_vid = self._key_trie[key]
        except KeyError:
            raise KeyError(key)
        child_pos = self._find_child(0, key_vid)
        if child_pos is None:
            raise KeyError(key)
        return self._resolve(child_pos)

    def __iter__(self) -> Iterator[str]:
        for kid in self._root_list:
            yield self._key_trie.restore_key(self.elbl[kid - 1])

    def __contains__(self, key: str) -> bool:
        try:
            key_vid = self._key_trie[key]
        except KeyError:
            return False
        return self._find_child(0, key_vid) is not None

    def __len__(self) -> int:
        return len(self._root_list)

    def __repr__(self) -> str:
        """Return an interpretable representation of the CompactTree."""
        return f"CompactTree.from_dict({self.to_dict()!r})"

    def __str__(self) -> str:
        """Return a string representation like a Python dict."""
        return str(self.to_dict())

    def close(self) -> None:
        if self.f is not None:
            self.f.close()

    def __enter__(self) -> "CompactTree":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __reduce__(self) -> tuple:
        """Support pickle by using serialize/deserialize.
        
        Returns a tuple (callable, args) where callable(*args) reconstructs the object.
        """
        import io
        buf = io.BytesIO()
        # Serialize to an in-memory buffer
        keys_bytes = self._key_trie.to_bytes()
        val_bytes = self._val_trie.to_bytes()
        _cc = self._child_count
        if sys.byteorder != 'little':
            _cc = array.array('I', _cc); _cc.byteswap()
        child_count_bytes = _cc.tobytes()
        _vcol = self.vcol
        if sys.byteorder != 'little':
            _vcol = array.array('I', _vcol); _vcol.byteswap()
        vcol_bytes = _vcol.tobytes()
        _elbl = self.elbl
        if sys.byteorder != 'little':
            _elbl = array.array('I', _elbl); _elbl.byteswap()
        elbl_bytes = _elbl.tobytes()
        
        buf.write(b"CTree")
        buf.write(struct.pack("<Q", 5))
        buf.write(struct.pack(
            "<QQQQQQQ",
            len(keys_bytes), len(val_bytes), len(child_count_bytes),
            len(vcol_bytes), len(elbl_bytes),
            self._key_vocab_size,
            self._val_vocab_size,
        ))
        buf.write(keys_bytes)
        buf.write(val_bytes)
        buf.write(child_count_bytes)
        buf.write(vcol_bytes)
        buf.write(elbl_bytes)

        serialized = buf.getvalue()
        return (self._unpickle_from_bytes, (serialized,))

    @staticmethod
    def _unpickle_from_bytes(data: bytes) -> "CompactTree":
        """Reconstruct CompactTree from serialized bytes (used by pickle)."""
        import io
        f = io.BytesIO(data)
        
        # Read header
        magic, ver = struct.unpack("<5sQ", f.read(13))
        assert magic == b"CTree" and ver == 5, (
            f"Unsupported CompactTree format version {ver} (expected 5)"
        )

        (
            keys_len, val_len, child_count_len, vcol_len, elbl_len,
            _key_vocab_size, _val_vocab_size,
        ) = struct.unpack("<QQQQQQQ", f.read(56))

        # Read and parse keys MarisaTrie
        tree = CompactTree.__new__(CompactTree)
        keys_bytes = f.read(keys_len)
        tree._key_trie = MarisaTrie.from_bytes(
            keys_bytes,
            cache_size=_key_vocab_size or None,
        )

        # Read values MarisaTrie
        val_bytes = f.read(val_len)
        tree._val_trie = MarisaTrie.from_bytes(
            val_bytes,
            cache_size=_val_vocab_size or None,
        )

        tree._key_vocab_size = _key_vocab_size
        tree._val_vocab_size = _val_vocab_size

        # Reconstruct child_start from stored child_count array
        _cc = array.array('I'); _cc.frombytes(f.read(child_count_len))
        if sys.byteorder != 'little': _cc.byteswap()
        from itertools import accumulate as _acc
        n_nodes = len(_cc)
        tree._child_start = array.array('I', list(_acc(_cc, initial=1))[:n_nodes])
        tree._child_count = _cc

        # Read vcol and elbl sections
        _vcol = array.array('I'); _vcol.frombytes(f.read(vcol_len))
        if sys.byteorder != 'little': _vcol.byteswap()
        tree.vcol = _vcol
        _elbl = array.array('I'); _elbl.frombytes(f.read(elbl_len))
        if sys.byteorder != 'little': _elbl.byteswap()
        tree.elbl = _elbl

        # No file handles
        tree.fs = None
        tree.f = None
        tree.mm = None

        tree._root_list = tree._list_children(0)
        return tree
