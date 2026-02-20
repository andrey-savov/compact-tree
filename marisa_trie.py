"""MARISA trie with minimal perfect hashing for word-to-index mapping."""

import gzip
import struct
from typing import BinaryIO, Iterable, Optional
from collections import deque
from functools import lru_cache

from bitarray import bitarray
from succinct.poppy import Poppy

from louds import LOUDS


class MarisaTrie:
    """Compact read-only word-to-index mapping backed by LOUDS + MPH.
    
    Ingests an iterable of strings, deduplicates, builds a MARISA trie
    with subtree word counts for minimal perfect hashing. Supports:
    
    * ``index(word)`` - get dense unique index in [0, N)
    * ``restore_key(idx)`` - get word from index
    * Serialize/deserialize to disk
    
    Indices are dense [0, N) but in arbitrary order (determined by
    trie structure, not lexicographic or insertion order).
    """

    # ------------------------------------------------------------------ #
    #  Construction                                                        #
    # ------------------------------------------------------------------ #

    def __init__(self, words: Iterable[str]) -> None:
        """Build a MarisaTrie from an iterable of strings.
        
        Args:
            words: Iterable of strings. Duplicates are silently removed.
        """
        # Deduplicate and collect unique words
        unique = list(dict.fromkeys(words))
        self._n = len(unique)
        
        if self._n == 0:
            # Empty trie
            self._louds = LOUDS(Poppy(bitarray([False])), bitarray([False]))
            self._labels = b""
            self._terminal = bitarray()
            self._counts = b""
            self._label_offsets = []
            self._tail_trie: Optional[MarisaTrie] = None
            self._root_is_terminal = False
            # Install C-level LRU cache as instance attribute (shadows class method)
            self.index = lru_cache(maxsize=4096)(self._index_uncached)
            return
        
        # Build intermediate trie structure
        root = self._build_intermediate_trie(unique)
        
        # Check if root is terminal (empty string in trie)
        self._root_is_terminal = "" in root
        
        # Apply path compression and convert to LOUDS
        self._build_louds(root)

        # Install C-level LRU cache as instance attribute (shadows class method)
        self.index = lru_cache(maxsize=4096)(self._index_uncached)

    def _build_intermediate_trie(self, words: list[str]) -> dict:
        """Build an intermediate dict-of-dicts trie from unique words.
        
        Returns:
            Root node of the trie. Each node is a dict mapping char -> child_node.
            Terminal nodes are marked with a sentinel key "".
        """
        root: dict = {}
        for word in words:
            node = root
            for char in word:
                if char not in node:
                    node[char] = {}
                node = node[char]
            node[""] = True
        return root

    def _build_louds(self, root: dict) -> None:
        """Convert intermediate trie to LOUDS with path compression and subtree counts.
        
        Args:
            root: Root of the intermediate trie (dict-of-dicts).
        """
        louds_bits = bitarray()
        labels_buf = bytearray()
        terminal_bits = bitarray()
        node_queue: deque = deque()
        
        # Node metadata: (intermediate_node, compressed_label, is_terminal)
        # We'll build a list of nodes in LOUDS order
        nodes_metadata: list = []
        
        # Track parent-child relationships for efficient count computation
        # children_map[parent_idx] = [child_idx1, child_idx2, ...]
        children_map: dict[int, list[int]] = {}
        next_node_idx = 0
        
        # BFS to build LOUDS topology and collect node metadata
        def _emit_children(inode: dict, parent_idx: int) -> None:
            """Emit children of an intermediate node, applying path compression."""
            nonlocal next_node_idx
            
            # Opt 2: iterate inline — no upfront list allocation per node
            child_indices = []
            for char, child_node in inode.items():
                if char == "":
                    continue
                
                # Opt 1: path compression without allocating a list each iteration
                label = char
                current = child_node
                while True:
                    has_term = "" in current
                    # len(current) - has_term gives non-terminal child count, no list alloc
                    if has_term or len(current) - has_term != 1:
                        break
                    # Single non-terminal child: grab it with one short loop
                    for k, v in current.items():
                        if k != "":
                            label += k
                            current = v
                            break
                
                louds_bits.append(True)
                child_idx = next_node_idx
                next_node_idx += 1
                child_indices.append(child_idx)
                nodes_metadata.append((current, label, "" in current))
                node_queue.append((current, child_idx))
            
            # Always emit the LOUDS 0-terminator for this node
            louds_bits.append(False)
            if child_indices:
                children_map[parent_idx] = child_indices
        
        # Emit root's children (parent is -1 for root)
        _emit_children(root, -1)
        
        # Process queue
        while node_queue:
            node, parent_idx = node_queue.popleft()
            _emit_children(node, parent_idx)
        
        # Opt 5: single pass over nodes_metadata for labels + terminal bits
        label_offsets = []
        offset = 0
        for _, label, is_term in nodes_metadata:
            label_bytes = label.encode("utf-8")
            lb_len = len(label_bytes)
            label_offsets.append(offset)
            labels_buf.extend(struct.pack("<I", lb_len))
            labels_buf.extend(label_bytes)
            offset += 4 + lb_len  # arithmetic, avoids len(labels_buf) call
            terminal_bits.append(is_term)
        
        # Store LOUDS, labels, and label offsets
        self._louds = LOUDS(Poppy(louds_bits), louds_bits)
        self._labels = bytes(labels_buf)
        self._terminal = terminal_bits
        self._label_offsets = label_offsets  # For O(1) label access
        
        # Compute subtree counts bottom-up using direct child relationships
        self._compute_counts_optimized(children_map)

        # Build fast {word: idx} mapping from the intermediate structures
        # (plain Python dicts, zero rank/select calls).  Stored temporarily
        # and consumed by the first to_dict() call; freed immediately after
        # so it does not inflate run-time memory.
        self._word_to_idx: dict[str, int] = (
            self._build_word_index_from_intermediate(nodes_metadata, children_map)
        )

        # No tail trie for now (simplified - full recursive MARISA not implemented)
        self._tail_trie = None

    def _compute_counts_optimized(self, children_map: dict[int, list[int]]) -> None:
        """Compute subtree word counts for each node (bottom-up) using direct child mapping.
        
        Args:
            children_map: Mapping from parent node index to list of child indices.
        """
        num_nodes = len(self._terminal)
        counts = [0] * num_nodes
        
        # Opt 6: use dict.get to avoid double-lookup ("in" + "[]") per node
        _get = children_map.get
        terminal = self._terminal
        
        # Bottom-up traversal: process nodes in reverse order
        for node_idx in range(num_nodes - 1, -1, -1):
            count = 1 if terminal[node_idx] else 0
            children = _get(node_idx)
            if children:
                for child_idx in children:
                    count += counts[child_idx]
            counts[node_idx] = count
        
        # Opt 4: single bulk struct.pack instead of N individual calls
        self._counts = struct.pack(f"<{num_nodes}I", *counts)

    def _build_word_index_from_intermediate(
        self,
        nodes_metadata: list,
        children_map: dict[int, list[int]],
    ) -> "dict[str, int]":
        """Build ``{word: idx}`` from intermediate BFS metadata without any
        LOUDS traversal.

        Uses the same pre-order index assignment as ``_index_uncached``:
        a node's terminal (if any) is numbered before its children,
        children ordered left-to-right.

        Args:
            nodes_metadata: List of ``(inode, label, is_terminal)`` tuples in
                BFS order, as produced by ``_build_louds``.
            children_map: Maps each node index to its ordered child indices;
                key ``-1`` holds the root's immediate children.
        """
        result: dict[str, int] = {}
        idx = 0

        if self._root_is_terminal:
            result[""] = 0
            idx = 1

        # DFS stack: (node_idx, accumulated_prefix)
        _get = children_map.get
        stack: list[tuple[int, str]] = []
        for child_idx in reversed(_get(-1, [])):
            stack.append((child_idx, nodes_metadata[child_idx][1]))

        while stack:
            node_idx, prefix = stack.pop()
            _, _, is_terminal = nodes_metadata[node_idx]
            if is_terminal:
                result[prefix] = idx
                idx += 1
            children = _get(node_idx)
            if children:
                for child_idx in reversed(children):
                    stack.append((child_idx, prefix + nodes_metadata[child_idx][1]))

        return result

    # ------------------------------------------------------------------ #
    #  Forward lookup: word -> index                                       #
    # ------------------------------------------------------------------ #

    def index(self, word: str) -> int:
        """Return the unique index for a word in [0, N).

        At runtime this method is shadowed by a per-instance
        ``lru_cache``-wrapped version of ``_index_uncached`` installed in
        ``__init__`` and ``from_bytes``, so it is never actually called.
        It is kept here solely as readable interface documentation.

        Args:
            word: The word to look up.

        Returns:
            Unique integer in [0, N).

        Raises:
            KeyError: If word is not in the trie.
        """
        return self._index_uncached(word)
    
    def _index_uncached(self, word: str) -> int:
        """Uncached implementation of index lookup."""
        if self._n == 0:
            raise KeyError(word)
        
        # Special case: empty string
        if word == "":
            if self._root_is_terminal:
                return 0
            raise KeyError(word)
        
        idx = 0
        remaining = word
        node = 0  # Root in LOUDS is node 0
        
        # If root is terminal, it occupies index 0
        if self._root_is_terminal:
            idx = 1
        
        while True:
            # Get children of current node
            child = self._louds.first_child(node)
            
            if child is None:
                # No children - must be at a terminal
                if remaining == "" and node > 0 and self._terminal[node - 1]:
                    return idx
                raise KeyError(word)
            
            # Try to match remaining against children's labels
            matched = False
            while child is not None:
                child_idx = child - 1
                label = self._get_label(child_idx)
                
                if remaining.startswith(label):
                    # Found matching child
                    # Add counts of all previous siblings
                    sibling = self._louds.first_child(node)
                    while sibling != child:
                        assert sibling is not None  # Won't be None since child exists
                        sibling_idx = sibling - 1
                        idx += self._get_count(sibling_idx)
                        sibling = self._louds.next_sibling(sibling)
                    
                    # If current node is terminal (before descending), add 1
                    if node > 0 and self._terminal[node - 1]:
                        idx += 1
                    
                    # Descend to child
                    remaining = remaining[len(label):]
                    node = child
                    matched = True
                    break
                
                child = self._louds.next_sibling(child)
            
            if not matched:
                raise KeyError(word)
            
            # Check if we've consumed the entire word
            if remaining == "":
                # Must be at a terminal node
                if self._terminal[node - 1]:
                    return idx
                raise KeyError(word)

    # ------------------------------------------------------------------ #
    #  Reverse lookup: index -> word                                       #
    # ------------------------------------------------------------------ #

    def restore_key(self, idx: int) -> str:
        """Return the word corresponding to an index.
        
        Args:
            idx: Index in [0, N).
            
        Returns:
            The word at that index.
            
        Raises:
            IndexError: If idx is out of range.
        """
        if idx < 0 or idx >= self._n:
            raise IndexError(f"Index {idx} out of range [0, {self._n})")
        
        # Special case: if root is terminal and idx is 0
        if self._root_is_terminal and idx == 0:
            return ""
        
        path = []
        node = 0  # Root
        target = idx
        
        # If root is terminal, it occupies index 0
        if self._root_is_terminal:
            target -= 1
        
        while True:
            # If current node is terminal and target is 0, we found it
            if node > 0 and self._terminal[node - 1]:
                if target == 0:
                    return "".join(path)
                target -= 1
            
            # Get children
            child = self._louds.first_child(node)
            if child is None:
                # No children but we haven't found it
                raise IndexError(f"Index {idx} out of range")
            
            # Find which child's subtree contains the target
            while child is not None:
                child_idx = child - 1
                count = self._get_count(child_idx)
                
                if target < count:
                    # Target is in this child's subtree
                    label = self._get_label(child_idx)
                    path.append(label)
                    node = child
                    break
                
                # Skip this subtree
                target -= count
                child = self._louds.next_sibling(child)
            else:
                # No child contained the target
                raise IndexError(f"Index {idx} out of range")

    # ------------------------------------------------------------------ #
    #  Helper methods                                                      #
    # ------------------------------------------------------------------ #

    def _get_label(self, node_idx: int) -> str:
        """Get the edge label for a node (0-indexed)."""
        # Use pre-computed offsets for O(1) access
        offset = self._label_offsets[node_idx]
        length = struct.unpack("<I", self._labels[offset:offset + 4])[0]
        return self._labels[offset + 4:offset + 4 + length].decode("utf-8")

    def _get_count(self, node_idx: int) -> int:
        """Get the subtree word count for a node (0-indexed)."""
        offset = node_idx * 4
        return struct.unpack("<I", self._counts[offset:offset + 4])[0]

    # ------------------------------------------------------------------ #
    #  Bulk enumeration                                                    #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, int]:
        """Return ``{word: index}`` for every word in O(N).

        **First call after construction** returns the index built cheaply
        from the intermediate trie (no LOUDS traversal) and immediately
        frees it from the instance to keep run-time memory minimal.

        **Subsequent calls**, or calls on a trie loaded from disk via
        ``from_bytes``, fall back to a single DFS over the LOUDS bit vector.
        """
        # Fast path: intermediate index present (set by _build_louds).
        # Pop it so the memory is released right after the caller is done.
        cached: dict[str, int] | None = self.__dict__.pop("_word_to_idx", None)
        if cached is not None:
            return cached

        # Fallback: single DFS over the LOUDS bit vector.
        result: dict[str, int] = {}
        if self._n == 0:
            return result

        idx = 0
        stack: list[tuple[int, str]] = [(0, "")]

        while stack:
            node, prefix = stack.pop()

            if node == 0:
                if self._root_is_terminal:
                    result[prefix] = idx
                    idx += 1
            else:
                if self._terminal[node - 1]:
                    result[prefix] = idx
                    idx += 1

            children: list[tuple[int, str]] = []
            child = self._louds.first_child(node)
            while child is not None:
                label = self._get_label(child - 1)
                children.append((child, prefix + label))
                child = self._louds.next_sibling(child)

            for c in reversed(children):
                stack.append(c)

        return result

    # ------------------------------------------------------------------ #
    #  Dunder methods                                                      #
    # ------------------------------------------------------------------ #

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
        """Return an interpretable representation."""
        return f"MarisaTrie({self._n} words)"

    # ------------------------------------------------------------------ #
    #  Serialization helpers                                               #
    # ------------------------------------------------------------------ #

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
    #  Serialization                                                       #
    # ------------------------------------------------------------------ #

    def to_bytes(self) -> bytes:
        """Serialize the trie to bytes.
        
        Returns:
            Binary representation of the trie.
        """
        import io
        buf = io.BytesIO()
        
        # Magic and version
        buf.write(b"MTrie")
        buf.write(struct.pack("<Q", 1))  # Version 1
        
        # Serialize components
        louds_bytes = self._louds._ba.tobytes()
        labels_bytes = bytes(self._labels)
        terminal_bytes = self._terminal.tobytes()
        counts_bytes = bytes(self._counts)
        
        # Write metadata
        buf.write(struct.pack("<Q", self._n))
        buf.write(struct.pack("<?", self._root_is_terminal))  # bool flag
        buf.write(struct.pack("<QQQQ",
                             len(louds_bytes),
                             len(labels_bytes),
                             len(terminal_bytes),
                             len(counts_bytes)))
        
        # Write sections
        buf.write(louds_bytes)
        buf.write(labels_bytes)
        buf.write(terminal_bytes)
        buf.write(counts_bytes)
        
        return buf.getvalue()

    @classmethod
    def from_bytes(cls, data: bytes) -> "MarisaTrie":
        """Deserialize a trie from bytes.
        
        Args:
            data: Binary representation from to_bytes().
            
        Returns:
            Reconstructed MarisaTrie.
        """
        import io
        f = io.BytesIO(data)
        
        # Read header
        magic, ver = struct.unpack("<5sQ", f.read(13))
        assert magic == b"MTrie" and ver == 1, f"Invalid format: {magic!r} v{ver}"
        
        # Read metadata
        n = struct.unpack("<Q", f.read(8))[0]
        root_is_terminal = struct.unpack("<?", f.read(1))[0]
        louds_len, labels_len, terminal_len, counts_len = struct.unpack("<QQQQ", f.read(32))
        
        # Read sections
        louds_bytes = f.read(louds_len)
        labels_bytes = f.read(labels_len)
        terminal_bytes = f.read(terminal_len)
        counts_bytes = f.read(counts_len)
        
        # Reconstruct
        trie = cls.__new__(cls)
        trie._n = n
        trie._root_is_terminal = root_is_terminal
        
        louds_ba = bitarray()
        louds_ba.frombytes(louds_bytes)
        trie._louds = LOUDS(Poppy(louds_ba), louds_ba)
        
        trie._labels = labels_bytes
        
        terminal_ba = bitarray()
        terminal_ba.frombytes(terminal_bytes)
        trie._terminal = terminal_ba
        
        trie._counts = counts_bytes
        trie._tail_trie = None
        
        # Reconstruct label offsets from labels_bytes
        trie._label_offsets = []
        offset = 0
        num_labels = len(terminal_ba)  # One label per terminal bit
        for i in range(num_labels):
            trie._label_offsets.append(offset)
            if offset >= len(labels_bytes):
                break
            length = struct.unpack("<I", labels_bytes[offset:offset + 4])[0]
            offset += 4 + length
        
        # Install C-level LRU cache as instance attribute (shadows class method)
        trie.index = lru_cache(maxsize=4096)(trie._index_uncached)

        return trie

    def serialize(self, url: str, storage_options: Optional[dict] = None) -> None:
        """Write the trie to a file.
        
        Args:
            url: Path or URL where to save.
            storage_options: fsspec options. Set compression='gzip' for gzip.
        """
        from fsspec.core import url_to_fs
        
        opts = storage_options or {}
        compression = opts.get("compression")
        fs, path = url_to_fs(url, **opts)
        
        data = self.to_bytes()
        
        with fs.open(path, "wb") as raw_stream:
            with self._wrap_write_stream(raw_stream, compression) as f:
                f.write(data)

    @classmethod
    def load(cls, url: str, storage_options: Optional[dict] = None) -> "MarisaTrie":
        """Load a trie from a file.
        
        Args:
            url: Path or URL to load from.
            storage_options: fsspec options. Set compression='gzip' if compressed.
            
        Returns:
            Loaded MarisaTrie.
        """
        from fsspec.core import url_to_fs
        
        opts = storage_options or {}
        compression = opts.get("compression")
        fs, path = url_to_fs(url, **opts)
        
        with fs.open(path, "rb") as raw_stream:
            with cls._wrap_read_stream(raw_stream, compression) as f:
                data = f.read()
        
        return cls.from_bytes(data)

    def __reduce__(self) -> tuple:
        """Support pickle by serializing to bytes."""
        return (self.from_bytes, (self.to_bytes(),))
