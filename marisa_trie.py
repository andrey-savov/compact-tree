"""MARISA trie with minimal perfect hashing for word-to-index mapping."""

import gzip
import struct
from typing import BinaryIO, Iterable, Optional
from collections import deque

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
            self._tail_trie: Optional[MarisaTrie] = None
            self._root_is_terminal = False
            return
        
        # Build intermediate trie structure
        root = self._build_intermediate_trie(unique)
        
        # Check if root is terminal (empty string in trie)
        self._root_is_terminal = "" in root
        
        # Apply path compression and convert to LOUDS
        self._build_louds(root)

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
            # Mark terminal with empty string sentinel
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
        
        # BFS to build LOUDS topology and collect node metadata
        def _emit_children(inode: dict) -> None:
            """Emit children of an intermediate node, applying path compression."""
            children = [(k, v) for k, v in inode.items() if k != ""]
            
            if not children:
                # Leaf node (terminal only, no children)
                louds_bits.append(False)
                return
            
            for char, child_node in children:
                # Apply path compression: if child has exactly one non-terminal child,
                # merge the edge labels
                label = char
                current = child_node
                while True:
                    # Check if current is terminal
                    is_term = "" in current
                    # Get non-terminal children
                    next_children = [(k, v) for k, v in current.items() if k != ""]
                    
                    # Stop if terminal or has multiple/zero children
                    if is_term or len(next_children) != 1:
                        break
                    
                    # Merge the single child's edge
                    next_char, next_node = next_children[0]
                    label += next_char
                    current = next_node
                
                # Emit the compressed edge
                louds_bits.append(True)
                is_terminal = "" in current
                nodes_metadata.append((current, label, is_terminal))
                node_queue.append(current)
            
            louds_bits.append(False)
        
        # Emit root's children
        _emit_children(root)
        
        # Process queue
        while node_queue:
            node = node_queue.popleft()
            _emit_children(node)
        
        # Pack labels (length-prefixed UTF-8 per node)
        for _, label, _ in nodes_metadata:
            label_bytes = label.encode("utf-8")
            labels_buf.extend(struct.pack("<I", len(label_bytes)))
            labels_buf.extend(label_bytes)
        
        # Terminal bits (one per node, aligned with nodes_metadata)
        for _, _, is_term in nodes_metadata:
            terminal_bits.append(is_term)
        
        # Store LOUDS and labels
        self._louds = LOUDS(Poppy(louds_bits), louds_bits)
        self._labels = bytes(labels_buf)
        self._terminal = terminal_bits
        
        # Compute subtree counts bottom-up
        self._compute_counts()
        
        # No tail trie for now (simplified - full recursive MARISA not implemented)
        self._tail_trie = None

    def _compute_counts(self) -> None:
        """Compute subtree word counts for each node (bottom-up)."""
        num_nodes = len(self._terminal)
        counts = [0] * num_nodes
        
        # Bottom-up traversal: process nodes in reverse LOUDS order
        for node_idx in range(num_nodes - 1, -1, -1):
            # Node indices are 1-based in LOUDS, but _terminal is 0-indexed
            louds_node = node_idx + 1
            
            # Count starts with 1 if terminal
            count = 1 if self._terminal[node_idx] else 0
            
            # Add counts from all children
            child = self._louds.first_child(louds_node)
            while child is not None:
                child_idx = child - 1  # Convert to 0-indexed
                count += counts[child_idx]
                child = self._louds.next_sibling(child)
            
            counts[node_idx] = count
        
        # Pack counts as uint32
        counts_buf = bytearray()
        for c in counts:
            counts_buf.extend(struct.pack("<I", c))
        self._counts = bytes(counts_buf)

    # ------------------------------------------------------------------ #
    #  Forward lookup: word -> index                                       #
    # ------------------------------------------------------------------ #

    def index(self, word: str) -> int:
        """Return the unique index for a word in [0, N).
        
        Args:
            word: The word to look up.
            
        Returns:
            Unique integer in [0, N).
            
        Raises:
            KeyError: If word is not in the trie.
        """
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
        offset = 0
        for i in range(node_idx + 1):
            length = struct.unpack("<I", self._labels[offset:offset + 4])[0]
            offset += 4
            if i == node_idx:
                return self._labels[offset:offset + length].decode("utf-8")
            offset += length
        return ""

    def _get_count(self, node_idx: int) -> int:
        """Get the subtree word count for a node (0-indexed)."""
        offset = node_idx * 4
        return struct.unpack("<I", self._counts[offset:offset + 4])[0]

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
