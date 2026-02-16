import gzip
import struct
from typing import Any, BinaryIO, Iterator, Optional
from collections import deque

from bitarray import bitarray
from succinct.poppy import Poppy

from louds import LOUDS

_INTERNAL = 0xFFFFFFFF  # vcol sentinel for internal (non-leaf) nodes


class CompactTree:
    """Compact read-only nested dict backed by LOUDS + DAWG + edge labels.

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
                   values_out: list[str]) -> None:
        """Recursively collect all keys and leaf values."""
        for k, v in d.items():
            keys_out.add(k)
            if isinstance(v, dict):
                CompactTree._walk_dict(v, keys_out, values_out)
            else:
                values_out.append(str(v))

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
    def from_dict(cls, data: dict[str, Any]) -> "CompactTree":
        """Build a *CompactTree* entirely in memory from a nested Python dict.

        Keys must be strings.  Leaf values are stored as strings (non-string
        values are converted via ``str()``).
        """
        # 1. Collect vocabulary and leaf values
        all_keys: set[str] = set()
        all_values: list[str] = []
        cls._walk_dict(data, all_keys, all_values)

        # 2. Sorted, deduplicated key list
        sorted_keys = sorted(all_keys)
        key2vid: dict[str, int] = {k: i for i, k in enumerate(sorted_keys)}

        # 3. Value table (length-prefixed UTF-8 strings, deduplicated)
        unique_vals = list(dict.fromkeys(all_values))
        val2vid = {v: i for i, v in enumerate(unique_vals)}

        # 4. BFS -> LOUDS bits + vcol + elbl
        louds_bits = bitarray()
        vcol_buf = bytearray()
        elbl_buf = bytearray()
        queue: deque[Optional[dict]] = deque()

        def _emit_children(node: dict[str, Any]) -> None:
            for key in sorted(node.keys(), key=lambda k: key2vid[k]):
                louds_bits.append(True)
                elbl_buf.extend(struct.pack("<I", key2vid[key]))
                child = node[key]
                if isinstance(child, dict):
                    vcol_buf.extend(struct.pack("<I", _INTERNAL))
                    queue.append(child)
                else:
                    vcol_buf.extend(struct.pack("<I", val2vid[str(child)]))
                    queue.append(None)          # leaf placeholder
            louds_bits.append(False)

        _emit_children(data)
        while queue:
            item = queue.popleft()
            if item is None:
                louds_bits.append(False)        # leaf: degree-0
            else:
                _emit_children(item)

        # 5. Assemble the CompactTree object
        tree = cls.__new__(cls)
        tree.fs = None
        tree.f = None
        tree.mm = None
        tree._dawg_keys = sorted_keys
        tree._keys_buf = bytes(cls._pack_strings(sorted_keys))
        tree.val = bytes(cls._pack_strings(unique_vals))
        ba = louds_bits
        tree.louds = LOUDS(Poppy(ba), ba)
        tree.vcol = bytes(vcol_buf)
        tree.elbl = bytes(elbl_buf)
        tree._key2vid = key2vid
        tree._louds_root_list = tree._list_children(0)
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
                assert magic == b"CTree" and ver == 2

                # Read section lengths
                keys_len, val_len, louds_len, vcol_len, elbl_len = struct.unpack(
                    "<QQQQQ", f.read(40),
                )

                # Read and parse keys section
                self._keys_buf = f.read(keys_len)
                self._dawg_keys = self._unpack_strings(self._keys_buf)
                self._key2vid = {k: i for i, k in enumerate(self._dawg_keys)}

                # Read values section
                self.val = f.read(val_len)

                # Read and parse LOUDS section
                ba = bitarray()
                ba.frombytes(f.read(louds_len))
                self.louds = LOUDS(Poppy(ba), ba)

                # Read vcol and elbl sections
                self.vcol = f.read(vcol_len)
                self.elbl = f.read(elbl_len)

        # No file handles kept open
        self.fs = None
        self.f = None
        self.mm = None

        self._louds_root_list = self._list_children(0)

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
        
        keys_bytes = bytes(self._keys_buf)
        val_bytes = bytes(self.val)
        louds_bytes = self.louds._ba.tobytes()
        vcol_bytes = bytes(self.vcol)
        elbl_bytes = bytes(self.elbl)
        
        with fs.open(path, "wb") as raw_stream:
            with self._wrap_write_stream(raw_stream, compression) as f:
                f.write(b"CTree")
                f.write(struct.pack("<Q", 2))
                f.write(struct.pack(
                    "<QQQQQ",
                    len(keys_bytes), len(val_bytes), len(louds_bytes),
                    len(vcol_bytes), len(elbl_bytes),
                ))
                f.write(keys_bytes)
                f.write(val_bytes)
                f.write(louds_bytes)
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
                kv = struct.unpack(
                    "<I", self.elbl[(kid - 1) * 4:(kid - 1) * 4 + 4],
                )[0]
                key = self._dawg_keys[kv]
                vv = struct.unpack(
                    "<I", self.vcol[(kid - 1) * 4:(kid - 1) * 4 + 4],
                )[0]
                if vv == _INTERNAL:
                    out[key] = _build(self._list_children(kid))
                else:
                    out[key] = self._vid_to_str(vv)
            return out
        return _build(self._louds_root_list)

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _vid_to_str(self, vid: int) -> str:
        """Convert a value ID to the corresponding string."""
        off = 0
        for i in range(vid + 1):
            ln = struct.unpack("<I", self.val[off:off + 4])[0]
            off += 4
            if i == vid:
                val_slice = self.val[off:off + ln]
                # Handle both bytes and memoryview for compatibility
                if isinstance(val_slice, memoryview):
                    return val_slice.tobytes().decode("utf-8")
                return val_slice.decode("utf-8")
            off += ln
        raise IndexError(f"vid {vid} out of range")

    def _list_children(self, louds_pos: int) -> list[int]:
        """List all children of a node at a given LOUDS position."""
        kid = self.louds.first_child(louds_pos)
        out: list[int] = []
        while kid is not None:
            out.append(kid)
            kid = self.louds.next_sibling(kid)
        return out

    def _find_child(self, kids: list[int], key_vid: int) -> Optional[int]:
        """Return the LOUDS position of the child whose edge label == *key_vid*."""
        for kid in kids:
            if struct.unpack("<I", self.elbl[(kid - 1) * 4:
                                             (kid - 1) * 4 + 4])[0] == key_vid:
                return kid
        return None

    def _resolve(self, child_pos: int) -> "str | CompactTree._Node":
        """Resolve a child position to either a leaf string or a ``_Node``."""
        vv = struct.unpack("<I", self.vcol[(child_pos - 1) * 4:
                                           (child_pos - 1) * 4 + 4])[0]
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
            if key not in self.tree._key2vid:
                raise KeyError(key)
            kids = self._children()
            child_pos = self.tree._find_child(kids, self.tree._key2vid[key])
            if child_pos is None:
                raise KeyError(key)
            return self.tree._resolve(child_pos)

        def __iter__(self) -> Iterator[str]:
            for kid in self._children():
                kv = struct.unpack("<I", self.tree.elbl[(kid - 1) * 4:
                                                        (kid - 1) * 4 + 4])[0]
                yield self.tree._dawg_keys[kv]

        def __contains__(self, key: str) -> bool:
            if key not in self.tree._key2vid:
                return False
            return self.tree._find_child(
                self._children(), self.tree._key2vid[key],
            ) is not None

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
        if key not in self._key2vid:
            raise KeyError(key)
        child_pos = self._find_child(
            self._louds_root_list, self._key2vid[key],
        )
        if child_pos is None:
            raise KeyError(key)
        return self._resolve(child_pos)

    def __iter__(self) -> Iterator[str]:
        for kid in self._louds_root_list:
            kv = struct.unpack("<I", self.elbl[(kid - 1) * 4:
                                               (kid - 1) * 4 + 4])[0]
            yield self._dawg_keys[kv]

    def __contains__(self, key: str) -> bool:
        if key not in self._key2vid:
            return False
        return self._find_child(
            self._louds_root_list, self._key2vid[key],
        ) is not None

    def __len__(self) -> int:
        return len(self._louds_root_list)

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
        keys_bytes = bytes(self._keys_buf)
        val_bytes = bytes(self.val)
        louds_bytes = self.louds._ba.tobytes()
        vcol_bytes = bytes(self.vcol)
        elbl_bytes = bytes(self.elbl)
        
        buf.write(b"CTree")
        buf.write(struct.pack("<Q", 2))
        buf.write(struct.pack(
            "<QQQQQ",
            len(keys_bytes), len(val_bytes), len(louds_bytes),
            len(vcol_bytes), len(elbl_bytes),
        ))
        buf.write(keys_bytes)
        buf.write(val_bytes)
        buf.write(louds_bytes)
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
        assert magic == b"CTree" and ver == 2

        # Read section lengths
        keys_len, val_len, louds_len, vcol_len, elbl_len = struct.unpack(
            "<QQQQQ", f.read(40),
        )

        # Read and parse keys section
        tree = CompactTree.__new__(CompactTree)
        tree._keys_buf = f.read(keys_len)
        tree._dawg_keys = CompactTree._unpack_strings(tree._keys_buf)
        tree._key2vid = {k: i for i, k in enumerate(tree._dawg_keys)}

        # Read values section
        tree.val = f.read(val_len)

        # Read and parse LOUDS section
        ba = bitarray()
        ba.frombytes(f.read(louds_len))
        tree.louds = LOUDS(Poppy(ba), ba)

        # Read vcol and elbl sections
        tree.vcol = f.read(vcol_len)
        tree.elbl = f.read(elbl_len)

        # No file handles
        tree.fs = None
        tree.f = None
        tree.mm = None

        tree._louds_root_list = tree._list_children(0)
        return tree
