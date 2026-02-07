import struct
import typing
from typing import Optional, Union
from collections import deque

import fsspec
from bitarray import bitarray
from succinct.poppy import Poppy

_INTERNAL = 0xFFFFFFFF  # vcol sentinel for internal (non-leaf) nodes


class _LOUDS:
    """LOUDS (Level-Order Unary Degree Sequence) trie built on a Poppy bit vector.

    Node 0 is the root.  Every other node v (≥ 1) corresponds to the
    v-th 1-bit (1-based) in the bit string.
    """

    def __init__(self, bv: Poppy, ba: Optional[bitarray] = None) -> None:
        self._bv = bv
        self._ba: bitarray = ba if ba is not None else bitarray()

    def first_child(self, v: int) -> typing.Optional[int]:
        if v == 0:
            p = 0
        else:
            p = self._bv.select_zero(v - 1) + 1
        if p < len(self._bv) and self._bv[p]:
            return self._bv.rank(p)
        return None

    def next_sibling(self, v: int) -> typing.Optional[int]:
        pos = self._bv.select(v - 1)  # position of the 1-bit for node v
        nxt = pos + 1
        if nxt < len(self._bv) and self._bv[nxt]:
            return v + 1
        return None

class CompactTree:
    """Compact read-only nested dict backed by LOUDS + DAWG + edge labels.

    Two ways to create:

    * ``CompactTree.from_dict(d)`` – build in memory from a Python dict.
    * ``CompactTree(url)``         – deserialise from storage (v1 or v2).

    Persist with ``tree.serialize(url)`` (always writes v2).
    """

    # ------------------------------------------------------------------ #
    #  Construction helpers                                                #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _walk_dict(d: dict, keys_out: set[str],
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
    def _unpack_strings(mv: memoryview) -> list[str]:
        """Decode length-prefixed UTF-8 strings from a memoryview."""
        out: list[str] = []
        off = 0
        while off < len(mv):
            ln = struct.unpack("<I", mv[off:off + 4])[0]
            off += 4
            out.append(mv[off:off + ln].tobytes().decode("utf-8"))
            off += ln
        return out

    # ------------------------------------------------------------------ #
    #  Factory: from Python dict                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(cls, data: dict) -> "CompactTree":
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

        def _emit_children(node: dict) -> None:
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
        tree._keys_buf = memoryview(bytes(cls._pack_strings(sorted_keys)))
        tree.val = memoryview(bytes(cls._pack_strings(unique_vals)))
        ba = louds_bits
        tree.louds = _LOUDS(Poppy(ba), ba)
        tree.vcol = memoryview(bytes(vcol_buf))
        tree.elbl = memoryview(bytes(elbl_buf))
        tree._key2vid = key2vid
        tree._louds_root_list = tree._list_children(0)
        return tree

    # ------------------------------------------------------------------ #
    #  Factory: from file / deserialise                                    #
    # ------------------------------------------------------------------ #

    def __init__(self, url: str, storage_options: Optional[dict] = None):
        """Deserialise a *CompactTree* from storage."""
        self.fs = fsspec.filesystem(
            url.split("://")[0] if "://" in url else "file",
            **(storage_options or {}),
        )
        self.f = self.fs.open(url, "rb")
        self.mm = memoryview(self.f.read())
        off = 0
        magic, ver = struct.unpack("<5sQ", self.mm[off:off + 13])
        off += 13
        assert magic == b"CTree" and ver == 2
        keys_len, val_len, louds_len, vcol_len, elbl_len = struct.unpack(
            "<QQQQQ", self.mm[off:off + 40],
        )
        off += 40
        self._keys_buf = self.mm[off:off + keys_len]
        off += keys_len
        self.val = self.mm[off:off + val_len]
        off += val_len
        ba = bitarray()
        ba.frombytes(bytes(self.mm[off:off + louds_len]))
        off += louds_len
        self.vcol = self.mm[off:off + vcol_len]
        off += vcol_len
        self.elbl = self.mm[off:off + elbl_len]
        self.louds = _LOUDS(Poppy(ba), ba)
        self._dawg_keys = self._unpack_strings(self._keys_buf)
        self._key2vid = {k: i for i, k in enumerate(self._dawg_keys)}
        self._louds_root_list = self._list_children(0)

    # ------------------------------------------------------------------ #
    #  Serialise (always v2)                                               #
    # ------------------------------------------------------------------ #

    def serialize(self, url: str, storage_options: Optional[dict] = None) -> None:
        """Write the tree to *url* in v2 binary format."""
        fs = fsspec.filesystem(
            url.split("://")[0] if "://" in url else "file",
            **(storage_options or {}),
        )
        keys_bytes = bytes(self._keys_buf)
        val_bytes = bytes(self.val)
        louds_bytes = self.louds._ba.tobytes()
        vcol_bytes = bytes(self.vcol)
        elbl_bytes = bytes(self.elbl)
        with fs.open(url, "wb") as f:
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
                return self.val[off:off + ln].tobytes().decode("utf-8")
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
    #  _Node  (nested dict interface for sub-trees)                        #
    # ------------------------------------------------------------------ #

    class _Node:
        def __init__(self, tree: "CompactTree", pos: int):
            self.tree = tree
            self.pos = pos

        def _children(self) -> list[int]:
            """List all children of the current node."""
            return self.tree._list_children(self.pos)

        def __getitem__(self, key: str) -> typing.Any:
            if key not in self.tree._key2vid:
                raise KeyError(key)
            kids = self._children()
            child_pos = self.tree._find_child(kids, self.tree._key2vid[key])
            if child_pos is None:
                raise KeyError(key)
            return self.tree._resolve(child_pos)

        def __iter__(self) -> typing.Iterator[str]:
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

    # ------------------------------------------------------------------ #
    #  Mapping-like interface (root level)                                 #
    # ------------------------------------------------------------------ #

    def __getitem__(self, key: str) -> typing.Any:
        if key not in self._key2vid:
            raise KeyError(key)
        child_pos = self._find_child(
            self._louds_root_list, self._key2vid[key],
        )
        if child_pos is None:
            raise KeyError(key)
        return self._resolve(child_pos)

    def __iter__(self) -> typing.Iterator[str]:
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

    def close(self) -> None:
        if self.f is not None:
            self.f.close()

    def __enter__(self) -> "CompactTree":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()