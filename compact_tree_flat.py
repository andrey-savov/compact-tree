from __future__ import annotations

"""CompactTreeFlat — flat tuple-keyed store backed by a single value MarisaTrie.

Unlike :class:`~compact_tree.CompactTree`, which preserves the full
nested-dict structure and returns intermediate :class:`_Node` proxies on
partial key access, ``CompactTreeFlat`` discards the tree hierarchy and
stores every leaf as a flat ``tuple[str, ...]`` → ``val_id`` mapping.

The only lookup interface is :meth:`get_path`, which takes all path
components at once and returns the leaf value directly — no intermediate
objects, no per-level ``__getitem__`` dispatch.

This is optimal when:

* The caller always knows the full path depth at call time.
* Intermediate navigation is not required.
* Memory must be minimised for deeply nested dicts with large, mostly-unique
  leaf-value vocabularies (only *values* are DAWG-compressed; keys live in a
  plain Python ``dict``).

Mutually exclusive with ``shared_trie`` (there is no key trie to share).

Usage::

    from compact_tree_flat import CompactTreeFlat

    d = {"a": {"x": "1"}, "b": {"x": "2", "y": "3"}}
    tree = CompactTreeFlat.from_dict(d)

    tree.get_path("a", "x")          # "1"
    tree.get_path("b", "y")          # "3"
    ("b", "x") in tree               # True — __contains__ takes a tuple
    len(tree)                         # 3  (total leaf paths)
    tree.to_dict()                   # {"a": {"x": "1"}, "b": {"x": "2", "y": "3"}}

    tree.serialize("tree.ctflat")
    tree2 = CompactTreeFlat("tree.ctflat")

    tree.serialize("tree.ctflat.gz", compression="gzip")
    tree3 = CompactTreeFlat("tree.ctflat.gz", compression="gzip")

    import pickle
    tree4 = pickle.loads(pickle.dumps(tree))
"""

import gzip
import io
import struct
import sys
from typing import Any, BinaryIO

from marisa_trie import MarisaTrie

_MAGIC = b"CTFlt"
_VERSION = 1


# ---------------------------------------------------------------------------
# Stream helpers (intentionally self-contained — no import from compact_tree)
# ---------------------------------------------------------------------------

def _wrap_read_stream(stream: BinaryIO, compression: str | None) -> BinaryIO:
    if compression == "gzip":
        return gzip.open(stream, "rb")  # type: ignore[return-value]
    if compression is None:
        return stream
    raise ValueError(f"Unsupported compression: {compression!r}")


def _wrap_write_stream(stream: BinaryIO, compression: str | None) -> BinaryIO:
    if compression == "gzip":
        return gzip.open(stream, "wb", compresslevel=9)  # type: ignore[return-value]
    if compression is None:
        return stream
    raise ValueError(f"Unsupported compression: {compression!r}")


# ---------------------------------------------------------------------------
# Recursive dict walker
# ---------------------------------------------------------------------------

def _walk_flat(
    node: dict[str, Any],
    prefix: tuple[str, ...],
    pairs: list[tuple[tuple[str, ...], str]],
    unique_values: set[str],
) -> None:
    """Recursively collect (full_path_tuple, leaf_str) pairs and leaf values."""
    for k, v in node.items():
        path = prefix + (k,)
        if type(v) is dict:
            _walk_flat(v, path, pairs, unique_values)
        else:
            leaf = v if type(v) is str else str(v)
            pairs.append((path, leaf))
            unique_values.add(leaf)


# ---------------------------------------------------------------------------
# Binary serialisation helpers
# ---------------------------------------------------------------------------

def _encode_paths(key_dict: dict[tuple[str, ...], int]) -> bytes:
    """Encode all (path, val_id) pairs into a compact byte buffer.

    Layout per entry::

        uint8    n           — number of path components (≤ 255)
        for each component:
            uint32 LE  comp_byte_len
            comp_byte_len bytes  (UTF-8)
        uint32 LE  val_id
    """
    buf = bytearray()
    for path in sorted(key_dict):
        vid = key_dict[path]
        n = len(path)
        if n > 255:
            raise ValueError(
                f"Path depth {n} exceeds maximum of 255: {path!r}"
            )
        buf.append(n)
        for comp in path:
            enc = comp.encode("utf-8")
            buf += struct.pack("<I", len(enc))
            buf += enc
        buf += struct.pack("<I", vid)
    return bytes(buf)


def _decode_paths(data: bytes) -> dict[tuple[str, ...], int]:
    """Decode a byte buffer produced by :func:`_encode_paths`."""
    out: dict[tuple[str, ...], int] = {}
    off = 0
    total = len(data)
    while off < total:
        n = data[off]
        off += 1
        components: list[str] = []
        for _ in range(n):
            comp_len = struct.unpack("<I", data[off: off + 4])[0]
            off += 4
            components.append(data[off: off + comp_len].decode("utf-8"))
            off += comp_len
        vid = struct.unpack("<I", data[off: off + 4])[0]
        off += 4
        out[tuple(components)] = vid
    return out


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class CompactTreeFlat:
    """Flat tuple-keyed compact store.  See module docstring for details."""

    # ------------------------------------------------------------------ #
    #  Factory: from Python dict                                           #
    # ------------------------------------------------------------------ #

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        *,
        vocabulary_size: int | None = 0,
    ) -> CompactTreeFlat:
        """Build a *CompactTreeFlat* in memory from a nested Python dict.

        Keys must be strings.  Leaf values are stored as strings (non-string
        values are converted via ``str()``).

        Args:
            vocabulary_size: Controls the ``lru_cache`` capacity on
                ``_val_trie``.  ``0`` (default) disables the LRU cache
                entirely.  Pass ``None`` to auto-size the cache to the full
                value vocabulary, or a positive int to cap it.
        """
        if vocabulary_size is not None:
            if isinstance(vocabulary_size, bool) or not isinstance(vocabulary_size, int):
                raise TypeError(
                    "vocabulary_size must be an int (or None); got "
                    f"{type(vocabulary_size).__name__}: {vocabulary_size!r}"
                )
            if vocabulary_size < 0:
                raise ValueError(
                    f"vocabulary_size must be 0 (disable), None (auto-size), "
                    f"or a positive integer; got {vocabulary_size!r}"
                )

        pairs: list[tuple[tuple[str, ...], str]] = []
        unique_values: set[str] = set()
        _walk_flat(data, (), pairs, unique_values)

        val_cache_size: int = (
            vocabulary_size
            if vocabulary_size is not None
            else len(unique_values)
        )
        val_trie = MarisaTrie(unique_values, cache_size=val_cache_size)
        val_id_map = val_trie.to_dict()

        key_dict: dict[tuple[str, ...], int] = {
            path: val_id_map[leaf] for path, leaf in pairs
        }

        obj = cls.__new__(cls)
        obj._key_dict = key_dict
        obj._val_trie = val_trie
        obj._val_vocab_size: int = val_cache_size
        return obj

    # ------------------------------------------------------------------ #
    #  Deserialise from storage                                            #
    # ------------------------------------------------------------------ #

    def __init__(
        self,
        url: str,
        *,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Load a *CompactTreeFlat* from *url*.

        Args:
            url: Path or URL to a file written by :meth:`serialize`.
            compression: ``None`` (default) or ``'gzip'``.
            **kwargs: Additional keyword arguments forwarded to
                ``fsspec.url_to_fs``.
        """
        from fsspec.core import url_to_fs

        fs, path = url_to_fs(url, **kwargs)
        with fs.open(path, "rb") as raw_stream:
            with _wrap_read_stream(raw_stream, compression) as f:
                _magic, _ver = struct.unpack("<5sQ", f.read(13))
                assert _magic == _MAGIC, f"Bad magic: {_magic!r}"
                assert _ver == _VERSION, f"Unsupported CTFlt version: {_ver}"

                # 4 × uint64 LE header
                n_paths, val_trie_len, paths_buf_len, val_vocab_size = \
                    struct.unpack("<QQQQ", f.read(32))

                val_bytes = f.read(val_trie_len)
                paths_bytes = f.read(paths_buf_len)

        self._val_trie = MarisaTrie.from_bytes(val_bytes, cache_size=val_vocab_size)
        self._val_vocab_size = val_vocab_size
        self._key_dict = _decode_paths(paths_bytes)
        assert len(self._key_dict) == n_paths, (
            f"Expected {n_paths} paths, decoded {len(self._key_dict)}"
        )

    # ------------------------------------------------------------------ #
    #  Serialise to storage                                                #
    # ------------------------------------------------------------------ #

    def serialize(
        self,
        url: str,
        *,
        compression: str | None = None,
        **kwargs: Any,
    ) -> None:
        """Write the tree to *url* in CTFlt v1 binary format.

        Args:
            url: Path or URL where to save the tree.
            compression: Optional compression.  Pass ``'gzip'`` for gzip
                output (level 9).
            **kwargs: Forwarded to ``fsspec.url_to_fs``.
        """
        from fsspec.core import url_to_fs

        val_bytes = self._val_trie.to_bytes()
        paths_bytes = _encode_paths(self._key_dict)

        fs, path = url_to_fs(url, **kwargs)
        with fs.open(path, "wb") as raw_stream:
            with _wrap_write_stream(raw_stream, compression) as f:
                f.write(_MAGIC)
                f.write(struct.pack("<Q", _VERSION))
                f.write(struct.pack(
                    "<QQQQ",
                    len(self._key_dict),
                    len(val_bytes),
                    len(paths_bytes),
                    self._val_vocab_size,
                ))
                f.write(val_bytes)
                f.write(paths_bytes)

    # ------------------------------------------------------------------ #
    #  Core lookup                                                         #
    # ------------------------------------------------------------------ #

    def get_path(self, *keys: str) -> str:
        """Return the leaf value at the given path.

        Args:
            *keys: One or more string path components.

        Returns:
            The leaf string at ``(*keys,)``.

        Raises:
            KeyError: If no leaf exists at the given path.
            TypeError: If no keys are provided.
        """
        if not keys:
            raise TypeError("get_path requires at least one key")
        vid = self._key_dict[keys]  # KeyError propagates naturally
        return self._val_trie.restore_key(vid)

    # ------------------------------------------------------------------ #
    #  Membership and size                                                 #
    # ------------------------------------------------------------------ #

    def __contains__(self, path: object) -> bool:
        """Return ``True`` if *path* (a tuple of strings) is a leaf path.

        Example::

            ("a", "x") in tree
        """
        return path in self._key_dict

    def __len__(self) -> int:
        """Return the total number of leaf paths."""
        return len(self._key_dict)

    # ------------------------------------------------------------------ #
    #  Materialise back to plain dict                                      #
    # ------------------------------------------------------------------ #

    def to_dict(self) -> dict[str, Any]:
        """Reconstruct the original nested Python dict."""
        out: dict[str, Any] = {}
        for path, vid in self._key_dict.items():
            leaf = self._val_trie.restore_key(vid)
            node = out
            for key in path[:-1]:
                if key not in node:
                    node[key] = {}
                node = node[key]
            node[path[-1]] = leaf
        return out

    # ------------------------------------------------------------------ #
    #  Pickle support                                                      #
    # ------------------------------------------------------------------ #

    def __reduce__(self) -> tuple[Any, tuple[bytes]]:
        buf = io.BytesIO()
        val_bytes = self._val_trie.to_bytes()
        paths_bytes = _encode_paths(self._key_dict)
        buf.write(_MAGIC)
        buf.write(struct.pack("<Q", _VERSION))
        buf.write(struct.pack(
            "<QQQQ",
            len(self._key_dict),
            len(val_bytes),
            len(paths_bytes),
            self._val_vocab_size,
        ))
        buf.write(val_bytes)
        buf.write(paths_bytes)
        return (CompactTreeFlat._unpickle_from_bytes, (buf.getvalue(),))

    @staticmethod
    def _unpickle_from_bytes(data: bytes) -> CompactTreeFlat:
        buf = io.BytesIO(data)
        _magic, _ver = struct.unpack("<5sQ", buf.read(13))
        assert _magic == _MAGIC
        assert _ver == _VERSION
        n_paths, val_trie_len, paths_buf_len, val_vocab_size = \
            struct.unpack("<QQQQ", buf.read(32))
        val_bytes = buf.read(val_trie_len)
        paths_bytes = buf.read(paths_buf_len)

        obj = CompactTreeFlat.__new__(CompactTreeFlat)
        obj._val_trie = MarisaTrie.from_bytes(val_bytes, cache_size=val_vocab_size)
        obj._val_vocab_size = val_vocab_size
        obj._key_dict = _decode_paths(paths_bytes)
        return obj

    # ------------------------------------------------------------------ #
    #  Repr                                                                #
    # ------------------------------------------------------------------ #

    def __repr__(self) -> str:
        return f"CompactTreeFlat.from_dict({self.to_dict()!r})"

    def __str__(self) -> str:
        return str(self.to_dict())
