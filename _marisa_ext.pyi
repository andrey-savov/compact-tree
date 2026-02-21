"""Type stubs for the _marisa_ext C extension."""

class TrieIndex:
    """C-level radix trie index for fast word-to-int lookup."""

    def __init__(
        self,
        label_bytes: bytes,
        label_off: bytes,
        label_len: bytes,
        is_terminal: bytes,
        ch_start: bytes,
        ch_cnt: bytes,
        ch_first_byte: bytes,
        ch_node_id: bytes,
        ch_pfx_count: bytes,
        n_nodes: int,
        total_n: int,
        root_is_terminal: int,
    ) -> None: ...

    def lookup(self, word: str) -> int:
        """Return the MPH index for word. Raises KeyError on miss."""
        ...


class TreeIndex:
    """C-level CompactTree traversal helper."""

    def __init__(
        self,
        elbl: bytes,
        vcol: bytes,
        child_start: bytes,
        child_count: bytes,
        n_tree_nodes: int,
        key_trie: TrieIndex,
        val_restore: object,
    ) -> None: ...

    def get(self, node_pos: int, key: str) -> "int | str":
        """Look up key among children of node_pos.

        Returns child_pos (int) for internal nodes or the leaf value string.
        Raises KeyError on miss.
        """
        ...

    def find(self, node_pos: int, key: str) -> int:
        """Like get() but returns -1 on miss instead of raising KeyError."""
        ...

    def get_path(self, node_pos: int, /, *keys: str) -> "int | str":
        """Traverse all keys in a single C call.

        Returns child_pos (int) for an internal final node, or the leaf
        value string. Raises KeyError on any miss.
        """
        ...
