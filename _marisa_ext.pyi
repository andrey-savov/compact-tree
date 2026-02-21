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
