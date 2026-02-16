"""LOUDS (Level-Order Unary Degree Sequence) trie implementation."""

from typing import Optional

from bitarray import bitarray
from succinct.poppy import Poppy


class LOUDS:
    """LOUDS (Level-Order Unary Degree Sequence) trie built on a Poppy bit vector.

    Node 0 is the root.  Every other node v (≥ 1) corresponds to the
    v-th 1-bit (1-based) in the bit string.
    """

    def __init__(self, bv: Poppy, ba: Optional[bitarray] = None) -> None:
        self._bv = bv
        self._ba: bitarray = ba if ba is not None else bitarray()

    def first_child(self, v: int) -> Optional[int]:
        if v == 0:
            p = 0
        else:
            p = self._bv.select_zero(v - 1) + 1
        if p < len(self._bv) and self._bv[p]:
            return self._bv.rank(p)
        return None

    def next_sibling(self, v: int) -> Optional[int]:
        pos = self._bv.select(v - 1)  # position of the 1-bit for node v
        nxt = pos + 1
        if nxt < len(self._bv) and self._bv[nxt]:
            return v + 1
        return None
