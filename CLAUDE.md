# compact-tree

Compact, read-only nested dictionary backed by succinct data structures.
`CompactTree` stores a nested Python `dict` using a LOUDS-encoded trie with
DAWG-style key/value deduplication, enabling low-memory random access and
efficient serialisation.

## Key concepts

### LOUDS (Level-Order Unary Degree Sequence)

LOUDS is a succinct tree representation that encodes an ordered tree into a
single bit string using roughly **2n** bits for **n** nodes (close to the
information-theoretic minimum).

**Encoding rule:** traverse the tree in breadth-first (level) order. For each
node, write _d_ `1`-bits (where _d_ is the node's number of children) followed
by a single `0`-bit. The resulting bit string fully describes the tree topology.

Example -- a root with children A (2 kids) and B (0 kids):

```
root  ->  1 1 0      (2 children, then 0-terminator)
A     ->  1 1 0      (2 children)
B     ->  0           (leaf)
...
```

Navigation relies on **rank** and **select** queries over the bit vector:

| Operation        | Formula                                        |
|------------------|------------------------------------------------|
| `first_child(v)` | find the (v-1)-th `0`, move one position right; if that bit is `1`, its rank gives the child node id |
| `next_sibling(v)` | find the `1`-bit for node v (`select(v-1)`), check the next position; if `1`, sibling is `v+1` |

The `LOUDS` class wraps a **Poppy** bit vector (from the `succinct` package)
which answers rank/select in O(1) time with small overhead.

### MarisaTrie

`MarisaTrie` is a compact word-to-index mapping built on a LOUDS-encoded trie
with path compression and minimal perfect hashing (MPH). It provides:

- **Path compression**: single-child edges are merged into one label.
- **Subtree counting**: enables MPH so every unique word gets a dense index in
  `[0, N)`.
- **Reverse lookup**: `restore_key(idx)` recovers the word from its index.
- **Serialisation**: `to_bytes()` / `from_bytes()` for embedding inside
  `CompactTree`'s binary format.

`CompactTree` uses two `MarisaTrie` instances -- one for keys and one for
values -- replacing the earlier length-prefixed UTF-8 buffers.

### DAWG-style key and value deduplication

A DAWG (Directed Acyclic Word Graph) compresses a dictionary by sharing common
structure. This project borrows the _deduplication_ idea from DAWGs without
building a full automaton:

- **Keys** are collected, sorted, and deduplicated via a `MarisaTrie`. Each
  unique key string is assigned a dense integer id. Edge labels in the trie
  store these integer ids rather than raw strings, so the same key appearing at
  multiple levels is stored only once.
- **Values** (leaves) are similarly deduplicated via a second `MarisaTrie`. Each
  leaf stores a value id pointing into the trie.

This gives DAWG-like space savings -- repeated keys and values across the nested
dict are stored once -- while keeping the trie structure simple.

## Architecture

```
CompactTree
  |
  +-- louds      : LOUDS       bit-vector tree topology (Poppy rank/select)
  +-- elbl       : bytes       edge labels  (uint32 key ids, 4 bytes per node)
  +-- vcol       : bytes       value column (uint32: value id or 0xFFFFFFFF for internal nodes)
  +-- _key_trie  : MarisaTrie  key vocabulary (word <-> dense index)
  +-- _val_trie  : MarisaTrie  value vocabulary (word <-> dense index)
```

Each non-root node `v` (1-indexed) occupies a 4-byte slot at offset
`(v-1)*4` in both `elbl` (its edge label / key id) and `vcol` (its value id
or the sentinel `0xFFFFFFFF` for internal nodes).

## Binary format (v3)

```
Magic   : 5 bytes   "CTree"
Version : 8 bytes   uint64 LE (always 3)
Header  : 5 x 8 bytes  lengths of: keys, values, louds, vcol, elbl
Payload : keys_bytes | val_bytes | louds_bytes | vcol_bytes | elbl_bytes
```

`keys_bytes` and `val_bytes` are serialised `MarisaTrie` instances (see
`MarisaTrie.to_bytes()`). `louds_bytes` is the raw bitarray, `vcol_bytes` and
`elbl_bytes` are packed uint32 arrays.

## Usage

```python
from compact_tree import CompactTree

# Build from a nested dict
tree = CompactTree.from_dict({"a": {"x": "1"}, "b": "2"})

# Access like a dict
tree["a"]["x"]   # "1"
tree["b"]        # "2"
"a" in tree      # True
len(tree)        # 2
list(tree)       # ["a", "b"]

# String representations
str(tree)        # "{'a': {'x': '1'}, 'b': '2'}"
repr(tree)       # "CompactTree.from_dict({'a': {'x': '1'}, 'b': '2'})"

# Serialise / deserialise
tree.serialize("tree.ctree")
tree2 = CompactTree("tree.ctree")

# Gzip compression
tree.serialize("tree.ctree.gz", storage_options={"compression": "gzip"})
tree3 = CompactTree("tree.ctree.gz", storage_options={"compression": "gzip"})

# Pickle support
import pickle
data = pickle.dumps(tree)
tree4 = pickle.loads(data)

# Materialise back to plain dict
tree.to_dict()
```

## Dependencies

- `bitarray` -- mutable bit arrays
- `succinct` (Poppy) -- rank/select in O(1)
- `fsspec` -- filesystem abstraction for local and remote storage
