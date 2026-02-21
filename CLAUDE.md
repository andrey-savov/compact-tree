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
- **C-level LRU-cached lookups**: per-instance `functools.lru_cache` wrapping
  `_index_uncached` is installed as an instance attribute at construction and
  deserialization time, giving near-zero overhead on cache hits.
- **Bulk enumeration**: `to_dict()` returns `{word: index}` for every word in
  O(N). On the first call after construction it returns a pre-built mapping
  from the intermediate trie data (zero LOUDS traversals) and frees it
  immediately; subsequent calls or calls on deserialized tries perform a single
  DFS over the LOUDS bit vector.
- **O(1) label access**: pre-computed `_label_offsets` array avoids sequential
  scans when reading edge labels.
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

### MarisaTrie internal layout (build time vs. run time)

During `MarisaTrie.__init__` two sets of structures coexist briefly:

| Structure | Purpose | Lifetime |
|---|---|---|
| `nodes_metadata` (list) | BFS-ordered `(inode, label, is_terminal)` tuples from path-compressed intermediate trie | build only, freed after `_build_louds` returns |
| `children_map` (dict) | parent-index → child-index list, same BFS order | build only |
| `_word_to_idx` (dict) | `{word: idx}` built from the above via `_build_word_index_from_intermediate()` | exists from end of `__init__` until first `to_dict()` call; freed immediately after |
| `_louds`, `_labels`, `_terminal`, `_counts`, `_label_offsets` | LOUDS trie for query-time navigation and serialization | permanent (run time + serialization) |
| `index` (lru_cache) | per-instance C-level cache wrapping `_index_uncached` | permanent (run time) |

The separation ensures that `_word_to_idx` (a full vocabulary dict, potentially
hundreds of thousands of entries) does not persist beyond the single `to_dict()`
call made by `CompactTree.from_dict()`, keeping steady-state memory minimal.

### from_dict build pipeline

```
from_dict(data, *, vocabulary_size=None)
  |
  +-- _walk_dict()              collect all_keys (set) + all_values (list)
  |
  +-- key_cache_size  =  vocabulary_size  or  len(all_keys)
  +-- val_cache_size  =  vocabulary_size  or  len(unique_values)
  |
  +-- MarisaTrie(all_keys, cache_size=key_cache_size)      build key trie
  |     _build_intermediate_trie() -> dict-of-dicts
  |     _build_louds()          -> LOUDS + _counts + _word_to_idx (via DFS over nodes_metadata)
  |
  +-- MarisaTrie(unique_values, cache_size=val_cache_size) build value trie  (same pipeline)
  |
  +-- key_trie.to_dict()        O(N) pop of _word_to_idx  ->  key_id: dict[str,int]
  +-- val_trie.to_dict()        O(M) pop of _word_to_idx  ->  val_id: dict[str,int]
  |
  +-- BFS over data             emit LOUDS bits + elbl + vcol
        _key_order_cache         frozenset-keyed, amortises sort + key_id lookup
                                 across sibling nodes with identical key sets
```

Critically, LOUDS rank/select is used only **3 times** during `from_dict`:
once per `Poppy.__init__` call (one key trie, one value trie, one CompactTree
LOUDS). All vocabulary lookups during BFS encoding are O(1) plain-dict hits.

## Binary format (v4)

```
Magic   : 5 bytes   "CTree"
Version : 8 bytes   uint64 LE (always 4; v3 files are still readable)
Header  : 7 x 8 bytes  lengths of: keys, values, louds, vcol, elbl,
                         key_vocab_size, val_vocab_size
Payload : keys_bytes | val_bytes | louds_bytes | vcol_bytes | elbl_bytes
```

`key_vocab_size` and `val_vocab_size` are the effective `lru_cache(maxsize=…)` values
used for the key and value `MarisaTrie` instances respectively.  They are set during
`from_dict` (either from the caller-supplied `vocabulary_size` hint or computed
automatically as `len(all_keys)` / `len(unique_values)`) and restored on every load
so that query-time caches are immediately correctly sized.

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
