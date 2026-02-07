# compact-tree

[![Tests](https://github.com/andrey-savov/compact-tree/workflows/Tests/badge.svg)](https://github.com/andrey-savov/compact-tree/actions)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Compact, read-only nested dictionary backed by succinct data structures.

`CompactTree` stores a nested Python `dict` using a LOUDS-encoded trie with DAWG-style key/value deduplication, enabling low-memory random access and efficient serialization.

## Features

- **Memory-efficient**: Uses succinct data structures (LOUDS trie + DAWG-style deduplication)
- **Fast lookups**: O(1) rank/select operations via Poppy bit vectors
- **Serializable**: Save and load from disk with efficient binary format
- **Read-only**: Optimized for lookup-heavy workloads
- **Storage-agnostic**: Works with local files and remote storage via `fsspec`

## Installation

```bash
pip install compact-tree
```

Or install from source:

```bash
git clone https://github.com/andrey-savov/compact-tree.git
cd compact-tree
pip install -e .
```

## Quick Start

```python
from compact_tree import CompactTree

# Build from a nested dict
tree = CompactTree.from_dict({
    "a": {
        "x": "1",
        "y": "2"
    },
    "b": "3"
})

# Access like a normal dict
print(tree["a"]["x"])   # "1"
print(tree["b"])        # "3"
print("a" in tree)      # True

# Serialize to file
tree.serialize("tree.ctree")

# Load from file
loaded_tree = CompactTree("tree.ctree")

# Convert back to plain dict
plain_dict = loaded_tree.to_dict()
```

## How It Works

### LOUDS (Level-Order Unary Degree Sequence)

LOUDS is a succinct tree representation that encodes an ordered tree into a single bit string using roughly **2n** bits for **n** nodes (close to the information-theoretic minimum).

**Encoding rule:** Traverse the tree in breadth-first (level) order. For each node, write _d_ `1`-bits (where _d_ is the node's number of children) followed by a single `0`-bit.

Example for a root with children A (2 kids) and B (0 kids):

```
root  ->  1 1 0      (2 children, then 0-terminator)
A     ->  1 1 0      (2 children)
B     ->  0          (leaf)
```

Navigation relies on **rank** and **select** queries:

| Operation        | Description                                        |
|------------------|---------------------------------------------------|
| `first_child(v)` | Find the (v-1)-th `0`, check next position       |
| `next_sibling(v)` | Find the `1`-bit for node v, check next position |

### DAWG-Style Deduplication

- **Keys** are collected, sorted, and deduplicated into a global vocabulary
- **Values** (leaves) are similarly deduplicated into a value table
- Edge labels store integer IDs rather than raw strings
- Same key/value appearing multiple times is stored only once

## Architecture

```
CompactTree
  |
  +-- louds    : _LOUDS        bit-vector tree topology (Poppy rank/select)
  +-- elbl     : memoryview    edge labels  (uint32 key ids, 4 bytes per node)
  +-- vcol     : memoryview    value column (uint32: value id or 0xFFFFFFFF for internal nodes)
  +-- _keys_buf: memoryview    length-prefixed UTF-8 key strings
  +-- val      : memoryview    length-prefixed UTF-8 value strings
```

## Binary Format (v2)

```
Magic   : 5 bytes   "CTree"
Version : 8 bytes   uint64 LE (always 2)
Header  : 5 x 8 bytes  lengths of: keys, values, louds, vcol, elbl
Payload : keys_bytes | val_bytes | louds_bytes | vcol_bytes | elbl_bytes
```

## Dependencies

- `bitarray` — Mutable bit arrays
- `succinct` (Poppy) — Rank/select in O(1)
- `fsspec` — Filesystem abstraction for local and remote storage

## Testing

```bash
pytest test_compact_tree.py
```

## Contributing

Contributions are welcome! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Based on LOUDS (Level-Order Unary Degree Sequence) tree representation
- Uses Poppy rank/select implementation from the `succinct` library
- Inspired by DAWG (Directed Acyclic Word Graph) compression techniques
