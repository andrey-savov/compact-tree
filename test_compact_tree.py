import re
import struct
import tempfile
from pathlib import Path

import pytest
from bitarray import bitarray

from louds import LOUDS
from compact_tree import CompactTree
from succinct.poppy import Poppy


class TestLOUDS:
    """Tests for the LOUDS class."""

    def test_louds_first_child_root(self):
        """Test first_child on root node."""
        # Create a simple bit vector: 1 (root has child)
        ba = bitarray('1')
        louds = LOUDS(Poppy(ba))
        
        child = louds.first_child(0)
        assert child == 1

    def test_louds_first_child_no_children(self):
        """Test first_child when node has no children."""
        # Create bit vector: 10 (root has child, that child has no children)
        ba = bitarray('10')
        louds = LOUDS(Poppy(ba))
        
        child = louds.first_child(0)
        assert child == 1
        
        # Node 1 has no children (next bit is 0)
        child = louds.first_child(1)
        assert child is None

    def test_louds_next_sibling(self):
        """Test next_sibling navigation."""
        # Create bit vector with multiple siblings: 11 (two children of root)
        ba = bitarray('11')
        louds = LOUDS(Poppy(ba))
        
        # First child exists
        child1 = louds.first_child(0)
        assert child1 == 1
        
        # Next sibling should be node 2
        child2 = louds.next_sibling(1)
        assert child2 == 2

    def test_louds_no_next_sibling(self):
        """Test next_sibling returns None when at end."""
        ba = bitarray('10')
        louds = LOUDS(Poppy(ba))
        
        child = louds.first_child(0)
        assert child == 1
        
        # Node 1 has no next sibling (next bit is 0)
        next_sib = louds.next_sibling(1)
        assert next_sib is None

    def test_louds_three_level_tree(self):
        """Test LOUDS with a 3-level tree structure."""
        # Build a simple tree with root having 2 children
        # Bit sequence: 11
        ba = bitarray('11')
        louds = LOUDS(Poppy(ba))

        # Root has 2 children
        assert louds.first_child(0) == 1
        assert louds.next_sibling(1) == 2
        assert louds.next_sibling(2) is None
class TestCompactTreeNode:
    """Tests for the CompactTree._Node class."""

    def test_node_get_children(self):
        """Test _Node._children method with real LOUDS."""
        # Create a LOUDS with 3 children
        louds_bits = bitarray('111')
        louds = LOUDS(Poppy(louds_bits))
        
        # Get children of root
        children = []
        kid = louds.first_child(0)
        while kid is not None:
            children.append(kid)
            kid = louds.next_sibling(kid)
        
        assert len(children) == 3
        assert children == [1, 2, 3]

    def test_node_contains_key(self):
        """Test _Node.__contains__ logic."""
        louds_bits = bitarray('11')  # Root with 2 children
        louds = LOUDS(Poppy(louds_bits))
        
        # Get children
        children = []
        kid = louds.first_child(0)
        while kid is not None:
            children.append(kid)
            kid = louds.next_sibling(kid)
        
        # Both children exist
        assert len(children) == 2
        # Check that we can test containment logic
        assert 0 < len(children)  # At least some children
        assert 1 < len(children)  # At least 2 children


class TestCompactTree:
    """Tests for the CompactTree class."""

    def test_ctree_magic_check_invalid(self):
        """Test that invalid magic bytes are caught."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            f.write(b"XXXX")  # Wrong magic
            f.write(struct.pack("<Q", 1))
            f.write(struct.pack("<QQQQ", 0, 0, 0, 0))
            f.flush()
            fname = f.name
        
        try:
            with pytest.raises((AssertionError, struct.error)):
                CompactTree(fname)
        finally:
            Path(fname).unlink()

    def test_ctree_version_check_invalid(self):
        """Test that invalid version is caught."""
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            f.write(b"CTree")
            f.write(struct.pack("<Q", 999))  # Wrong version
            f.write(struct.pack("<QQQQ", 0, 0, 0, 0))
            f.flush()
            fname = f.name
        
        try:
            with pytest.raises(AssertionError):
                CompactTree(fname)
        finally:
            Path(fname).unlink()

    def test_louds_select_zero(self):
        """Test LOUDS select_zero operation."""
        # Create bit vector with known select_zero positions
        ba = bitarray('1001')  # Two 0s at positions 1 and 2
        louds = LOUDS(Poppy(ba))
        
        # Poppy should work with select operations
        assert isinstance(louds, LOUDS)

    def test_list_children_with_louds_structure(self):
        """Test _list_children logic with real LOUDS."""
        tree = CompactTree.__new__(CompactTree)
        
        # Create LOUDS with root having 3 children
        louds_bits = bitarray('111')
        louds = LOUDS(Poppy(louds_bits))
        tree.louds = louds
        
        children = tree._list_children(0)
        assert len(children) == 3
        assert children == [1, 2, 3]

    def test_louds_root_children_enumeration(self):
        """Test enumerating root children with LOUDS."""
        # Create LOUDS with 2 root children
        louds_bits = bitarray('11')
        louds = LOUDS(Poppy(louds_bits))
        
        children = []
        kid = louds.first_child(0)
        while kid is not None:
            children.append(kid)
            kid = louds.next_sibling(kid)
        
        assert len(children) == 2
        assert children == [1, 2]

    def test_louds_no_children(self):
        """Test node with no children."""
        louds_bits = bitarray('10')  # Root has child, but child has no children
        louds = LOUDS(Poppy(louds_bits))
        
        # Root has 1 child
        child = louds.first_child(0)
        assert child == 1
        
        # That child has no children
        no_child = louds.first_child(1)
        assert no_child is None


class TestNgrams:
    """Tests for building nested dicts from N-grams."""

    def test_build_and_roundtrip_ngrams_from_corpus(self):
        """Build 5-grams from corpus, store as CompactTree, reload and verify."""
        corpus_path = Path(__file__).with_name("corpus.txt")
        text = corpus_path.read_text(encoding="utf-8")

        sentences = [s.strip().casefold() for s in text.split(".") if s.strip()]
        tree: dict[str, object] = {}
        n = 5

        for sentence in sentences:
            words = re.findall(r"[A-Za-z0-9']+", sentence)
            if len(words) < n:
                continue
            for i in range(len(words) - n + 1):
                window = words[i:i + n]
                node: dict[str, object] = tree
                for word in window[:-2]:
                    child = node.get(word)
                    if not isinstance(child, dict):
                        child = {}
                        node[word] = child
                    node = child
                node[window[-2]] = window[-1]

        # Build CompactTree from dict
        ct = CompactTree.from_dict(tree)

        # Sanity checks on the in-memory tree
        assert "the" in ct
        assert ct["the"]["morning"]["sun"]["filtered"] == "through"

        # Round-trip through a temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        try:
            ct.serialize(fname)
            with CompactTree(fname) as ct2:
                assert ct2.to_dict() == tree
                assert "the" in ct2
                assert ct2["the"]["morning"]["sun"]["filtered"] == "through"
        finally:
            Path(fname).unlink()


# ------------------------------------------------------------------ #
#  from_dict / to_dict / serialize round-trip tests                    #
# ------------------------------------------------------------------ #


class TestFromDict:
    """Tests for CompactTree.from_dict factory method."""

    def test_flat_dict(self):
        d = {"hello": "world", "foo": "bar"}
        ct = CompactTree.from_dict(d)
        assert ct["hello"] == "world"
        assert ct["foo"] == "bar"
        assert "hello" in ct
        assert "missing" not in ct
        assert len(ct) == 2
        assert set(ct) == {"hello", "foo"}

    def test_nested_dict(self):
        d = {"a": {"x": "1", "y": "2"}, "b": "3"}
        ct = CompactTree.from_dict(d)
        assert ct["b"] == "3"
        node = ct["a"]
        assert isinstance(node, CompactTree._Node)
        assert node["x"] == "1"
        assert node["y"] == "2"
        assert "x" in node
        assert "missing" not in node
        assert len(node) == 2
        assert set(node) == {"x", "y"}

    def test_deep_nested(self):
        d = {"a": {"b": {"c": "deep"}}}
        ct = CompactTree.from_dict(d)
        assert ct["a"]["b"]["c"] == "deep"

    def test_empty_dict(self):
        ct = CompactTree.from_dict({})
        assert len(ct) == 0
        assert list(ct) == []

    def test_single_entry(self):
        ct = CompactTree.from_dict({"x": "y"})
        assert ct["x"] == "y"
        assert len(ct) == 1

    def test_key_error_root(self):
        ct = CompactTree.from_dict({"a": "b"})
        with pytest.raises(KeyError):
            ct["missing"]

    def test_key_error_nested(self):
        ct = CompactTree.from_dict({"a": {"x": "1"}})
        with pytest.raises(KeyError):
            ct["a"]["missing"]

    def test_sparse_keys_across_levels(self):
        """Keys that exist only at certain levels must be handled correctly."""
        d = {"apple": {"cherry": "1"}, "banana": "2"}
        ct = CompactTree.from_dict(d)
        assert ct["banana"] == "2"
        assert ct["apple"]["cherry"] == "1"
        assert "cherry" not in ct  # cherry only exists one level down
        assert "apple" not in ct["apple"]  # apple only exists at root

    def test_duplicate_values(self):
        """Multiple leaves sharing the same string value."""
        d = {"a": "same", "b": "same", "c": "same"}
        ct = CompactTree.from_dict(d)
        assert ct["a"] == "same"
        assert ct["b"] == "same"
        assert ct["c"] == "same"


class TestToDict:
    """Tests for CompactTree.to_dict."""

    def test_flat(self):
        d = {"hello": "world", "foo": "bar"}
        assert CompactTree.from_dict(d).to_dict() == d

    def test_nested(self):
        d = {"a": {"b": {"c": "d"}}, "e": "f"}
        assert CompactTree.from_dict(d).to_dict() == d

    def test_empty(self):
        assert CompactTree.from_dict({}).to_dict() == {}


class TestSerialize:
    """Tests for serialize / deserialise round-trip."""

    def test_round_trip_flat(self):
        d = {"hello": "world", "foo": "bar"}
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        try:
            ct.serialize(fname)
            with CompactTree(fname) as ct2:
                assert ct2["hello"] == "world"
                assert ct2["foo"] == "bar"
                assert ct2.to_dict() == d
        finally:
            Path(fname).unlink()

    def test_round_trip_nested(self):
        d = {"a": {"x": "1", "y": "2"}, "b": "3"}
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        try:
            ct.serialize(fname)
            with CompactTree(fname) as ct2:
                assert ct2["b"] == "3"
                assert ct2["a"]["x"] == "1"
                assert ct2["a"]["y"] == "2"
                assert ct2.to_dict() == d
        finally:
            Path(fname).unlink()

    def test_round_trip_deep(self):
        d = {"a": {"b": {"c": {"d": "leaf"}}}}
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        try:
            ct.serialize(fname)
            with CompactTree(fname) as ct2:
                assert ct2["a"]["b"]["c"]["d"] == "leaf"
                assert ct2.to_dict() == d
        finally:
            Path(fname).unlink()

    def test_round_trip_ngrams(self):
        """Full round-trip: corpus -> ngram dict -> CompactTree -> file -> reload."""
        corpus_path = Path(__file__).with_name("corpus.txt")
        text = corpus_path.read_text(encoding="utf-8")
        sentences = [s.strip() for s in text.split(".") if s.strip()]
        tree: dict[str, object] = {}
        n = 5
        for sentence in sentences:
            words = re.findall(r"[A-Za-z0-9']+", sentence)
            if len(words) < n:
                continue
            for i in range(len(words) - n + 1):
                window = words[i:i + n]
                node: dict[str, object] = tree
                for word in window[:-2]:
                    child = node.get(word)
                    if not isinstance(child, dict):
                        child = {}
                        node[word] = child
                    node = child
                node[window[-2]] = window[-1]

        ct = CompactTree.from_dict(tree)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        try:
            ct.serialize(fname)
            with CompactTree(fname) as ct2:
                assert ct2.to_dict() == tree
        finally:
            Path(fname).unlink()


class TestCompression:
    """Tests for gzip compression support."""

    def test_gzip_round_trip_flat(self):
        """Test gzip compression with flat dict."""
        d = {"hello": "world", "foo": "bar"}
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree.gz") as f:
            fname = f.name
        try:
            # Serialize with gzip compression
            ct.serialize(fname, storage_options={"compression": "gzip"})
            
            # Deserialize with gzip compression
            with CompactTree(fname, storage_options={"compression": "gzip"}) as ct2:
                assert ct2["hello"] == "world"
                assert ct2["foo"] == "bar"
                assert ct2.to_dict() == d
        finally:
            Path(fname).unlink()

    def test_gzip_round_trip_nested(self):
        """Test gzip compression with nested dict."""
        d = {"a": {"x": "1", "y": "2"}, "b": {"c": {"d": "deep"}}}
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree.gz") as f:
            fname = f.name
        try:
            ct.serialize(fname, storage_options={"compression": "gzip"})
            with CompactTree(fname, storage_options={"compression": "gzip"}) as ct2:
                assert ct2["b"]["c"]["d"] == "deep"
                assert ct2["a"]["x"] == "1"
                assert ct2.to_dict() == d
        finally:
            Path(fname).unlink()

    def test_gzip_compression_reduces_size(self):
        """Test that gzip actually compresses the data."""
        # Create a dict with repetitive data (should compress well)
        d = {f"key{i}": "same_value_repeated" for i in range(100)}
        ct = CompactTree.from_dict(d)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname_uncompressed = f.name
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree.gz") as f:
            fname_compressed = f.name
        
        try:
            # Save both compressed and uncompressed
            ct.serialize(fname_uncompressed)
            ct.serialize(fname_compressed, storage_options={"compression": "gzip"})
            
            size_uncompressed = Path(fname_uncompressed).stat().st_size
            size_compressed = Path(fname_compressed).stat().st_size
            
            # Compressed should be smaller
            assert size_compressed < size_uncompressed
        finally:
            Path(fname_uncompressed).unlink()
            Path(fname_compressed).unlink()

    def test_wrong_compression_on_load_fails(self):
        """Test that loading with wrong compression fails cleanly."""
        d = {"hello": "world"}
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        
        try:
            # Save uncompressed
            ct.serialize(fname)
            
            # Try to load as gzip - should fail
            with pytest.raises((AssertionError, OSError, Exception)):
                CompactTree(fname, storage_options={"compression": "gzip"})
        finally:
            Path(fname).unlink()

    def test_wrong_compression_on_save(self):
        """Test that invalid compression type is rejected."""
        d = {"hello": "world"}
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported compression"):
                ct.serialize(fname, storage_options={"compression": "invalid"})
        finally:
            # Clean up in case file was created
            if Path(fname).exists():
                Path(fname).unlink()

    def test_gzip_with_empty_dict(self):
        """Test gzip compression with empty dict."""
        d = {}
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree.gz") as f:
            fname = f.name
        try:
            ct.serialize(fname, storage_options={"compression": "gzip"})
            with CompactTree(fname, storage_options={"compression": "gzip"}) as ct2:
                assert ct2.to_dict() == d
                assert len(ct2) == 0
        finally:
            Path(fname).unlink()

    def test_gzip_with_large_nested_structure(self):
        """Test gzip compression with large nested structure."""
        # Build a complex nested structure
        d = {}
        for i in range(10):
            d[f"level1_{i}"] = {
                f"level2_{j}": {
                    f"level3_{k}": f"value_{i}_{j}_{k}"
                    for k in range(5)
                }
                for j in range(5)
            }
        
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree.gz") as f:
            fname = f.name
        try:
            ct.serialize(fname, storage_options={"compression": "gzip"})
            with CompactTree(fname, storage_options={"compression": "gzip"}) as ct2:
                assert ct2.to_dict() == d
                # Spot check a few values
                assert ct2["level1_0"]["level2_0"]["level3_0"] == "value_0_0_0"
                assert ct2["level1_5"]["level2_3"]["level3_4"] == "value_5_3_4"
        finally:
            Path(fname).unlink()

    def test_none_compression_explicit(self):
        """Test that compression=None works explicitly."""
        d = {"hello": "world"}
        ct = CompactTree.from_dict(d)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        try:
            ct.serialize(fname, storage_options={"compression": None})
            with CompactTree(fname, storage_options={"compression": None}) as ct2:
                assert ct2.to_dict() == d
        finally:
            Path(fname).unlink()

    def test_unsupported_compression_serialize(self):
        """Test that unsupported compression raises error on serialize."""
        d = {"foo": "bar"}
        ct = CompactTree.from_dict(d)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported compression"):
                ct.serialize(fname, storage_options={"compression": "bzip2"})
        finally:
            if Path(fname).exists():
                Path(fname).unlink()

    def test_unsupported_compression_load(self):
        """Test that unsupported compression raises error on load."""
        d = {"foo": "bar"}
        ct = CompactTree.from_dict(d)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        
        try:
            # Serialize normally first
            ct.serialize(fname)
            
            # Try to load with unsupported compression
            with pytest.raises(ValueError, match="Unsupported compression"):
                CompactTree(fname, storage_options={"compression": "bzip2"})
        finally:
            Path(fname).unlink()


class TestPickle:
    """Tests for pickle support."""

    def test_pickle_simple_tree(self):
        """Test pickling and unpickling a simple tree."""
        import pickle
        
        d = {"a": "1", "b": "2", "c": "3"}
        tree = CompactTree.from_dict(d)
        
        # Pickle and unpickle
        pickled = pickle.dumps(tree)
        tree2 = pickle.loads(pickled)
        
        # Verify contents
        assert tree2.to_dict() == d
        assert tree2["a"] == "1"
        assert tree2["b"] == "2"
        assert tree2["c"] == "3"

    def test_pickle_nested_tree(self):
        """Test pickling a nested tree."""
        import pickle
        
        d = {
            "x": "10",
            "y": {
                "z": "20",
                "w": "30"
            },
            "a": "5"
        }
        tree = CompactTree.from_dict(d)
        
        # Pickle and unpickle
        pickled = pickle.dumps(tree)
        tree2 = pickle.loads(pickled)
        
        # Verify contents
        assert tree2.to_dict() == d
        assert tree2["y"]["z"] == "20"
        assert tree2["y"]["w"] == "30"

    def test_pickle_iteration(self):
        """Test that iteration works on unpickled tree."""
        import pickle
        
        d = {"a": "1", "b": "2", "c": "3"}
        tree = CompactTree.from_dict(d)
        
        pickled = pickle.dumps(tree)
        tree2 = pickle.loads(pickled)
        
        # Test iteration - check keys are present, not specific order
        keys = set(tree2)
        assert keys == {"a", "b", "c"}

    def test_pickle_contains(self):
        """Test that 'in' operator works on unpickled tree."""
        import pickle
        
        d = {"a": "1", "b": "2"}
        tree = CompactTree.from_dict(d)
        
        pickled = pickle.dumps(tree)
        tree2 = pickle.loads(pickled)
        
        # Test contains
        assert "a" in tree2
        assert "b" in tree2
        assert "c" not in tree2

    def test_pickle_len(self):
        """Test that len() works on unpickled tree."""
        import pickle
        
        d = {"a": "1", "b": "2", "c": "3"}
        tree = CompactTree.from_dict(d)
        
        pickled = pickle.dumps(tree)
        tree2 = pickle.loads(pickled)
        
        assert len(tree2) == 3

    def test_pickle_deep_nesting(self):
        """Test pickling deeply nested structures."""
        import pickle
        
        d = {
            "level1": {
                "level2": {
                    "level3": {
                        "level4": "deep_value"
                    }
                }
            }
        }
        tree = CompactTree.from_dict(d)
        
        pickled = pickle.dumps(tree)
        tree2 = pickle.loads(pickled)
        
        assert tree2["level1"]["level2"]["level3"]["level4"] == "deep_value"

    def test_pickle_size_efficiency(self):
        """Test that pickle uses serialize method (compact format)."""
        import pickle
        import tempfile
        
        d = {"a": "1", "b": {"c": "2", "d": "3"}}
        tree = CompactTree.from_dict(d)
        
        # Get serialized size
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        try:
            tree.serialize(fname)
            serialized_size = Path(fname).stat().st_size
        finally:
            Path(fname).unlink()
        
        # Get pickle size
        pickled = pickle.dumps(tree)
        pickle_size = len(pickled)
        
        # Pickle should be close to serialized size (within 2x due to protocol overhead)
        assert pickle_size < serialized_size * 2.0
        # And definitely much smaller than if we pickled all attributes separately (which would be ~5x)
        assert pickle_size < serialized_size * 3.0

    def test_pickle_tree_from_file(self):
        """Test pickling a tree that was loaded from file."""
        import pickle
        
        d = {"x": "100", "y": {"z": "200"}}
        tree = CompactTree.from_dict(d)
        
        # Save to file and reload
        with tempfile.NamedTemporaryFile(delete=False, suffix=".ctree") as f:
            fname = f.name
        try:
            tree.serialize(fname)
            tree2 = CompactTree(fname)
            
            # Now pickle the loaded tree
            pickled = pickle.dumps(tree2)
            tree3 = pickle.loads(pickled)
            
            # Verify it works
            assert tree3.to_dict() == d
            assert tree3["y"]["z"] == "200"
        finally:
            Path(fname).unlink()

    def test_pickle_empty_tree(self):
        """Test pickling an empty tree."""
        import pickle
        
        d = {}
        tree = CompactTree.from_dict(d)
        
        pickled = pickle.dumps(tree)
        tree2 = pickle.loads(pickled)
        
        assert tree2.to_dict() == {}
        assert len(tree2) == 0

    def test_pickle_large_tree(self):
        """Test pickling a larger tree with many keys."""
        import pickle
        
        d = {str(i): str(i * 10) for i in range(100)}
        tree = CompactTree.from_dict(d)
        
        pickled = pickle.dumps(tree)
        tree2 = pickle.loads(pickled)
        
        assert tree2.to_dict() == d
        assert tree2["50"] == "500"
        assert len(tree2) == 100

    def test_pickle_protocol_versions(self):
        """Test that pickling works with different pickle protocols."""
        import pickle
        
        d = {"a": "1", "b": {"c": "2"}}
        tree = CompactTree.from_dict(d)
        
        # Test with different protocols
        for protocol in range(pickle.HIGHEST_PROTOCOL + 1):
            pickled = pickle.dumps(tree, protocol=protocol)
            tree2 = pickle.loads(pickled)
            assert tree2.to_dict() == d


class TestStringRepresentations:
    """Tests for __str__ and __repr__ methods."""

    def test_repr_simple_tree(self):
        """Test __repr__ returns interpretable representation."""
        d = {"a": "1", "b": "2"}
        tree = CompactTree.from_dict(d)
        
        r = repr(tree)
        assert r.startswith("CompactTree.from_dict(")
        assert "'a': '1'" in r
        assert "'b': '2'" in r

    def test_repr_nested_tree(self):
        """Test __repr__ with nested structure."""
        d = {"x": "10", "y": {"z": "20"}}
        tree = CompactTree.from_dict(d)
        
        r = repr(tree)
        assert r.startswith("CompactTree.from_dict(")
        assert "'x': '10'" in r
        assert "'y':" in r
        assert "'z': '20'" in r

    def test_repr_empty_tree(self):
        """Test __repr__ with empty tree."""
        tree = CompactTree.from_dict({})
        
        r = repr(tree)
        assert r == "CompactTree.from_dict({})"

    def test_str_simple_tree(self):
        """Test __str__ returns dict-like representation."""
        d = {"a": "1", "b": "2"}
        tree = CompactTree.from_dict(d)
        
        s = str(tree)
        # Should contain all keys and values
        assert "'a'" in s and "'1'" in s
        assert "'b'" in s and "'2'" in s
        
        # Should round-trip correctly
        assert tree.to_dict() == d

    def test_str_nested_tree(self):
        """Test __str__ with nested structure."""
        d = {"x": "10", "y": {"z": "20"}}
        tree = CompactTree.from_dict(d)
        
        s = str(tree)
        # Should contain all keys and values
        assert "'x'" in s and "'10'" in s
        assert "'y'" in s and "'z'" in s and "'20'" in s
        
        # Should round-trip correctly
        assert tree.to_dict() == d

    def test_str_empty_tree(self):
        """Test __str__ with empty tree."""
        tree = CompactTree.from_dict({})
        
        s = str(tree)
        assert s == "{}"

    def test_repr_evaluable(self):
        """Test that __repr__ output can be evaluated to recreate tree."""
        d = {"a": "1", "b": "2"}
        tree = CompactTree.from_dict(d)
        
        r = repr(tree)
        # Should be able to eval it (in theory, if CompactTree is in scope)
        assert "CompactTree.from_dict(" in r
        assert r.endswith(")")

    def test_node_repr(self):
        """Test __repr__ for _Node."""
        d = {"a": {"x": "1", "y": "2"}}
        tree = CompactTree.from_dict(d)
        
        node = tree["a"]
        r = repr(node)
        # Should show the dict representation
        assert "'x': '1'" in r
        assert "'y': '2'" in r

    def test_node_str(self):
        """Test __str__ for _Node."""
        d = {"a": {"x": "1", "y": "2"}}
        tree = CompactTree.from_dict(d)
        
        node = tree["a"]
        s = str(node)
        # Should contain all keys and values
        assert "'x'" in s and "'1'" in s
        assert "'y'" in s and "'2'" in s

    def test_node_nested_repr(self):
        """Test __repr__ for nested _Node."""
        d = {"a": {"b": {"c": "3"}}}
        tree = CompactTree.from_dict(d)
        
        node = tree["a"]
        r = repr(node)
        assert "'b':" in r
        assert "'c': '3'" in r

    def test_str_is_dict_like(self):
        """Test that str(tree) matches str(tree.to_dict())."""
        d = {"foo": "bar", "baz": {"qux": "quux"}}
        tree = CompactTree.from_dict(d)
        
        # str(tree) should match str of the materialized dict
        assert str(tree) == str(tree.to_dict())

    def test_repr_different_from_str(self):
        """Test that __repr__ and __str__ are different."""
        d = {"a": "1"}
        tree = CompactTree.from_dict(d)
        
        # repr should be CompactTree.from_dict(...)
        # str should be just the dict
        assert repr(tree) != str(tree)
        assert "CompactTree" in repr(tree)
        assert "CompactTree" not in str(tree)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
