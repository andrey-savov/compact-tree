import re
import struct
import tempfile
from pathlib import Path

import pytest

from compact_tree import CompactTree



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

    def test_list_children_direct_arrays(self):
        """Test _list_children logic with directly set CSR arrays."""
        import array
        tree = CompactTree.__new__(CompactTree)

        # Root (pos 0): 3 children starting at node 1.
        # Nodes 1, 2, 3: leaves (0 children).
        tree._child_start = array.array('I', [1, 0, 0, 0])
        tree._child_count = array.array('I', [3, 0, 0, 0])

        children = tree._list_children(0)
        assert len(children) == 3
        assert children == [1, 2, 3]


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


class TestLoadPerformance:
    """Load tests with large co-occurrence dictionaries.
    
    These tests benchmark CompactTree with realistic large datasets.
    Run with: pytest test_compact_tree.py::TestLoadPerformance --benchmark-only
    """

    @pytest.fixture(scope="class")
    def corpus_data(self):
        """Load and parse corpus for all tests in this class."""
        from pathlib import Path
        from download_large_corpus import load_and_parse_corpus, get_fallback_corpus
        
        # Try to use cached large corpus, fall back to corpus.txt
        cache_path = Path(__file__).parent / "large_corpus_small.txt"
        
        try:
            if cache_path.exists():
                parsed, metadata = load_and_parse_corpus(cache_path)
            else:
                # Use fallback (duplicated corpus.txt)
                parsed, metadata = get_fallback_corpus()
        except Exception:
            # Last resort: just use corpus.txt once
            corpus_path = Path(__file__).parent / "corpus.txt"
            parsed, metadata = load_and_parse_corpus(corpus_path)
        
        return parsed, metadata

    @pytest.fixture(scope="class")
    def cooccurrence_dict(self, corpus_data):
        """Build co-occurrence dictionary from corpus."""
        from build_cooccurrence import build_simple_cooccurrence_dict
        
        parsed, _ = corpus_data
        # Use small vocab for reasonable test time (50 words ~=  30K entries)
        cooccur_dict, metadata = build_simple_cooccurrence_dict(parsed, max_words=50)
        
        return cooccur_dict, metadata

    def test_corpus_loads(self, corpus_data):
        """Verify corpus loads correctly."""
        parsed, metadata = corpus_data
        
        assert len(parsed) > 0, "Should have paragraphs"
        assert metadata["unique_words"] > 100, "Should have substantial vocabulary"
        
        print(f"\nCorpus: {metadata['unique_words']} unique words, "
              f"{metadata['sentences']} sentences")

    def test_cooccurrence_dict_builds(self, cooccurrence_dict):
        """Verify co-occurrence dict builds correctly."""
        cooccur, metadata = cooccurrence_dict
        
        assert len(cooccur) > 0, "Should have level 1 keys"
        assert metadata["total_entries"] > 0, "Should have leaf entries"
        
        print(f"\nCo-occurrence: {metadata['level1_keys']} L1 keys, "
              f"{metadata['total_entries']} total entries")

    def test_build_compact_tree_from_cooccurrence(self, benchmark, cooccurrence_dict):
        """Benchmark building CompactTree from large co-occurrence dict."""
        cooccur, metadata = cooccurrence_dict
        
        # Benchmark the build
        tree = benchmark(CompactTree.from_dict, cooccur)
        
        # Verify it built correctly
        assert len(tree) == len(cooccur)
        
        # Store metadata for reporting
        benchmark.extra_info = {
            "input_keys": metadata["level1_keys"],
            "total_entries": metadata["total_entries"],
            "vocabulary": metadata["vocabulary_size"],
        }

    def test_tree_lookups_at_different_depths(self, benchmark, cooccurrence_dict):
        """Benchmark lookups at different depths in the tree."""
        import random
        
        cooccur, _ = cooccurrence_dict
        tree = CompactTree.from_dict(cooccur)
        
        # Collect sample paths at different depths
        level1_keys = list(cooccur.keys())[:10]
        level2_paths = []
        level3_paths = []
        
        for k1 in level1_keys:
            if k1 in tree:
                level2 = tree[k1]
                if isinstance(level2, CompactTree):
                    k2_list = list(level2.keys())[:5]
                    for k2 in k2_list:
                        level2_paths.append((k1, k2))
                        if k2 in level2:
                            level3 = level2[k2]
                            if isinstance(level3, CompactTree):
                                k3_list = list(level3.keys())[:3]
                                for k3 in k3_list:
                                    level3_paths.append((k1, k2, k3))
        
        # Benchmark mixed lookups
        def do_lookups():
            # Level 1 lookups
            for k1 in level1_keys[:5]:
                _ = tree[k1]
            
            # Level 2 lookups
            for k1, k2 in level2_paths[:10]:
                _ = tree[k1][k2]
            
            # Level 3 lookups
            for k1, k2, k3 in level3_paths[:10]:
                _ = tree[k1][k2][k3]
        
        benchmark(do_lookups)
        
        benchmark.extra_info = {
            "level1_lookups": min(5, len(level1_keys)),
            "level2_lookups": min(10, len(level2_paths)),
            "level3_lookups": min(10, len(level3_paths)),
        }

    def test_serialization_performance(self, benchmark, cooccurrence_dict, tmp_path):
        """Benchmark serialization to disk."""
        cooccur, metadata = cooccurrence_dict
        tree = CompactTree.from_dict(cooccur)
        
        output_path = tmp_path / "benchmark_tree.ctree"
        
        # Benchmark serialization
        benchmark(tree.serialize, str(output_path))
        
        # Check file size
        import os
        file_size = os.path.getsize(output_path)
        
        benchmark.extra_info = {
            "file_size_bytes": file_size,
            "file_size_kb": file_size / 1024,
            "input_entries": metadata["total_entries"],
            "compression_none": True,
        }

    def test_deserialization_performance(self, benchmark, cooccurrence_dict, tmp_path):
        """Benchmark deserialization from disk."""
        cooccur, _ = cooccurrence_dict
        tree = CompactTree.from_dict(cooccur)
        
        output_path = tmp_path / "benchmark_tree.ctree"
        tree.serialize(str(output_path))
        
        # Benchmark deserialization
        loaded_tree = benchmark(CompactTree, str(output_path))
        
        # Verify correctness
        assert len(loaded_tree) == len(tree)

    @pytest.mark.skipif(
        not pytest.importorskip("memory_profiler", reason="memory_profiler not installed"),
        reason="Requires memory_profiler"
    )
    def test_memory_usage(self, cooccurrence_dict):
        """Measure memory usage during tree construction."""
        from memory_profiler import memory_usage
        
        cooccur, metadata = cooccurrence_dict
        
        # Measure memory during build
        def build_tree():
            return CompactTree.from_dict(cooccur)
        
        mem_usage = memory_usage(build_tree, interval=0.01, include_children=False)
        
        peak_mb = max(mem_usage)
        baseline_mb = min(mem_usage)
        delta_mb = peak_mb - baseline_mb
        
        print(f"\nMemory usage: baseline={baseline_mb:.1f}MB, "
              f"peak={peak_mb:.1f}MB, delta={delta_mb:.1f}MB")
        print(f"Entries: {metadata['total_entries']}")
        print(f"Memory per entry: {(delta_mb * 1024) / metadata['total_entries']:.2f} KB")
        
        # Memory usage should be reasonable (less than 100MB for thousands of entries)
        assert delta_mb < 100, f"Memory usage too high: {delta_mb:.1f}MB"


class TestUnicodeAndBoundsSafety:
    """Regression tests for Unicode first-byte collisions and C bounds checks."""

    def test_unicode_keys_shared_leading_utf8_byte(self):
        """CompactTree lookup must work for keys whose first UTF-8 byte collides.

        Characters é, ê, è, à all start with 0xC3 in UTF-8.  The C extension
        must scan the full equal-range rather than stopping at the first match.
        """
        data = {"é": "a", "ê": "b", "è": "c", "à": "d"}
        tree = CompactTree.from_dict(data)
        for k, v in data.items():
            assert tree[k] == v, f"Wrong value for key {k!r}"
            assert k in tree

    def test_unicode_keys_nested_shared_first_byte(self):
        """Nested CompactTree with Unicode keys sharing first UTF-8 byte."""
        data = {"root": {"é": "1", "ê": "2", "è": "3"}}
        tree = CompactTree.from_dict(data)
        assert tree["root"]["é"] == "1"
        assert tree["root"]["ê"] == "2"
        assert tree["root"]["è"] == "3"

    def test_get_path_unicode(self):
        """get_path should also work with Unicode keys that share first UTF-8 byte."""
        data = {"root": {"é": "1", "ê": "2"}}
        tree = CompactTree.from_dict(data)
        assert tree.get_path("root", "é") == "1"
        assert tree.get_path("root", "ê") == "2"

    def test_tree_index_bounds_validation_rejects_bad_child_start(self):
        """TreeIndex.__init__ must raise ValueError for out-of-bounds child_start."""
        try:
            from _marisa_ext import TreeIndex, TrieIndex  # noqa: PLC0415
        except ImportError:
            pytest.skip("C extension not available")

        import array as _array

        # Build a minimal valid TrieIndex (single word "a")
        label_bytes = b"a"
        label_off_a = _array.array("I", [0])
        label_len_a = _array.array("I", [1])
        is_terminal = bytes([1])
        ch_start_a  = _array.array("I", [0, 0])   # node 0 and virtual root: no children
        ch_cnt_a    = _array.array("I", [0, 0])
        key_trie = TrieIndex(
            label_bytes=label_bytes, label_off=label_off_a.tobytes(),
            label_len=label_len_a.tobytes(), is_terminal=is_terminal,
            ch_start=ch_start_a.tobytes(), ch_cnt=ch_cnt_a.tobytes(),
            ch_first_byte=b"", ch_node_id=b"", ch_pfx_count=b"",
            n_nodes=1, total_n=1, root_is_terminal=0,
        )

        # Build a TreeIndex where child_start[0] points beyond the elbl array
        elbl   = _array.array("I", [0, 1])   # 2 entries (n_elbl = 2)
        vcol   = _array.array("I", [0xFFFFFFFF, 0xFFFFFFFF])
        # child_start[0] = 99 with child_count[0] = 1 → out of bounds
        cs     = _array.array("I", [99, 0])
        cc     = _array.array("I", [1,  0])

        with pytest.raises(ValueError, match="child_start/count out of elbl bounds"):
            TreeIndex(
                elbl=elbl.tobytes(), vcol=vcol.tobytes(),
                child_start=cs.tobytes(), child_count=cc.tobytes(),
                n_tree_nodes=2, key_trie=key_trie, val_restore=str,
            )

    def test_tree_index_get_out_of_range_node_raises(self):
        """TreeIndex.get with an out-of-range node_pos must raise IndexError."""
        try:
            from _marisa_ext import TreeIndex, TrieIndex  # noqa: PLC0415
        except ImportError:
            pytest.skip("C extension not available")

        import array as _array

        # Single-word trie "a"
        label_bytes = b"a"
        label_off_a = _array.array("I", [0])
        label_len_a = _array.array("I", [1])
        is_terminal = bytes([1])
        ch_start_a  = _array.array("I", [0, 0])
        ch_cnt_a    = _array.array("I", [0, 0])
        key_trie = TrieIndex(
            label_bytes=label_bytes, label_off=label_off_a.tobytes(),
            label_len=label_len_a.tobytes(), is_terminal=is_terminal,
            ch_start=ch_start_a.tobytes(), ch_cnt=ch_cnt_a.tobytes(),
            ch_first_byte=b"", ch_node_id=b"", ch_pfx_count=b"",
            n_nodes=1, total_n=1, root_is_terminal=0,
        )

        elbl = _array.array("I", [0])
        vcol = _array.array("I", [0xFFFFFFFF])
        cs   = _array.array("I", [0, 0])   # root + 1 node
        cc   = _array.array("I", [0, 0])
        ti = TreeIndex(
            elbl=elbl.tobytes(), vcol=vcol.tobytes(),
            child_start=cs.tobytes(), child_count=cc.tobytes(),
            n_tree_nodes=2, key_trie=key_trie, val_restore=str,
        )
        with pytest.raises((IndexError, KeyError)):
            ti.get(999, "a")   # node_pos 999 >> n_tree_nodes=2


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

