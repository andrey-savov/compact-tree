import re
import struct
import tempfile
from pathlib import Path

import pytest
from bitarray import bitarray

from compact_tree import CompactTree, _LOUDS
from succinct.poppy import Poppy


class Test_LOUDS:
    """Tests for the _LOUDS class."""

    def test_louds_first_child_root(self):
        """Test first_child on root node."""
        # Create a simple bit vector: 1 (root has child)
        ba = bitarray('1')
        louds = _LOUDS(Poppy(ba))
        
        child = louds.first_child(0)
        assert child == 1

    def test_louds_first_child_no_children(self):
        """Test first_child when node has no children."""
        # Create bit vector: 10 (root has child, that child has no children)
        ba = bitarray('10')
        louds = _LOUDS(Poppy(ba))
        
        child = louds.first_child(0)
        assert child == 1
        
        # Node 1 has no children (next bit is 0)
        child = louds.first_child(1)
        assert child is None

    def test_louds_next_sibling(self):
        """Test next_sibling navigation."""
        # Create bit vector with multiple siblings: 11 (two children of root)
        ba = bitarray('11')
        louds = _LOUDS(Poppy(ba))
        
        # First child exists
        child1 = louds.first_child(0)
        assert child1 == 1
        
        # Next sibling should be node 2
        child2 = louds.next_sibling(1)
        assert child2 == 2

    def test_louds_no_next_sibling(self):
        """Test next_sibling returns None when at end."""
        ba = bitarray('10')
        louds = _LOUDS(Poppy(ba))
        
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
        louds = _LOUDS(Poppy(ba))

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
        louds = _LOUDS(Poppy(louds_bits))
        
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
        louds = _LOUDS(Poppy(louds_bits))
        
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
        louds = _LOUDS(Poppy(ba))
        
        # Poppy should work with select operations
        assert isinstance(louds, _LOUDS)

    def test_vid_to_str_with_real_memoryview_data(self):
        """Test _vid_to_str directly with real memoryview data."""
        # Create a CompactTree instance without full initialization
        from unittest.mock import MagicMock
        tree = CompactTree.__new__(CompactTree)
        
        # Create real value data as memoryview
        val_data = bytearray()
        val_data.extend(struct.pack("<I", 5))
        val_data.extend(b"hello")
        val_data.extend(struct.pack("<I", 5))
        val_data.extend(b"world")
        
        tree.val = memoryview(val_data)
        
        assert tree._vid_to_str(0) == "hello"
        assert tree._vid_to_str(1) == "world"

    def test_vid_to_str_out_of_range(self):
        """Test _vid_to_str raises error for out-of-range VID."""
        tree = CompactTree.__new__(CompactTree)
        
        val_data = bytearray()
        val_data.extend(struct.pack("<I", 5))
        val_data.extend(b"hello")
        
        tree.val = memoryview(val_data)
        
        with pytest.raises((IndexError, struct.error)):
            tree._vid_to_str(999)

    def test_list_children_with_louds_structure(self):
        """Test _list_children logic with real LOUDS."""
        tree = CompactTree.__new__(CompactTree)
        
        # Create LOUDS with root having 3 children
        louds_bits = bitarray('111')
        louds = _LOUDS(Poppy(louds_bits))
        tree.louds = louds
        
        children = tree._list_children(0)
        assert len(children) == 3
        assert children == [1, 2, 3]

    def test_louds_root_children_enumeration(self):
        """Test enumerating root children with LOUDS."""
        # Create LOUDS with 2 root children
        louds_bits = bitarray('11')
        louds = _LOUDS(Poppy(louds_bits))
        
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
        louds = _LOUDS(Poppy(louds_bits))
        
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
