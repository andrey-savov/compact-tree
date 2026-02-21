"""Tests for MarisaTrie."""

import pickle
import tempfile
from pathlib import Path

import pytest

from marisa_trie import MarisaTrie


class TestMarisaTrie:
    """Tests for MarisaTrie construction and basic operations."""

    def test_build_empty(self):
        """Test building a trie from empty input."""
        trie = MarisaTrie([])
        assert len(trie) == 0

    def test_build_single(self):
        """Test building a trie with a single word."""
        trie = MarisaTrie(["hello"])
        assert len(trie) == 1
        assert trie["hello"] == 0
        assert "hello" in trie

    def test_build_basic(self):
        """Test building a trie with multiple words."""
        words = ["apple", "banana", "cherry"]
        trie = MarisaTrie(words)
        
        assert len(trie) == 3
        
        # All words should be present
        for word in words:
            assert word in trie
        
        # All indices should be unique and in range [0, N)
        indices = {trie[w] for w in words}
        assert len(indices) == 3
        assert indices <= set(range(3))

    def test_duplicates_deduped(self):
        """Test that duplicates are silently removed."""
        words = ["apple", "banana", "apple", "cherry", "banana"]
        trie = MarisaTrie(words)
        
        # Should have 3 unique words
        assert len(trie) == 3
        assert "apple" in trie
        assert "banana" in trie
        assert "cherry" in trie

    def test_dense_indices(self):
        """Test that indices are dense (cover entire range [0, N))."""
        words = ["one", "two", "three", "four", "five"]
        trie = MarisaTrie(words)
        
        indices = {trie[w] for w in words}
        assert indices == set(range(5))

    def test_contains(self):
        """Test __contains__ operator."""
        trie = MarisaTrie(["foo", "bar", "baz"])
        
        assert "foo" in trie
        assert "bar" in trie
        assert "baz" in trie
        assert "qux" not in trie
        assert "" not in trie

    def test_missing_key(self):
        """Test that KeyError is raised for unknown words."""
        trie = MarisaTrie(["foo", "bar"])
        
        with pytest.raises(KeyError):
            _ = trie["baz"]
        
        with pytest.raises(KeyError):
            _ = trie[""]
        
        # Test prefix of existing word that's not in trie
        trie2 = MarisaTrie(["foobar"])
        with pytest.raises(KeyError):
            _ = trie2["foo"]  # Prefix exists in trie structure but not terminal
        
        # Test extension of existing word
        trie3 = MarisaTrie(["foo"])
        with pytest.raises(KeyError):
            _ = trie3["foobar"]  # Would go past the terminal

    def test_len(self):
        """Test __len__ matches unique word count."""
        assert len(MarisaTrie([])) == 0
        assert len(MarisaTrie(["a"])) == 1
        assert len(MarisaTrie(["a", "b", "c"])) == 3
        assert len(MarisaTrie(["a", "a", "a"])) == 1

    def test_deterministic(self):
        """Test that same input produces same index mapping."""
        words = ["apple", "banana", "cherry", "date"]
        
        trie1 = MarisaTrie(words)
        trie2 = MarisaTrie(words)
        
        for word in words:
            assert trie1[word] == trie2[word]

    def test_empty_string(self):
        """Test that empty string is a valid word."""
        trie = MarisaTrie(["", "a", "ab"])
        
        assert "" in trie
        assert len(trie) == 3
        
        # Should be able to get index for empty string
        idx = trie[""]
        assert 0 <= idx < 3
        
        # Should be able to restore empty string
        assert trie.restore_key(idx) == ""
        
        # Test __getitem__ for empty string returns same as index()
        assert trie[""] == trie.index("")

    def test_shared_prefixes(self):
        """Test words with shared prefixes."""
        words = ["ab", "abc", "abd", "a", "b"]
        trie = MarisaTrie(words)
        
        assert len(trie) == 5
        
        # All should be found
        for word in words:
            assert word in trie
        
        # Indices should be dense
        indices = {trie[w] for w in words}
        assert indices == set(range(5))
        
        # Test that non-existent words are not found
        assert "abcd" not in trie  # Extension of existing word
        assert "ac" not in trie    # Not a valid path


class TestRestoreKey:
    """Tests for reverse lookup (index -> word)."""

    def test_restore_key(self):
        """Test restore_key returns correct words."""
        words = ["apple", "banana", "cherry"]
        trie = MarisaTrie(words)
        
        # For each word, restore_key(index(word)) should return word
        for word in words:
            idx = trie[word]
            assert trie.restore_key(idx) == word

    def test_restore_key_roundtrip(self):
        """Test that index and restore_key are inverses."""
        words = ["foo", "bar", "baz", "qux"]
        trie = MarisaTrie(words)
        
        # For all indices in [0, N)
        for i in range(len(trie)):
            word = trie.restore_key(i)
            assert trie[word] == i

    def test_restore_key_out_of_range(self):
        """Test that IndexError is raised for invalid indices."""
        trie = MarisaTrie(["foo", "bar"])
        
        with pytest.raises(IndexError):
            trie.restore_key(-1)
        
        with pytest.raises(IndexError):
            trie.restore_key(2)
        
        with pytest.raises(IndexError):
            trie.restore_key(10)

    def test_restore_key_empty_trie(self):
        """Test restore_key on empty trie."""
        trie = MarisaTrie([])
        
        with pytest.raises(IndexError):
            trie.restore_key(0)

    def test_restore_key_only_empty_string(self):
        """Test restore_key when trie contains only empty string."""
        trie = MarisaTrie([""])
        
        assert len(trie) == 1
        assert trie.restore_key(0) == ""
        assert trie[""] == 0

    def test_restore_key_complex(self):
        """Test restore_key with complex word set."""
        words = ["a", "ab", "abc", "abd", "ac", "b", "ba", "bb"]
        trie = MarisaTrie(words)
        
        # Every index should restore to a word in the original set
        restored = {trie.restore_key(i) for i in range(len(trie))}
        assert restored == set(words)


class TestSerialize:
    """Tests for serialization and deserialization."""

    def test_serialize_roundtrip(self):
        """Test serialize and load preserve the trie."""
        words = ["apple", "banana", "cherry", "date", "elderberry"]
        trie = MarisaTrie(words)
        
        # Serialize to temp file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mtrie") as tmp:
            tmp_path = tmp.name
        
        try:
            trie.serialize(tmp_path)
            
            # Load back
            trie2 = MarisaTrie.load(tmp_path)
            
            # Verify all words and indices match
            assert len(trie2) == len(trie)
            for word in words:
                assert word in trie2
                assert trie2[word] == trie[word]
            
            # Verify restore_key works
            for i in range(len(trie)):
                assert trie2.restore_key(i) == trie.restore_key(i)
        
        finally:
            Path(tmp_path).unlink()

    def test_to_bytes_from_bytes(self):
        """Test to_bytes and from_bytes."""
        words = ["foo", "bar", "baz"]
        trie = MarisaTrie(words)
        
        # Serialize to bytes
        data = trie.to_bytes()
        assert isinstance(data, bytes)
        assert len(data) > 0
        
        # Deserialize
        trie2 = MarisaTrie.from_bytes(data)
        
        # Verify
        assert len(trie2) == len(trie)
        for word in words:
            assert trie2[word] == trie[word]

    def test_serialize_empty(self):
        """Test serializing an empty trie."""
        trie = MarisaTrie([])
        
        data = trie.to_bytes()
        trie2 = MarisaTrie.from_bytes(data)
        
        assert len(trie2) == 0


class TestCompression:
    """Tests for gzip compression support."""

    def test_serialize_gzip(self):
        """Test serialize with gzip compression."""
        words = ["apple", "apricot", "banana", "blueberry", "cherry"]
        trie = MarisaTrie(words)
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mtrie.gz") as tmp:
            tmp_path = tmp.name
        
        try:
            # Serialize with gzip
            trie.serialize(tmp_path, storage_options={"compression": "gzip"})
            
            # Load back with gzip
            trie2 = MarisaTrie.load(tmp_path, storage_options={"compression": "gzip"})
            
            # Verify
            assert len(trie2) == len(trie)
            for word in words:
                assert trie2[word] == trie[word]
        
        finally:
            Path(tmp_path).unlink()

    def test_unsupported_compression_serialize(self):
        """Test that unsupported compression raises error on serialize."""
        trie = MarisaTrie(["foo", "bar"])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mtrie") as tmp:
            tmp_path = tmp.name
        
        try:
            with pytest.raises(ValueError, match="Unsupported compression"):
                trie.serialize(tmp_path, storage_options={"compression": "bzip2"})
        finally:
            # Clean up if file was created
            if Path(tmp_path).exists():
                Path(tmp_path).unlink()

    def test_unsupported_compression_load(self):
        """Test that unsupported compression raises error on load."""
        trie = MarisaTrie(["foo", "bar"])
        
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mtrie") as tmp:
            tmp_path = tmp.name
        
        try:
            # First serialize normally
            trie.serialize(tmp_path)
            
            # Try to load with unsupported compression
            with pytest.raises(ValueError, match="Unsupported compression"):
                MarisaTrie.load(tmp_path, storage_options={"compression": "bzip2"})
        finally:
            Path(tmp_path).unlink()


class TestPickle:
    """Tests for pickle support."""

    def test_pickle_roundtrip(self):
        """Test that pickle preserves the trie."""
        words = ["foo", "bar", "baz", "qux"]
        trie = MarisaTrie(words)
        
        # Pickle
        data = pickle.dumps(trie)
        
        # Unpickle
        trie2 = pickle.loads(data)
        
        # Verify
        assert len(trie2) == len(trie)
        for word in words:
            assert trie2[word] == trie[word]
        
        for i in range(len(trie)):
            assert trie2.restore_key(i) == trie.restore_key(i)

    def test_pickle_empty(self):
        """Test pickling an empty trie."""
        trie = MarisaTrie([])
        
        data = pickle.dumps(trie)
        trie2 = pickle.loads(data)
        
        assert len(trie2) == 0


class TestRepr:
    """Tests for string representations."""

    def test_repr(self):
        """Test __repr__."""
        trie = MarisaTrie(["foo", "bar", "baz"])
        
        r = repr(trie)
        assert "MarisaTrie" in r
        assert "3 words" in r

    def test_repr_empty(self):
        """Test __repr__ for empty trie."""
        trie = MarisaTrie([])
        
        r = repr(trie)
        assert "MarisaTrie" in r
        assert "0 words" in r


class TestEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_single_char_words(self):
        """Test trie with single-character words."""
        words = ["a", "b", "c", "d", "e"]
        trie = MarisaTrie(words)
        
        assert len(trie) == 5
        indices = {trie[w] for w in words}
        assert indices == set(range(5))
        
        # Test lookups that don't exist
        with pytest.raises(KeyError):
            _ = trie["f"]
        with pytest.raises(KeyError):
            _ = trie["ab"]

    def test_only_empty_string(self):
        """Test trie with only empty string."""
        trie = MarisaTrie([""])
        
        assert len(trie) == 1
        assert "" in trie
        assert trie[""] == 0
        assert trie.restore_key(0) == ""

    def test_long_words(self):
        """Test trie with long words."""
        words = ["a" * 100, "b" * 100, "a" * 99 + "b"]
        trie = MarisaTrie(words)
        
        assert len(trie) == 3
        for word in words:
            assert word in trie

    def test_unicode_words(self):
        """Test trie with Unicode characters."""
        words = ["café", "naïve", "😀", "日本語"]
        trie = MarisaTrie(words)
        
        assert len(trie) == 4
        for word in words:
            assert word in trie
            idx = trie[word]
            assert trie.restore_key(idx) == word

    def test_path_compression(self):
        """Test that path compression works correctly."""
        # Words designed to trigger path compression
        words = ["abc", "abcdef", "abcdefghi"]
        trie = MarisaTrie(words)
        
        assert len(trie) == 3
        for word in words:
            assert word in trie
            idx = trie[word]
            assert trie.restore_key(idx) == word
        
        # Test lookups that should fail
        with pytest.raises(KeyError):
            _ = trie["ab"]  # Prefix not in trie
        with pytest.raises(KeyError):
            _ = trie["abcd"]  # Between two compressed paths

    def test_many_words(self):
        """Test trie with many words."""
        words = [f"word{i:04d}" for i in range(1000)]
        trie = MarisaTrie(words)
        
        assert len(trie) == 1000
        
        # Check a sample
        for i in [0, 100, 500, 999]:
            word = words[i]
            assert word in trie
            idx = trie[word]
            assert trie.restore_key(idx) == word
        
        # Verify all indices are dense
        all_indices = {trie[w] for w in words}
        assert all_indices == set(range(1000))

    def test_unicode_shared_first_utf8_byte(self):
        """Words that share the same leading UTF-8 byte must all be found.

        Many Latin-1 accented characters encode with the same first byte
        (0xC3 in UTF-8), e.g. é (U+00E9 → 0xC3 0xA9) and ê (U+00EA → 0xC3
        0xAA).  The C extension dispatches children by first UTF-8 byte, so
        it must scan the entire equal-range and use full-label comparison —
        not stop at the first candidate — to return the correct index.
        """
        # These four characters all encode with leading byte 0xC3 in UTF-8.
        words = ["é", "ê", "è", "à"]
        trie = MarisaTrie(words)
        assert len(trie) == 4
        # Every word must be found, and indices must be dense.
        indices = {trie[w] for w in words}
        assert indices == set(range(4))
        for w in words:
            assert trie.restore_key(trie[w]) == w

    def test_unicode_shared_first_utf8_byte_with_prefix(self):
        """Same first-byte collision, but words share an ASCII prefix."""
        # "cé", "cê", "cè" all have 'c' as their ASCII first byte, then a
        # multibyte character whose leading byte is 0xC3.
        words = ["cé", "cê", "cè", "ca"]
        trie = MarisaTrie(words)
        assert len(trie) == 4
        indices = {trie[w] for w in words}
        assert indices == set(range(4))
        for w in words:
            assert trie.restore_key(trie[w]) == w
