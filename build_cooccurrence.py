"""Build co-occurrence dictionaries from parsed corpus.

Creates multi-level nested dictionaries where each level represents
word co-occurrences at different granularities:
- Level 1: Words appearing in the same sentence
- Level 2: Words appearing in the same paragraph
- Level 3: Words appearing in the entire text (with frequency counts)
"""

from collections import Counter, defaultdict
from typing import Dict, List, Any, Tuple


def build_cooccurrence_dict(
    parsed_corpus: List[List[List[str]]],
    max_words: int = None,
) -> Tuple[Dict[str, Any], dict]:
    """Build a 3-level co-occurrence dictionary from parsed corpus.
    
    Structure:
        Level 1: word -> {other_word: {"sentence": True}}
        Level 2: word -> other_word -> {another_word: {"paragraph": True}}
        Level 3: word -> other_word -> another_word -> str(frequency)
    
    Args:
        parsed_corpus: Nested structure [paragraph][sentence][word]
        max_words: Optional limit on vocabulary size (uses most common words)
    
    Returns:
        Tuple of:
        - Nested dictionary with 3-level co-occurrence structure
        - Metadata dict with statistics
    """
    # First pass: collect all word frequencies to optionally filter
    word_counter = Counter()
    for paragraph in parsed_corpus:
        for sentence in paragraph:
            word_counter.update(sentence)
    
    # Optionally filter to most common words
    if max_words and len(word_counter) > max_words:
        vocab = {word for word, _ in word_counter.most_common(max_words)}
    else:
        vocab = set(word_counter.keys())
    
    # Build co-occurrence structures at each level
    # Level 1: sentence-level co-occurrences
    sentence_cooccur = defaultdict(set)
    
    # Level 2: paragraph-level co-occurrences
    paragraph_cooccur = defaultdict(set)
    
    # Level 3: corpus-level co-occurrences (all words)
    corpus_words = set()
    
    # Track nested structure for metrics
    total_entries = 0
    
    # Process corpus
    for paragraph in parsed_corpus:
        # Collect all words in this paragraph
        para_words = set()
        
        for sentence in paragraph:
            # Filter to vocabulary
            sent_words = [w for w in sentence if w in vocab]
            
            # Record sentence-level co-occurrences
            for word in sent_words:
                for other in sent_words:
                    if word != other:
                        sentence_cooccur[word].add(other)
            
            para_words.update(sent_words)
        
        # Record paragraph-level co-occurrences
        for word in para_words:
            for other in para_words:
                if word != other:
                    paragraph_cooccur[word].add(other)
        
        corpus_words.update(para_words)
    
    # Build the final nested dictionary structure
    result = {}
    
    for word1 in sorted(vocab):
        if word1 not in sentence_cooccur:
            continue
        
        level1 = {}
        
        # Add sentence-level co-occurrences
        for word2 in sorted(sentence_cooccur[word1]):
            level2 = {}
            
            # Add paragraph-level co-occurrences for this word pair
            if word2 in paragraph_cooccur:
                # Find words that co-occur with both word1 and word2 in paragraphs
                common_para = paragraph_cooccur[word1] & paragraph_cooccur[word2]
                
                for word3 in sorted(common_para):
                    if word3 != word1 and word3 != word2:
                        # Level 3: use frequency as value
                        freq = word_counter[word3]
                        level2[word3] = str(freq)
                        total_entries += 1
            
            if level2:
                level1[word2] = level2
        
        if level1:
            result[word1] = level1
    
    # Compute metadata
    metadata = {
        "vocabulary_size": len(vocab),
        "level1_keys": len(result),
        "total_entries": total_entries,
        "avg_depth": total_entries / len(result) if result else 0,
        "sentence_pairs": sum(len(v) for v in sentence_cooccur.values()),
        "paragraph_pairs": sum(len(v) for v in paragraph_cooccur.values()),
    }
    
    return result, metadata


def build_simple_cooccurrence_dict(
    parsed_corpus: List[List[List[str]]],
    max_words: int = None,
) -> Tuple[Dict[str, Any], dict]:
    """Build a simpler 3-level co-occurrence dictionary with consistent structure.
    
    This version ensures every path has exactly 3 levels for better testing.
    
    Structure:
        word1 -> word2 -> word3 -> "context"
        
    Where:
        - word1, word2 co-occur in a sentence
        - word2, word3 co-occur in a paragraph  
        - context describes the relationship
    
    Args:
        parsed_corpus: Nested structure [paragraph][sentence][word]
        max_words: Optional limit on vocabulary size (uses most common words)
    
    Returns:
        Tuple of:
        - Nested dictionary with 3-level structure
        - Metadata dict with statistics
    """
    # Collect word frequencies
    word_counter = Counter()
    for paragraph in parsed_corpus:
        for sentence in paragraph:
            word_counter.update(sentence)
    
    # Filter vocabulary if needed
    if max_words and len(word_counter) > max_words:
        vocab = {word for word, _ in word_counter.most_common(max_words)}
    else:
        vocab = set(word_counter.keys())
    
    # Build co-occurrence at different levels
    sentence_pairs = defaultdict(set)  # word -> set of co-occurring words in sentences
    paragraph_pairs = defaultdict(set)  # word -> set of co-occurring words in paragraphs
    
    for paragraph in parsed_corpus:
        # Paragraph-level words
        para_words = set()
        
        for sentence in paragraph:
            sent_words = [w for w in sentence if w in vocab]
            
            # Sentence-level co-occurrence
            for i, word in enumerate(sent_words):
                for other in sent_words[i+1:]:
                    sentence_pairs[word].add(other)
                    sentence_pairs[other].add(word)
            
            para_words.update(sent_words)
        
        # Paragraph-level co-occurrence
        para_words = sorted(para_words)
        for i, word in enumerate(para_words):
            for other in para_words[i+1:]:
                paragraph_pairs[word].add(other)
                paragraph_pairs[other].add(word)
    
    # Build 3-level nested dict
    result = {}
    total_entries = 0
    
    for word1 in sorted(sentence_pairs.keys()):
        level1 = {}
        
        for word2 in sorted(sentence_pairs[word1]):
            if word2 not in paragraph_pairs:
                continue
            
            level2 = {}
            
            for word3 in sorted(paragraph_pairs[word2]):
                if word3 != word1 and word3 != word2:
                    # Store frequency as the leaf value
                    level2[word3] = f"freq_{word_counter[word3]}"
                    total_entries += 1
            
            if level2:
                level1[word2] = level2
        
        if level1:
            result[word1] = level1
    
    metadata = {
        "vocabulary_size": len(vocab),
        "level1_keys": len(result),
        "total_entries": total_entries,
        "avg_entries_per_level1": total_entries / len(result) if result else 0,
        "unique_words_used": len(word_counter),
    }
    
    return result, metadata


if __name__ == "__main__":
    # Demo with corpus.txt
    from pathlib import Path
    from download_large_corpus import load_and_parse_corpus
    
    corpus_path = Path(__file__).parent / "corpus.txt"
    if corpus_path.exists():
        print("Building co-occurrence dict from corpus.txt...")
        parsed, parse_meta = load_and_parse_corpus(corpus_path)
        
        print(f"\nParsed corpus:")
        print(f"  Paragraphs: {parse_meta['paragraphs']}")
        print(f"  Sentences: {parse_meta['sentences']}")
        print(f"  Unique words: {parse_meta['unique_words']}")
        
        cooccur, cooccur_meta = build_simple_cooccurrence_dict(parsed, max_words=100)
        
        print(f"\nCo-occurrence dictionary:")
        print(f"  Level 1 keys: {cooccur_meta['level1_keys']}")
        print(f"  Total entries: {cooccur_meta['total_entries']}")
        print(f"  Avg entries per L1 key: {cooccur_meta['avg_entries_per_level1']:.1f}")
        
        # Show sample
        if cooccur:
            sample_key = list(cooccur.keys())[0]
            print(f"\nSample entry: {sample_key}")
            level1 = cooccur[sample_key]
            if level1:
                sample_key2 = list(level1.keys())[0]
                print(f"  -> {sample_key2}")
                level2 = level1[sample_key2]
                if level2:
                    sample_key3 = list(level2.keys())[0]
                    value = level2[sample_key3]
                    print(f"    -> {sample_key3} = {value}")
