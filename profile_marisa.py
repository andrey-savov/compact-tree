"""Profile MarisaTrie construction."""

import cProfile
import pstats
from io import StringIO
from pathlib import Path

from marisa_trie import MarisaTrie


def load_corpus_words(corpus_path: str) -> list[str]:
    """Load words from corpus file."""
    with open(corpus_path, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Split into words (simple whitespace split)
    words = text.split()
    
    # Also split by common punctuation to get more unique words
    import re
    words = re.findall(r'\b\w+\b', text.lower())
    
    return words


def build_trie():
    """Build a MarisaTrie from corpus words."""
    corpus_path = Path(__file__).parent / "corpus.txt"
    words = load_corpus_words(str(corpus_path))
    
    print(f"Building trie from {len(words)} words ({len(set(words))} unique)...")
    trie = MarisaTrie(words)
    print(f"Trie built with {len(trie)} entries")
    
    return trie


def profile_build():
    """Profile the trie building process."""
    profiler = cProfile.Profile()
    
    # Profile the build
    profiler.enable()
    trie = build_trie()
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by cumulative time)")
    print("="*80 + "\n")
    
    stats.print_stats(30)  # Top 30 functions
    
    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by total time)")
    print("="*80 + "\n")
    
    stats.sort_stats('tottime')
    stats.print_stats(30)
    
    return trie


if __name__ == "__main__":
    profile_build()
