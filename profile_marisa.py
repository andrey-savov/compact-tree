"""Profile MarisaTrie construction."""

import cProfile
import pstats
import random
import string
from pathlib import Path

from marisa_trie import MarisaTrie

TARGET = 500_000


def generate_words(n: int, seed: int = 42) -> list[str]:
    """Generate n unique random lowercase English-like words (4-12 chars)."""
    rng = random.Random(seed)
    # Weighted towards common English letter frequencies
    letters = "aaaaabbccddeeeeeeffgghhiiiijkllmmnnnooooppqrrssssttttuuuvwwxyz"
    words: set[str] = set()
    while len(words) < n:
        length = rng.randint(4, 12)
        word = "".join(rng.choice(letters) for _ in range(length))
        words.add(word)
    result = list(words)
    rng.shuffle(result)
    return result


def build_trie(words: list[str]) -> MarisaTrie:
    """Build a MarisaTrie from a list of words."""
    print(f"Building trie from {len(words):,} unique words...")
    trie = MarisaTrie(words)
    print(f"Trie built with {len(trie):,} entries")
    return trie


def profile_build():
    """Profile the trie building process."""
    print(f"Generating {TARGET:,} words...")
    words = generate_words(TARGET)
    print(f"Generated {len(words):,} unique words.\n")

    profiler = cProfile.Profile()

    # Profile the build
    profiler.enable()
    trie = build_trie(words)
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
