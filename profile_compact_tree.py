"""Profile CompactTree construction with co-occurrence data."""

import cProfile
import pstats
from pathlib import Path

from compact_tree import CompactTree
from download_large_corpus import load_and_parse_corpus
from build_cooccurrence import build_simple_cooccurrence_dict


def build_cooccurrence():
    """Build co-occurrence dict from corpus."""
    corpus_path = Path(__file__).parent / "corpus.txt"
    parsed, _ = load_and_parse_corpus(corpus_path)
    
    print("Building co-occurrence dict...")
    cooccur, metadata = build_simple_cooccurrence_dict(parsed, max_words=50)
    print(f"Dict built: {metadata['level1_keys']} L1 keys, {metadata['total_entries']} total entries")
    
    return cooccur, metadata


def build_tree(cooccur_dict):
    """Build CompactTree from co-occurrence dict."""
    print("\nBuilding CompactTree...")
    tree = CompactTree.from_dict(cooccur_dict)
    print(f"Tree built with {len(tree)} top-level keys")
    return tree


def profile_build():
    """Profile the tree building process."""
    # Build co-occurrence dict first (not profiled)
    cooccur, metadata = build_cooccurrence()
    
    # Profile the CompactTree build
    profiler = cProfile.Profile()
    
    profiler.enable()
    tree = build_tree(cooccur)
    profiler.disable()
    
    # Print stats
    stats = pstats.Stats(profiler)
    stats.strip_dirs()
    stats.sort_stats('cumulative')
    
    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by cumulative time)")
    print("="*80 + "\n")
    
    stats.print_stats(40)  # Top 40 functions
    
    print("\n" + "="*80)
    print("PROFILING RESULTS (sorted by total time)")
    print("="*80 + "\n")
    
    stats.sort_stats('tottime')
    stats.print_stats(40)
    
    # Print function call breakdown
    print("\n" + "="*80)
    print("MOST CALLED FUNCTIONS")
    print("="*80 + "\n")
    
    stats.sort_stats('calls')
    stats.print_stats(30)
    
    return tree


if __name__ == "__main__":
    profile_build()
