"""Download large text corpus for load testing.

Downloads a public domain text from Project Gutenberg for benchmarking.
Caches to local file to avoid repeated downloads.
"""

import re
import sys
from pathlib import Path
from typing import List, Tuple

try:
    import requests
except ImportError:
    print("Warning: requests library not installed. Install with 'pip install requests'", file=sys.stderr)
    requests = None


CORPUS_FILES = {
    "small": {
        "name": "Alice's Adventures in Wonderland",
        "url": "https://www.gutenberg.org/files/11/11-0.txt",
        "path": "large_corpus_small.txt",
        "expected_words": 27000,
    },
    "medium": {
        "name": "Pride and Prejudice",
        "url": "https://www.gutenberg.org/files/1342/1342-0.txt",
        "path": "large_corpus_medium.txt",
        "expected_words": 120000,
    },
    "large": {
        "name": "Moby Dick",
        "url": "https://www.gutenberg.org/files/2701/2701-0.txt",
        "path": "large_corpus_large.txt",
        "expected_words": 215000,
    },
}


def download_corpus(size: str = "small", cache_dir: Path = None) -> Path:
    """Download a text corpus of specified size.
    
    Args:
        size: One of 'small', 'medium', 'large'
        cache_dir: Directory to cache downloaded files (default: current dir)
    
    Returns:
        Path to the downloaded corpus file
    
    Raises:
        ValueError: If size is invalid
        RuntimeError: If download fails
    """
    if size not in CORPUS_FILES:
        raise ValueError(f"Invalid size '{size}'. Choose from: {list(CORPUS_FILES.keys())}")
    
    if cache_dir is None:
        cache_dir = Path(__file__).parent
    
    corpus_info = CORPUS_FILES[size]
    corpus_path = cache_dir / corpus_info["path"]
    
    # Return cached file if it exists
    if corpus_path.exists():
        print(f"Using cached corpus: {corpus_path}")
        return corpus_path
    
    # Download if requests is available
    if requests is None:
        raise RuntimeError("requests library required for download. Install with 'pip install requests'")
    
    print(f"Downloading {corpus_info['name']} (~{corpus_info['expected_words']:,} words)...")
    print(f"URL: {corpus_info['url']}")
    
    try:
        response = requests.get(corpus_info["url"], timeout=30)
        response.raise_for_status()
        
        # Save to file
        corpus_path.write_text(response.text, encoding="utf-8")
        print(f"Downloaded to: {corpus_path}")
        
        # Report stats
        word_count = len(response.text.split())
        print(f"Word count: {word_count:,}")
        
        return corpus_path
        
    except requests.RequestException as e:
        raise RuntimeError(f"Failed to download corpus: {e}")


def load_and_parse_corpus(corpus_path: Path) -> Tuple[List[List[List[str]]], dict]:
    """Load corpus and parse into hierarchical structure.
    
    Args:
        corpus_path: Path to corpus text file
    
    Returns:
        Tuple of:
        - Nested structure: [text][paragraph][sentence][word]
        - Metadata dict with stats
    """
    text = corpus_path.read_text(encoding="utf-8")
    
    # Remove Project Gutenberg header/footer boilerplate
    # Header ends at "*** START OF"
    # Footer starts at "*** END OF"
    start_markers = ["*** START OF THIS PROJECT GUTENBERG", "*** START OF THE PROJECT GUTENBERG"]
    end_markers = ["*** END OF THIS PROJECT GUTENBERG", "*** END OF THE PROJECT GUTENBERG"]
    
    for marker in start_markers:
        if marker in text:
            text = text.split(marker, 1)[1]
            # Skip the rest of that line
            text = text.split("\n", 1)[1] if "\n" in text else text
            break
    
    for marker in end_markers:
        if marker in text:
            text = text.split(marker, 1)[0]
            break
    
    # Split into paragraphs (chunks separated by blank lines)
    paragraphs = re.split(r'\n\s*\n', text.strip())
    paragraphs = [p.strip() for p in paragraphs if p.strip()]
    
    # Parse each paragraph into sentences, then words
    parsed_structure = []
    total_sentences = 0
    total_words = 0
    unique_words = set()
    
    for para_text in paragraphs:
        # Split into sentences (simple split on .!?)
        sentences = re.split(r'[.!?]+', para_text)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        parsed_para = []
        for sent_text in sentences:
            # Extract words (alphanumeric + apostrophes)
            words = re.findall(r"\b[A-Za-z0-9']+\b", sent_text)
            words = [w.lower() for w in words]
            
            if words:  # Only add non-empty sentences
                parsed_para.append(words)
                total_sentences += 1
                total_words += len(words)
                unique_words.update(words)
        
        if parsed_para:  # Only add non-empty paragraphs
            parsed_structure.append(parsed_para)
    
    metadata = {
        "paragraphs": len(parsed_structure),
        "sentences": total_sentences,
        "total_words": total_words,
        "unique_words": len(unique_words),
        "file": str(corpus_path),
    }
    
    return parsed_structure, metadata


def get_fallback_corpus() -> Tuple[List[List[List[str]]], dict]:
    """Get fallback corpus from existing corpus.txt.
    
    Repeats it multiple times to get more data.
    
    Returns:
        Same format as load_and_parse_corpus
    """
    corpus_path = Path(__file__).parent / "corpus.txt"
    
    if not corpus_path.exists():
        raise FileNotFoundError("corpus.txt not found")
    
    # Parse once
    parsed, metadata = load_and_parse_corpus(corpus_path)
    
    # Duplicate it 10 times to get more data
    duplicated = parsed * 10
    
    metadata["paragraphs"] = len(duplicated)
    metadata["sentences"] = metadata["sentences"] * 10
    metadata["total_words"] = metadata["total_words"] * 10
    metadata["note"] = "Duplicated corpus.txt 10x (fallback mode)"
    
    return duplicated, metadata


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Download corpus for load testing")
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Corpus size to download"
    )
    parser.add_argument(
        "--fallback",
        action="store_true",
        help="Use fallback corpus.txt instead of downloading"
    )
    
    args = parser.parse_args()
    
    try:
        if args.fallback:
            print("Using fallback corpus...")
            parsed, metadata = get_fallback_corpus()
        else:
            corpus_path = download_corpus(args.size)
            parsed, metadata = load_and_parse_corpus(corpus_path)
        
        print("\nCorpus statistics:")
        print(f"  Paragraphs: {metadata['paragraphs']:,}")
        print(f"  Sentences: {metadata['sentences']:,}")
        print(f"  Total words: {metadata['total_words']:,}")
        print(f"  Unique words: {metadata['unique_words']:,}")
        
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
