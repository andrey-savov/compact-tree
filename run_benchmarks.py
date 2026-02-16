"""Run benchmark suite for CompactTree.

This script provides a convenient interface for running benchmarks
with different corpus sizes and generating reports.
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_benchmarks(
    size: str = "small",
    download: bool = False,
    format: str = "terminal",
    compare: bool = False,
    save: bool = True,
) -> int:
    """Run the benchmark suite.
    
    Args:
        size: Corpus size ('small', 'medium', 'large')
        download: Whether to download corpus first
        format: Output format ('terminal', 'markdown', 'json')
        compare: Compare against previous benchmark
        save: Save results for future comparison
    
    Returns:
        Exit code (0 for success)
    """
    # Download corpus if requested
    if download:
        print(f"Downloading {size} corpus...")
        result = subprocess.run(
            [sys.executable, "download_large_corpus.py", "--size", size],
            cwd=Path(__file__).parent,
        )
        if result.returncode != 0:
            print("Warning: Corpus download failed, will use fallback", file=sys.stderr)
    
    # Build pytest command
    cmd = [
        sys.executable,
        "-m",
        "pytest",
        "test_compact_tree.py::TestLoadPerformance",
        "--benchmark-only",
        "-v",
    ]
    
    # Add output format options
    if format == "json":
        cmd.append("--benchmark-json=benchmark_results.json")
    
    # Add comparison
    if compare:
        cmd.append("--benchmark-compare")
    
    # Add autosave
    if save:
        cmd.append("--benchmark-autosave")
    
    # Run benchmarks
    print(f"\nRunning benchmarks...")
    print(f"Command: {' '.join(cmd)}\n")
    
    result = subprocess.run(cmd, cwd=Path(__file__).parent)
    
    return result.returncode


def generate_markdown_report() -> None:
    """Generate a markdown report from benchmark results."""
    import json
    
    results_path = Path(__file__).parent / "benchmark_results.json"
    
    if not results_path.exists():
        print("No benchmark results found. Run with --format=json first.", file=sys.stderr)
        return
    
    with open(results_path) as f:
        data = json.load(f)
    
    # Generate markdown
    md = ["# CompactTree Benchmark Results\n"]
    md.append(f"**Machine**: {data.get('machine_info', {}).get('node', 'Unknown')}")
    md.append(f"**Python**: {data.get('machine_info', {}).get('python_version', 'Unknown')}\n")
    
    md.append("## Benchmark Results\n")
    md.append("| Test | Min (ms) | Max (ms) | Mean (ms) | StdDev | Rounds |")
    md.append("|------|----------|----------|-----------|--------|--------|")
    
    for benchmark in data.get("benchmarks", []):
        name = benchmark["name"].replace("test_", "")
        stats = benchmark["stats"]
        
        md.append(
            f"| {name} | "
            f"{stats['min']*1000:.2f} | "
            f"{stats['max']*1000:.2f} | "
            f"{stats['mean']*1000:.2f} | "
            f"{stats['stddev']*1000:.4f} | "
            f"{stats['rounds']} |"
        )
    
    md.append("\n## Extra Info\n")
    for benchmark in data.get("benchmarks", []):
        if "extra_info" in benchmark:
            md.append(f"### {benchmark['name']}")
            for key, value in benchmark["extra_info"].items():
                md.append(f"- **{key}**: {value}")
            md.append("")
    
    report = "\n".join(md)
    print(report)
    
    # Save to file
    output_path = Path(__file__).parent / "BENCHMARK_RESULTS.md"
    output_path.write_text(report)
    print(f"\nReport saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Run CompactTree benchmarks",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "--size",
        choices=["small", "medium", "large"],
        default="small",
        help="Corpus size to use"
    )
    
    parser.add_argument(
        "--download",
        action="store_true",
        help="Download corpus before running benchmarks"
    )
    
    parser.add_argument(
        "--format",
        choices=["terminal", "json", "markdown"],
        default="terminal",
        help="Output format"
    )
    
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Compare against previous benchmark results"
    )
    
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Don't save results for future comparison"
    )
    
    parser.add_argument(
        "--generate-report",
        action="store_true",
        help="Generate markdown report from existing JSON results"
    )
    
    args = parser.parse_args()
    
    if args.generate_report:
        generate_markdown_report()
        return 0
    
    # Run benchmarks
    exit_code = run_benchmarks(
        size=args.size,
        download=args.download,
        format=args.format,
        compare=args.compare,
        save=not args.no_save,
    )
    
    # Generate markdown report if JSON output was requested
    if args.format == "json" and exit_code == 0:
        print("\nGenerating markdown report...")
        generate_markdown_report()
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
