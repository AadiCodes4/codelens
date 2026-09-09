"""Command-line interface: `python -m codelens.cli index <path>` / `search <query>`."""

from __future__ import annotations

import argparse
import json
import os
import sys

from .index import SearchIndex

DEFAULT_INDEX_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "index.pkl")


def cmd_index(args):
    idx = SearchIndex(n_components=args.components)
    stats = idx.build(args.path)
    idx.save(args.index_path)
    print(f"Indexed {stats['num_chunks']} chunks from {stats['num_files']} files "
          f"under {args.path!r} in {stats['build_seconds']}s")
    print(f"Vocabulary size: {stats['vocabulary_size']}, "
          f"SVD components: {stats['svd_components']}, "
          f"explained variance: {stats['explained_variance_ratio']}")
    print(f"Saved index to {args.index_path}")


def cmd_search(args):
    if not os.path.exists(args.index_path):
        print(f"No index found at {args.index_path!r}. Run `index <path>` first.", file=sys.stderr)
        sys.exit(1)
    idx = SearchIndex.load(args.index_path)
    results = idx.search(args.query, top_k=args.top_k)
    if args.json:
        print(json.dumps(results, indent=2))
        return
    print(f'Top {len(results)} results for "{args.query}":\n')
    for r in results:
        loc = f"{r['file_path']}:{r['start_line']}-{r['end_line']}"
        print(f"  [{r['rank']}] score={r['score']:.3f}  {r['kind']:<12}  {r['qualname']:<30}  {loc}")
    print()


def main():
    parser = argparse.ArgumentParser(prog="codelens", description="Semantic code search CLI.")
    parser.add_argument("--index-path", default=DEFAULT_INDEX_PATH)
    sub = parser.add_subparsers(dest="command", required=True)

    p_index = sub.add_parser("index", help="Build a search index over a directory.")
    p_index.add_argument("path")
    p_index.add_argument("--components", type=int, default=128)
    p_index.set_defaults(func=cmd_index)

    p_search = sub.add_parser("search", help="Search a previously built index.")
    p_search.add_argument("query")
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--json", action="store_true")
    p_search.set_defaults(func=cmd_search)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
