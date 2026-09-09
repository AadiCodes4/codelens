# CodeLens

CodeLens is a semantic code search engine. Point it at a local codebase, and it finds the function or class that actually answers a question like *"where do we compute the segmentation loss"*, even when the query doesn't share exact words with the code.

It's a personal, from-scratch project. No external LLM API, no API key, no GPU, and no downloaded model weights are required to run it, the whole thing works fully offline.

## How the search actually works

Most "semantic search" demos either call an embeddings API or download a pretrained sentence-transformer model. CodeLens intentionally does neither, so it stays runnable anywhere with just `pip install`. Instead:

1. Each Python file is parsed with the `ast` module, not split by line count, so every chunk is a complete function, method, or class, never half of one.
2. Every chunk's text is TF-IDF vectorized, with identifiers first split on `snake_case` and `camelCase` boundaries, so a query for "risk score" can match code that only ever writes `risk_score`.
3. The TF-IDF matrix is reduced with Truncated SVD (latent semantic analysis), a real, decades-old semantic search technique that groups terms which tend to co-occur across the codebase, so a query and a chunk that share few exact words but related concepts still end up close together.
4. A query is embedded the same way and ranked against every chunk by cosine similarity.

This is real semantic search, not lexical keyword grep, but it is honestly a lighter-weight approach than a neural embedding model. It works best when the query shares some vocabulary with the code or its docstrings, and it's more likely to miss when the wording is completely unrelated (a documented example of this is below). That trade-off is the whole point: it's dependency-light and fully offline, not a claim that it beats a modern embedding model.

## What's actually in the repo

```
codelens/
  codelens/
    chunker.py     - AST-based chunking of Python files into functions/methods/classes
    index.py       - TF-IDF + Truncated SVD search index, with save/load
    main.py        - FastAPI app: POST /index, POST /search, GET /health, GET /stats
    cli.py         - command-line interface: codelens index <path> / search <query>
  frontend/
    index.html     - single-file React search UI (CDN React, no build step)
  tests/
    test_chunker.py - AST chunking correctness
    test_index.py    - index build/search/save/load behavior
    test_api.py       - FastAPI endpoint integration tests
  requirements.txt
  Dockerfile
  pyproject.toml
```

## Running it

```
pip install -r requirements.txt
uvicorn codelens.main:app --reload
```

Then either use the CLI directly:

```
python -m codelens.cli index /path/to/a/codebase
python -m codelens.cli search "compute the segmentation loss"
```

or open `frontend/index.html` in a browser (talks to `http://127.0.0.1:8000` by default) and use the search UI.

### With Docker

```
docker build -t codelens .
docker run --rm -p 8000:8000 codelens
```

## This was actually run, on a real codebase

Everything below is real output. Rather than index a toy example, CodeLens was pointed at [neuroscribe-imaging-pipeline](https://github.com/AadiCodes4/neuroscribe-imaging-pipeline), another one of my repos, a real FastAPI + PyTorch project it had never seen before.

```
$ python -m codelens.cli index neuroscribe-imaging-pipeline/ --components 64
Indexed 96 chunks from 14 files under 'neuroscribe-imaging-pipeline/' in 0.553s
Vocabulary size: 3721, SVD components: 64, explained variance: 0.8363
```

```
$ python -m codelens.cli search "grad cam heatmap saliency"
  [1] score=0.867  function      compute_saliency    backend/app/interpretability.py:120-141
  [2] score=0.710  function      grad_cam             backend/app/interpretability.py:55-108
  [3] score=0.590  function      segment                backend/app/main.py:142-189

$ python -m codelens.cli search "train the model for one epoch"
  [1] score=0.783  function      _load_or_train_model  backend/app/main.py:71-101
  [2] score=0.686  function      _lifespan               backend/app/main.py:48-50
  [3] score=0.577  function      train                     backend/train.py:37-85
```

Both queries correctly surfaced the right functions on top, in a codebase CodeLens had never indexed before, using only the code and docstrings themselves.

**An honest example of where it misses:** searching `"upload file to s3 storage"` against that same codebase ranked a test file above the actual `StorageClient.upload_bytes` method, because the query says "upload file" while the code says "upload bytes." Rewording the query to `"upload bytes to a bucket"` correctly puts `StorageClient.upload_bytes` in first place. That's the real trade-off of a TF-IDF-based approach over a neural embedding: it leans on shared vocabulary, and can miss when a query and the code describe the same idea in different words.

### Dogfooding: CodeLens searching its own source

```
$ python -m codelens.cli index . --components 32
Indexed 53 chunks from 9 files under '.' in 0.046s
Vocabulary size: 2453, SVD components: 32, explained variance: 0.7765

$ python -m codelens.cli search "expand snake case and camel case identifiers"
  [1] score=0.876  function  _tokenize_identifiers  codelens/index.py:41-54
```

### Tests

```
$ pytest tests/ -v
...
============================== 18 passed in 1.34s ==============================
```

18 tests, covering AST chunking edge cases (syntax errors, module-level leftover code, classes vs. their methods), index behavior (relevant queries actually rank the right file first, not just "something" comes back), save/load producing byte-identical results, and full FastAPI request/response round trips.

## Known limitations

- TF-IDF + SVD is a real semantic technique, but it is not a neural embedding model. It leans on shared vocabulary between the query and the code (see the honest example above).
- Only Python, Markdown, and plain-text files are chunked meaningfully today; other languages fall back to being un-indexed rather than badly indexed.
- No authentication on the API. This is a local developer tool, not designed to be exposed on the open internet as-is.
- The SVD-based fallback to raw TF-IDF cosine similarity on very small corpora (fewer than a few documents) is a real, tested code path, not just theoretical, see `test_empty_directory_raises_value_error` and the small-corpus handling in `index.py`, but it means search quality is necessarily weaker on a tiny codebase than a large one.

## License

MIT
