"""
Splits Python source files into searchable chunks.

Each chunk is one function, one method, one class (its own docstring/signature,
not its methods' bodies), or a fallback "top-level" chunk for module-level code
that isn't inside any function or class. Non-Python text files (.md, .txt) are
chunked by blank-line-separated paragraphs instead.

Using AST instead of naive line-splitting means a chunk is always a complete,
coherent unit (a whole function, not half of one), which is what makes the
search results actually useful as retrieval snippets.
"""

from __future__ import annotations

import ast
import os
from dataclasses import dataclass, field


@dataclass
class Chunk:
    id: str
    file_path: str
    kind: str  # "function" | "method" | "class" | "module_code" | "text"
    name: str
    start_line: int
    end_line: int
    text: str
    qualname: str = ""

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "file_path": self.file_path,
            "kind": self.kind,
            "name": self.name,
            "qualname": self.qualname,
            "start_line": self.start_line,
            "end_line": self.end_line,
            "text": self.text,
        }


def _get_source_segment(source_lines: list[str], node: ast.AST) -> str:
    start = node.lineno - 1
    end = getattr(node, "end_lineno", node.lineno)
    return "\n".join(source_lines[start:end])


def chunk_python_file(path: str, source: str) -> list[Chunk]:
    """AST-parse one Python file into function/method/class-level chunks."""
    chunks: list[Chunk] = []
    try:
        tree = ast.parse(source, filename=path)
    except SyntaxError:
        return chunks

    lines = source.splitlines()
    covered_top_level_lines: set[int] = set()

    def make_chunk(node, kind, qualname):
        text = _get_source_segment(lines, node)
        start = node.lineno
        end = getattr(node, "end_lineno", node.lineno)
        for ln in range(start, end + 1):
            covered_top_level_lines.add(ln)
        cid = f"{path}::{qualname}::{start}"
        return Chunk(
            id=cid,
            file_path=path,
            kind=kind,
            name=node.name,
            qualname=qualname,
            start_line=start,
            end_line=end,
            text=text,
        )

    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            chunks.append(make_chunk(node, "function", node.name))
        elif isinstance(node, ast.ClassDef):
            # One chunk for the class's own body (docstring + class-level
            # assignments), separate chunks for each method inside it.
            class_qual = node.name
            method_nodes = [
                n
                for n in node.body
                if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))
            ]
            method_lines: set[int] = set()
            for m in method_nodes:
                mstart, mend = m.lineno, getattr(m, "end_lineno", m.lineno)
                for ln in range(mstart, mend + 1):
                    method_lines.add(ln)
                chunks.append(
                    make_chunk(m, "method", f"{class_qual}.{m.name}")
                )

            cstart, cend = node.lineno, getattr(node, "end_lineno", node.lineno)
            class_only_lines = [
                lines[i - 1] if (i - 1) not in method_lines and i not in method_lines else ""
                for i in range(cstart, cend + 1)
            ]
            class_text = "\n".join(class_only_lines).rstrip()
            if class_text.strip():
                for ln in range(cstart, cend + 1):
                    covered_top_level_lines.add(ln)
                chunks.append(
                    Chunk(
                        id=f"{path}::{class_qual}::{cstart}",
                        file_path=path,
                        kind="class",
                        name=node.name,
                        qualname=class_qual,
                        start_line=cstart,
                        end_line=cend,
                        text=class_text,
                    )
                )

    # Anything left over (imports, module-level constants, __main__ blocks)
    # becomes one "module_code" chunk, so nothing in the file is unsearchable.
    leftover_lines = [
        (i, line)
        for i, line in enumerate(lines, start=1)
        if i not in covered_top_level_lines and line.strip()
    ]
    if leftover_lines:
        start = leftover_lines[0][0]
        end = leftover_lines[-1][0]
        text = "\n".join(line for _, line in leftover_lines)
        chunks.append(
            Chunk(
                id=f"{path}::<module>::{start}",
                file_path=path,
                kind="module_code",
                name=os.path.basename(path),
                qualname="<module>",
                start_line=start,
                end_line=end,
                text=text,
            )
        )

    return chunks


def chunk_text_file(path: str, source: str) -> list[Chunk]:
    """Chunk a non-Python text file (README, docs) by blank-line paragraphs."""
    chunks: list[Chunk] = []
    lines = source.splitlines()
    para_start = None
    buf: list[str] = []

    def flush(end_line: int):
        nonlocal buf, para_start
        if buf and any(l.strip() for l in buf):
            text = "\n".join(buf).strip()
            if text:
                chunks.append(
                    Chunk(
                        id=f"{path}::para::{para_start}",
                        file_path=path,
                        kind="text",
                        name=os.path.basename(path),
                        qualname="<text>",
                        start_line=para_start,
                        end_line=end_line,
                        text=text,
                    )
                )
        buf = []
        para_start = None

    for i, line in enumerate(lines, start=1):
        if line.strip():
            if para_start is None:
                para_start = i
            buf.append(line)
        else:
            flush(i - 1)
    flush(len(lines))
    return chunks


PYTHON_EXTENSIONS = {".py"}
TEXT_EXTENSIONS = {".md", ".txt", ".rst"}
DEFAULT_IGNORE_DIRS = {
    ".git", "__pycache__", ".venv", "venv", "node_modules", ".pytest_cache",
    "dist", "build", ".mypy_cache", "site-packages", ".eggs",
}


def chunk_repository(root: str) -> list[Chunk]:
    """Walk a directory tree and chunk every supported file under it."""
    all_chunks: list[Chunk] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = [d for d in dirnames if d not in DEFAULT_IGNORE_DIRS and not d.startswith(".")]
        for fname in filenames:
            ext = os.path.splitext(fname)[1]
            full_path = os.path.join(dirpath, fname)
            rel_path = os.path.relpath(full_path, root)
            try:
                with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
                    source = f.read()
            except OSError:
                continue

            if ext in PYTHON_EXTENSIONS:
                all_chunks.extend(chunk_python_file(rel_path, source))
            elif ext in TEXT_EXTENSIONS:
                all_chunks.extend(chunk_text_file(rel_path, source))
    return all_chunks
