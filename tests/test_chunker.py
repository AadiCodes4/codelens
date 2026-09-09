import textwrap

from codelens.chunker import chunk_python_file, chunk_text_file


SAMPLE_SOURCE = textwrap.dedent(
    '''
    """Module docstring."""

    import os

    CONSTANT = 42


    def top_level_function(x, y):
        """Adds two numbers."""
        return x + y


    class Widget:
        """A small widget."""

        count = 0

        def __init__(self, name):
            self.name = name

        def greet(self):
            return f"hello {self.name}"


    if __name__ == "__main__":
        print(top_level_function(1, 2))
    '''
).strip("\n")


def test_extracts_one_chunk_per_function():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    functions = [c for c in chunks if c.kind == "function"]
    assert len(functions) == 1
    assert functions[0].name == "top_level_function"
    assert "return x + y" in functions[0].text


def test_extracts_methods_separately_from_class_body():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    methods = [c for c in chunks if c.kind == "method"]
    method_names = {c.qualname for c in methods}
    assert method_names == {"Widget.__init__", "Widget.greet"}

    class_chunks = [c for c in chunks if c.kind == "class"]
    assert len(class_chunks) == 1
    # the class chunk should contain the docstring/attribute but NOT the
    # method bodies (those are their own chunks)
    assert "A small widget" in class_chunks[0].text
    assert "def greet" not in class_chunks[0].text


def test_module_level_code_is_still_captured():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    module_chunks = [c for c in chunks if c.kind == "module_code"]
    assert module_chunks, "leftover top-level code (imports, __main__ block) should be its own chunk"
    combined = "\n".join(c.text for c in module_chunks)
    assert "import os" in combined
    assert '__main__' in combined


def test_every_chunk_has_a_valid_line_range():
    chunks = chunk_python_file("sample.py", SAMPLE_SOURCE)
    for c in chunks:
        assert c.start_line >= 1
        assert c.end_line >= c.start_line


def test_syntax_error_returns_no_chunks_without_raising():
    chunks = chunk_python_file("broken.py", "def f(:\n  pass")
    assert chunks == []


def test_chunk_text_file_splits_on_blank_lines():
    text = "Paragraph one.\nStill paragraph one.\n\nParagraph two.\n\n\nParagraph three."
    chunks = chunk_text_file("readme.md", text)
    assert len(chunks) == 3
    assert "Paragraph one" in chunks[0].text
    assert "Paragraph two" in chunks[1].text
    assert "Paragraph three" in chunks[2].text
