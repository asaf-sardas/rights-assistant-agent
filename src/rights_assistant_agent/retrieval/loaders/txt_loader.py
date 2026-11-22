"""Load Kol Zchut pages saved as local .txt files.

Expected file format:

    Title: <page title>
    URL: <page url>

    <body text...>

Header lines ("Key: value") come first, then one empty line, then the body.
"""

from pathlib import Path

from langchain_core.documents import Document

REQUIRED_HEADERS = ("title", "url")


def parse_page(raw: str, source: str) -> Document:
    """Parse the text of one file into a Document.

    Rules:
    - Header lines look like "Key: value" and end at the first empty line.
    - Header keys are case-insensitive ("Title", "title", "TITLE" are the same).
    - "title" and "url" are required and must not be empty.
    - The body is everything after the first empty line, stripped, and must not be empty.
    - Any problem raises ValueError with a message that starts with `source`.
    """
    raw = raw.replace("\r\n", "\n")  # tolerate Windows line endings

    header, separator, body = raw.partition("\n\n")
    if not separator:
        raise ValueError(
            f"{source}: expected an empty line between the header and the body"
        )

    body = body.strip()
    if not body:
        raise ValueError(f"{source}: body is empty")

    metadata = _parse_header(header)
    for key in REQUIRED_HEADERS:
        if not metadata.get(key):  # catches both a missing key and an empty value
            raise ValueError(f"{source}: missing or empty required header '{key}'")

    metadata["source"] = source
    return Document(page_content=body, metadata=metadata)


def _parse_header(header: str) -> dict[str, str]:
    """Turn "Key: value" lines into {"key": "value"}. Lines without a colon are ignored."""
    metadata: dict[str, str] = {}
    for line in header.splitlines():
        key, colon, value = line.partition(":")  # splits at the FIRST colon only
        if colon:
            metadata[key.strip().lower()] = value.strip()
    return metadata


def load_txt_documents(data_dir: Path) -> list[Document]:
    """Read every *.txt file in `data_dir` and parse it into a Document.

    Files are processed in sorted order so results are stable between runs.
    """
    documents: list[Document] = []
    for path in sorted(data_dir.glob("*.txt")):
        raw = path.read_text(encoding="utf-8-sig")  # -sig strips a BOM if present
        documents.append(parse_page(raw, source=path.name))
    return documents
