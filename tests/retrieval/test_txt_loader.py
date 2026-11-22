"""Tests for the txt loader. These describe the expected behaviour (the spec)."""

import pytest

from rights_assistant_agent.retrieval.loaders.txt_loader import (
    load_txt_documents,
    parse_page,
)

PAGE = """Title: פיצויי פיטורים
URL: https://www.kolzchut.org.il/he/פיצויי_פיטורים

עובד שפוטר אחרי שנת עבודה אחת לפחות זכאי לפיצויי פיטורים.

סכום הפיצויים הוא משכורת חודשית אחת לכל שנת עבודה.
"""


def test_parses_title_url_and_body():
    doc = parse_page(PAGE, source="severance.txt")

    assert doc.metadata == {
        "title": "פיצויי פיטורים",
        "url": "https://www.kolzchut.org.il/he/פיצויי_פיטורים",
        "source": "severance.txt",
    }
    assert doc.page_content.startswith("עובד שפוטר")
    # the body keeps its own empty lines (paragraphs) but is stripped at the edges
    assert "\n\nסכום הפיצויים" in doc.page_content
    assert not doc.page_content.endswith("\n")


def test_header_keys_are_case_insensitive():
    raw = "title: כותרת\nurl: https://example.org\n\nגוף"
    doc = parse_page(raw, source="x.txt")
    assert doc.metadata["title"] == "כותרת"
    assert doc.metadata["url"] == "https://example.org"


@pytest.mark.parametrize("missing", ["Title", "URL"])
def test_missing_required_header_raises_with_file_name(missing):
    lines = {"Title": "Title: כותרת", "URL": "URL: https://example.org"}
    del lines[missing]
    raw = "\n".join(lines.values()) + "\n\nגוף"

    with pytest.raises(ValueError, match="broken.txt"):
        parse_page(raw, source="broken.txt")


@pytest.mark.parametrize(
    ("raw", "error"),
    [
        ("Title: A\nURL: https://a\nbody", "expected an empty line"),
        ("Title:\nURL: https://a\n\nbody", "'title'"),
        ("Title:   \nURL: https://a\n\nbody", "'title'"),
        ("Title: A\nURL:\n\nbody", "'url'"),
        ("Title: A\nURL: https://a\n\n", "body is empty"),
        ("Title: A\nURL: https://a\n\n   \n  ", "body is empty"),
    ],
    ids=[
        "no-empty-line",
        "empty-title",
        "blank-title",
        "empty-url",
        "no-body",
        "blank-body",
    ],
)
def test_invalid_pages_raise_clear_errors(raw, error):
    with pytest.raises(ValueError, match=error) as exc:
        parse_page(raw, source="bad.txt")
    assert str(exc.value).startswith("bad.txt:")


def test_windows_line_endings_are_accepted():
    doc = parse_page("Title: A\r\nURL: https://a\r\n\r\nbody", source="win.txt")
    assert doc.metadata["title"] == "A"
    assert doc.page_content == "body"


def test_colons_in_body_are_left_untouched():
    doc = parse_page(
        "Title: A\nURL: https://a\n\nשעות עבודה: 08:00-17:00", source="x.txt"
    )
    assert doc.page_content == "שעות עבודה: 08:00-17:00"


def test_load_txt_documents_reads_sorted_and_handles_bom(tmp_path):
    (tmp_path / "b.txt").write_text(
        "Title: B\nURL: https://b\n\nbody b", encoding="utf-8"
    )
    # utf-8-sig writes a BOM, like Windows Notepad sometimes does
    (tmp_path / "a.txt").write_text(
        "Title: A\nURL: https://a\n\nbody a", encoding="utf-8-sig"
    )
    (tmp_path / "notes.md").write_text("ignored", encoding="utf-8")

    docs = load_txt_documents(tmp_path)

    assert [d.metadata["source"] for d in docs] == ["a.txt", "b.txt"]
    assert docs[0].metadata["title"] == "A"  # no BOM garbage before "Title"
