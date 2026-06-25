"""Tests for the universal file reader (multi-format extraction)."""

import csv

import pytest

from sabi.filereader import read_any


def test_reads_plain_text(tmp_path):
    p = tmp_path / "notes.txt"
    p.write_text("hello from sabi")
    assert "hello from sabi" in read_any(p)


def test_reads_csv(tmp_path):
    p = tmp_path / "data.csv"
    p.write_text("name,score\nAda,91\n")
    out = read_any(p)
    assert "Ada" in out and "91" in out


def test_reads_json(tmp_path):
    p = tmp_path / "cfg.json"
    p.write_text('{"a": 1, "b": [2, 3]}')
    out = read_any(p)
    assert '"a": 1' in out


def test_missing_file(tmp_path):
    assert "not found" in read_any(tmp_path / "nope.pdf").lower()


def test_truncation(tmp_path):
    p = tmp_path / "big.txt"
    p.write_text("x" * 10000)
    out = read_any(p, max_chars=500)
    assert "truncated" in out.lower()
    assert len(out) < 700


def test_reads_docx(tmp_path):
    docx = pytest.importorskip("docx")
    p = tmp_path / "doc.docx"
    d = docx.Document()
    d.add_paragraph("Quarterly revenue grew by twenty percent.")
    d.save(str(p))
    assert "twenty percent" in read_any(p)


def test_reads_xlsx(tmp_path):
    openpyxl = pytest.importorskip("openpyxl")
    p = tmp_path / "s.xlsx"
    wb = openpyxl.Workbook(); ws = wb.active
    ws.append(["Region", "Sales"]); ws.append(["Lagos", 140])
    wb.save(str(p))
    out = read_any(p)
    assert "Lagos" in out and "140" in out


def test_reads_html(tmp_path):
    pytest.importorskip("bs4")
    p = tmp_path / "page.html"
    p.write_text("<html><body><h1>Title</h1><p>Body text here</p></body></html>")
    out = read_any(p)
    assert "Body text here" in out
