"""Keep Chinese slide sources structurally aligned with the English decks."""

import re
from pathlib import Path

import pytest


SLIDES = Path(__file__).resolve().parents[1] / "slides"
DECKS = ["example", "template", *[f"lecture-{number:02}" for number in range(1, 8)]]


def fenced_blocks(source: str) -> list[str]:
    return [match.group(0) for match in re.finditer(r"(```|~~~).*?\1", source, re.DOTALL)]


def slide_ids(source: str) -> list[str]:
    return re.findall(r'<!-- \.slide:.*?\bid="([^"]+)".*?-->', source)


def link_targets(source: str) -> list[str]:
    return re.findall(r"\]\(([^\n)]+)\)", source)


def bare_urls(source: str) -> list[str]:
    urls = re.findall(r'https?://[^\s<>"\')\]，。；、（）]+', source)
    return [url.rstrip(".,;:") for url in urls]


@pytest.mark.parametrize("deck", DECKS)
def test_chinese_slides_preserve_executable_structure(deck):
    english = (SLIDES / deck / "slides.md").read_text()
    chinese = (SLIDES / deck / "slides.zh.md").read_text()

    assert re.search(r"[\u4e00-\u9fff]", chinese)
    assert chinese.count("\n---\n") == english.count("\n---\n")
    assert slide_ids(chinese) == slide_ids(english)
    assert fenced_blocks(chinese) == fenced_blocks(english)
    assert link_targets(chinese) == link_targets(english)
    assert bare_urls(chinese) == bare_urls(english)
    assert re.findall(r"`[^`\n]+`", chinese) == re.findall(r"`[^`\n]+`", english)
    assert re.findall(r"\$\$.*?\$\$", chinese, re.DOTALL) == re.findall(
        r"\$\$.*?\$\$", english, re.DOTALL
    )
