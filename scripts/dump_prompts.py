"""Write the system prompts out exactly as the model receives them.

The stored templates under generation/prompts/ carry placeholders — the category
prompt's taxonomy list is filled in from input/categoryauto_labelling.csv at call
time — so the file on disk is not quite what gets sent. This renders each prompt
and saves the full text, for reading, diffing or pasting into a playground.

Run:  python -m scripts.dump_prompts [--out-dir DIR]
"""

import argparse
from pathlib import Path

from generation.keywords_category import render_system_instruction
from generation.llm import load_prompt

ROOT = Path(__file__).resolve().parents[1]
OUTPUT_DIR = ROOT / "output/prompts"


def rendered_prompts():
    """{name: full system prompt text} for every prompt the code sends."""
    return {
        # No placeholders: the article prompt is sent exactly as stored.
        "news_multi": load_prompt("news_multi"),
        "keywords_category": render_system_instruction(),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=OUTPUT_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for name, text in rendered_prompts().items():
        path = args.out_dir / f"{name}.txt"
        path.write_text(text, encoding="utf-8")
        print(f"{path}  ({len(text):,} chars, {text.count(chr(10)) + 1} lines)")


if __name__ == "__main__":
    main()
