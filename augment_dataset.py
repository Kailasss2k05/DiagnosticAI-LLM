import argparse
import json
import random
import re
from typing import Dict, List


RECORD_PATTERN = re.compile(
    r"### Instruction:\s*\r?\n"
    r"Answer the electrician query\s*\r?\n\s*\r?\n"
    r"### Input:\s*\r?\n(.*?)\s*\r?\n\s*\r?\n"
    r"### Response:\s*\r?\n"
    r"Fault:\s*(.*?)\s*\r?\n"
    r"Cause:\s*(.*?)\s*\r?\n"
    r"Action:\s*(.*?)(?=\r?\n\s*\r?\n---|\Z)",
    re.DOTALL,
)

# Phrase-level replacements to keep meaning while varying wording.
PHRASE_OPTIONS = {
    "circuit breaker": ["breaker", "electrical breaker"],
    "trips": ["keeps tripping", "shuts off"],
    "immediately": ["right away", "instantly"],
    "won't": ["will not", "does not"],
    "flicker": ["blink", "flash"],
    "flickers": ["blinks", "flashes"],
    "outlet": ["socket", "power outlet"],
    "panel": ["electrical panel", "breaker panel"],
    "motor": ["motor unit", "drive motor"],
    "hot": ["overheating", "too warm"],
    "no power": ["power is off", "not getting power"],
    "not working": ["doesn't work", "stopped working"],
    "after": ["following", "once"],
    "when": ["while", "whenever"],
    "reads": ["shows", "measures"],
    "ground fault": ["ground leak", "leak to ground"],
}

PREFIX_OPTIONS = [
    "",
    "Issue: ",
    "Customer report: ",
    "Need help: ",
]

SUFFIX_OPTIONS = [
    "",
    " Please advise.",
    " Need a quick fix.",
]



def parse_records(text: str) -> List[Dict[str, str]]:
    records: List[Dict[str, str]] = []
    for m in RECORD_PATTERN.finditer(text):
        records.append(
            {
                "instruction": "Answer the electrician query",
                "input": re.sub(r"\s+", " ", m.group(1)).strip(),
                "fault": re.sub(r"\s+", " ", m.group(2)).strip(),
                "cause": re.sub(r"\s+", " ", m.group(3)).strip(),
                "action": re.sub(r"\s+", " ", m.group(4)).strip(),
            }
        )
    return records



def _replace_phrase_once(text: str, phrase: str, replacement: str) -> str:
    pattern = re.compile(re.escape(phrase), re.IGNORECASE)
    return pattern.sub(replacement, text, count=1)



def paraphrase_input(text: str, rng: random.Random) -> str:
    out = text.strip()

    # Phrase swaps.
    keys = sorted(PHRASE_OPTIONS.keys(), key=len, reverse=True)
    for phrase in keys:
        if rng.random() < 0.45 and re.search(re.escape(phrase), out, flags=re.IGNORECASE):
            out = _replace_phrase_once(out, phrase, rng.choice(PHRASE_OPTIONS[phrase]))

    # Light framing changes.
    if rng.random() < 0.35:
        out = rng.choice(PREFIX_OPTIONS) + out

    if rng.random() < 0.35:
        suffix = rng.choice(SUFFIX_OPTIONS)
        if suffix and not out.endswith((".", "?", "!")):
            out += "."
        out += suffix

    # Optional question form without changing meaning.
    if rng.random() < 0.25 and not out.endswith("?"):
        out = "How to fix this: " + out

    out = re.sub(r"\s+", " ", out).strip()
    return out



def normalize_key(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower()



def augment_records(records: List[Dict[str, str]], variants_per_sample: int, seed: int) -> List[Dict[str, str]]:
    rng = random.Random(seed)
    seen = {normalize_key(r["input"]) for r in records}

    augmented: List[Dict[str, str]] = []
    for rec in records:
        created = 0
        attempts = 0

        while created < variants_per_sample and attempts < variants_per_sample * 12:
            attempts += 1
            candidate_input = paraphrase_input(rec["input"], rng)
            key = normalize_key(candidate_input)

            if not candidate_input or key in seen:
                continue

            seen.add(key)
            created += 1
            augmented.append(
                {
                    "instruction": rec["instruction"],
                    "input": candidate_input,
                    "fault": rec["fault"],
                    "cause": rec["cause"],
                    "action": rec["action"],
                }
            )

    return augmented



def write_text_dataset(path: str, records: List[Dict[str, str]]) -> None:
    lines: List[str] = []
    for r in records:
        lines.extend(
            [
                "### Instruction:",
                "Answer the electrician query",
                "",
                "### Input:",
                r["input"],
                "",
                "### Response:",
                f"Fault: {r['fault']}",
                f"Cause: {r['cause']}",
                f"Action: {r['action']}",
                "",
                "---",
                "",
            ]
        )

    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines).rstrip() + "\n")



def write_jsonl(path: str, records: List[Dict[str, str]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            obj = {
                "instruction": r["instruction"],
                "input": r["input"],
                "output": f"Fault: {r['fault']}\\nCause: {r['cause']}\\nAction: {r['action']}",
            }
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")



def main() -> None:
    parser = argparse.ArgumentParser(description="Augment electrician dataset while keeping labels unchanged.")
    parser.add_argument("--input", default="electric.txt", help="Input dataset in prompt format")
    parser.add_argument("--output", default="electric_augmented.txt", help="Output augmented dataset file")
    parser.add_argument("--variants", type=int, default=1, help="New variants to generate per original sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for reproducible augmentation")
    parser.add_argument(
        "--keep-originals",
        action="store_true",
        help="Include original samples in output (default true)",
    )
    parser.add_argument("--jsonl", default="", help="Optional JSONL output path")
    args = parser.parse_args()

    with open(args.input, "r", encoding="utf-8") as f:
        text = f.read()

    originals = parse_records(text)
    if not originals:
        raise SystemExit("No valid samples found. Check input format.")

    augmented = augment_records(originals, variants_per_sample=max(args.variants, 0), seed=args.seed)

    if args.keep_originals or True:
        final_records = originals + augmented
    else:
        final_records = augmented

    write_text_dataset(args.output, final_records)

    if args.jsonl:
        write_jsonl(args.jsonl, final_records)

    print(f"Original samples: {len(originals)}")
    print(f"New augmented samples: {len(augmented)}")
    print(f"Total output samples: {len(final_records)}")
    print(f"Saved: {args.output}")
    if args.jsonl:
        print(f"Saved JSONL: {args.jsonl}")


if __name__ == "__main__":
    main()
