import re
from pathlib import Path

INPUT_PATH = Path("electric_augmented.txt")
OUTPUT_PATH = Path("train_final.txt")

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

UNSAFE_HINTS = {
    "live",
    "arc",
    "smoke",
    "burn",
    "shock",
    "fire",
    "utility",
    "service entrance",
    "main lug",
    "meter",
    "panel buzzing",
    "energized",
    "flood",
    "sparking",
    "high voltage",
}

WORD_MAP = {
    "verify": "check",
    "inspect": "check",
    "investigate": "check",
    "utilize": "use",
    "determine": "find",
    "conductor": "wire",
    "receptacle": "outlet",
    "replace if needed": "replace if faulty",
    "de-energize": "turn off power",
    "de energize": "turn off power",
}

STOPWORDS = {
    "the",
    "a",
    "an",
    "to",
    "for",
    "of",
    "on",
    "at",
    "in",
    "with",
    "by",
    "all",
    "any",
    "that",
    "this",
    "then",
    "immediately",
    "carefully",
}


def normalize_text(s: str) -> str:
    s = re.sub(r"\s+", " ", s).strip(" .;:-")
    low = s.lower()
    for k, v in WORD_MAP.items():
        low = low.replace(k, v)
    return low


def is_unsafe_context(fault: str, cause: str, action: str) -> bool:
    combined = f"{fault} {cause} {action}".lower()
    return any(hint in combined for hint in UNSAFE_HINTS)


def split_steps(action: str) -> list[str]:
    a = normalize_text(action)
    a = re.sub(r"\band\s+", "; ", a)
    parts = [p.strip(" .;:-") for p in re.split(r"[;,.]", a) if p.strip()]
    if not parts:
        return []
    return parts[:2]


def compact_step(step: str, max_words: int = 6) -> str:
    words = [w for w in step.split() if w not in STOPWORDS]
    if not words:
        words = step.split()
    words = words[:max_words]
    if words and words[0] in {"check", "tighten", "replace", "test", "reset", "clean", "isolate", "measure", "repair"}:
        pass
    elif words:
        words[0] = words[0]
    return " ".join(words)


def standardize_action(fault: str, cause: str, action: str) -> str:
    if is_unsafe_context(fault, cause, action):
        return "Turn off power; consult technician"

    steps = split_steps(action)
    if not steps:
        return "Check wiring; repair or replace faulty part"

    steps = [compact_step(s) for s in steps if s]
    steps = [s for s in steps if s]
    if not steps:
        return "Check wiring; repair or replace faulty part"

    if len(steps) == 1:
        # Keep two-step consistency where possible.
        if "replace" in steps[0] or "repair" in steps[0]:
            steps = [steps[0], "test operation"]
        else:
            steps = [steps[0], "repair or replace faulty part"]

    action_out = "; ".join(steps[:2])

    words = action_out.split()
    if len(words) > 12:
        action_out = " ".join(words[:12]).rstrip(";,. ")

    if len(action_out.split()) < 4:
        action_out = "Check wiring; repair or replace faulty part"

    # Sentence case for consistency.
    action_out = action_out[0].upper() + action_out[1:]
    return action_out


def main() -> None:
    text = INPUT_PATH.read_text(encoding="utf-8")
    records = []

    for m in RECORD_PATTERN.finditer(text):
        input_text = re.sub(r"\s+", " ", m.group(1)).strip()
        fault = re.sub(r"\s+", " ", m.group(2)).strip()
        cause = re.sub(r"\s+", " ", m.group(3)).strip()
        action = re.sub(r"\s+", " ", m.group(4)).strip()
        new_action = standardize_action(fault, cause, action)

        records.append((input_text, fault, cause, new_action))

    if not records:
        raise SystemExit("No valid samples found in input file.")

    out_lines = []
    for input_text, fault, cause, action in records:
        out_lines.extend(
            [
                "### Instruction:",
                "Answer the electrician query",
                "",
                "### Input:",
                input_text,
                "",
                "### Response:",
                f"Fault: {fault}",
                f"Cause: {cause}",
                f"Action: {action}",
                "",
                "---",
                "",
            ]
        )

    OUTPUT_PATH.write_text("\n".join(out_lines).rstrip() + "\n", encoding="utf-8")

    action_lines = [r[3] for r in records]
    over_12 = sum(1 for a in action_lines if len(a.split()) > 12)
    over_2_steps = sum(1 for a in action_lines if len(a.split(";")) > 2)

    print(f"Samples: {len(records)}")
    print(f"Saved: {OUTPUT_PATH}")
    print(f"ActionsOver12Words: {over_12}")
    print(f"ActionsOver2Steps: {over_2_steps}")


if __name__ == "__main__":
    main()
