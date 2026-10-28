import json
import os
import time
import anthropic
from getpass import getpass
import openai


PROMPT_PREFIX = "Let's think step by step and output the final answer within \\boxed{}.\n"
TEST_LIMIT = None  
# os.environ["ANTHROPIC_API_KEY"] = ""
client = anthropic.Anthropic()  # used by the LLM judge (grade_answer) below


def save_json_atomic(obj, path):
    """Write via a temp file + os.replace, so a write that gets interrupted (kernel killed,
    notebook closed mid-execution, a fresh kernel hitting a NameError before `obj` even exists,
    ...) can never leave `path` truncated to 0 bytes -- the old file stays intact until the new
    one is fully written and swapped in."""
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
    os.replace(tmp, path)



def extract_boxed_answer(text: str) -> str | None:
    """Extract the content of the last \\boxed{...} in the text (handles nested braces).

    Falls back to the bare text when the model skips \\boxed{} entirely -- it does this on
    trivial questions (answers just "2" with no derivation), which is a correct answer in the
    wrong format, not a missing one. The length/no-newline guard keeps this from ever firing on
    a genuinely truncated multi-paragraph derivation, which should still count as "no answer".
    """
    marker = "\\boxed{"
    start = text.rfind(marker)

    if start == -1:
        stripped = text.strip()
        if stripped and "\n" not in stripped and len(stripped) <= 40:
            return stripped
        return None

    i = start + len(marker)
    depth = 1
    content_start = i

    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1

    return text[content_start:i - 1] if depth == 0 else None

