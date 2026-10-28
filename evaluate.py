"""
Grading functions for QFT-Bench: a sympy pre-check (run_sympy) followed by an LLM judge (run_llm_judge). 

run: 
python evaluate.py -output_path ${OUTPUT_PATH}
"""

import argparse
import json
import os
import sys
import tempfile
from datasets import load_dataset
from judge import run_sympy, run_llm_judge
from utils import extract_boxed_answer, save_json_atomic

DATASET = "LLMsHEP/QFT-Bench"
SPLIT = "test"


def add_model_answers(output_path):
    """Set `model_answer` on every entry to the content of the last \\boxed{...} in its `solution`
    (utils.extract_boxed_answer, None if there is none). run_sympy and run_llm_judge read the
    field from the file, so it is written back before they run; an existing value is overwritten."""
    with open(output_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    for i, r in enumerate(results):
        if "solution" not in r:
            sys.exit(f"entry {i} of {output_path} has no `solution` field (the model's full response)")
        r["model_answer"] = extract_boxed_answer(r["solution"] or "")

    save_json_atomic(results, output_path)
    n_none = sum(r["model_answer"] is None for r in results)
    print(f"Extracted model_answer from `solution` for {len(results)} responses ({n_none} without a \\boxed{{}} answer)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-output_path", "--output_path", dest="output_path", required=True,
                        help="JSON file with the model's responses (`solution`); the verdicts are written back into it")
    args = parser.parse_args()

    if not os.environ.get("ANTHROPIC_API_KEY"):
        sys.exit("ANTHROPIC_API_KEY is not set -- export your Claude API key before running the LLM judge.")
    if not os.path.isfile(args.output_path):
        sys.exit(f"output_path not found: {args.output_path}")

    # run_llm_judge reads the question bank from a file path (only to backfill reference_solution),
    # so dump the online dataset to a temporary JSON file for the duration of the run.
    questions = load_dataset(DATASET, split=SPLIT).to_list()
    print(f"Loaded {len(questions)} questions from {DATASET} ({SPLIT})")

    add_model_answers(args.output_path)

    with tempfile.TemporaryDirectory() as tmp:
        input_path = os.path.join(tmp, "qft_bench.json")
        with open(input_path, "w", encoding="utf-8") as f:
            json.dump(questions, f, ensure_ascii=False)

        run_sympy(args.output_path, save_back=True)
        run_llm_judge(args.output_path, input_path)


if __name__ == "__main__":
    main()
