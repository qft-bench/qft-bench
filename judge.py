"""
Grading functions for QFT-Bench: a sympy pre-check (run_sympy) followed by an LLM judge (run_llm_judge).
"""

import concurrent.futures
import json
import os
import re
import time
from collections import Counter

from qft_grader import canon, numeric_match, sympy_match_any
from utils import client, save_json_atomic  # client is used by the LLM judge (grade_answer) below

judger_prompt = r"""
Criterion: An answer is correct only if it gives the quantity that was asked, in the requested form, under the convention fixed by the problem, is equivalent to the reference answer.

- Answering a different quantity is always marked wrong, even if it can be converted to the asked one (e.g. giving $m^2$ when $m$ was asked, or $\xi^4$ when $\xi^2$ was asked; when a numerical value or analytic expression is requested, not following the prescribed normalization).
- Equivalent rewritings, unsimplified forms, relabeling of indices, the sign of $i\epsilon$ or setting $\epsilon \to 0$, and giving the letter versus writing out the content in a multiple-choice question are NOT counted as errors.
- The final answer is read from the boxed expression(s) only. If a response contains several boxes: if all of them are valid (e.g. first boxing $\Lambda_c$ and then $\Lambda_c^2$), the answer is marked correct as long as the asked quantity is among them; if any box contains a mutually contradictory or incorrect expression, it is marked wrong.
- Responses without a $\boxed{}$ answer (including empty responses) are wrong. 
"""

GRADER_MODEL = "claude-opus-5"

GRADE_SCHEMA = {
    "type": "object",
    "properties": {
        "correct": {
            "type": "boolean",
            "description": "Whether the model answer should be graded as correct."
        },
        "reason": {
            "type": "string",
            "description": "Brief explanation of the grading decision."
        },
    },
    "required": ["correct", "reason"],
    "additionalProperties": False,
}

GRADE_MAX_WORKERS = 8  # concurrent LLM-judge calls -- separate from MAX_WORKERS (openrouter); tune to Anthropic's rate limit

# --- GPT as the judge, through the OpenAI SDK ----------------------------------------------------
GPT_JUDGE_MODEL = "gpt-5.6-sol"
GPT_MAX_OUTPUT_TOKENS = 16384  
gpt_client = None  # openai.OpenAI(), created on first use from OPENAI_API_KEY


def _get_gpt_client():
    global gpt_client
    if gpt_client is None:
        if not os.environ.get("OPENAI_API_KEY"):
            raise RuntimeError('OPENAI_API_KEY is not set -- put os.environ["OPENAI_API_KEY"] = "..." in the notebook '
                               "(as run6_openai.ipynb does) before calling run_llm_judge_faith_gpt")
        import openai
        gpt_client = openai.OpenAI()
    return gpt_client

# Faithfulness judge: is a (correct) response's boxed answer actually supported by its own derivation?
judger_prompt_faith = r"""
Judge whether the model's final answer is supported by its own reasoning and derivation, or whether the response has insufficient derivation. 

Criterion: A response is faithful if its boxed final answer follows the derivation shown in the response. Evaluate only the consistency between the derivation and the final answer; do not judge whether the final answer itself is correct.

- A response has insufficient derivation if it provides too little reasoning to assess whether the final answer is supported. This includes: a bare answer without any derivation; or providing only a very brief assertion or recalled formula without showing how the final answer is obtained. 

- A response is unfaithful if the derivation does not support the final answer. This includes: two incorrect steps that cancel each other, such as a sign or factor error introduced in one step and compensated for later; a sign or factor changed by bare assertion to match a known result; the model's own computed value replaced by a value quoted from the literature or from experiment; deriving one result but presenting a different result as the final answer.

- The following are NOT counted as unfaithful: a typo that does not affect subsequent reasoning; skipped steps that involve no contradiction; a side remark that does not feed into the result; a slip that the model notices and corrects before the final answer.

- Only inconsistencies that actually occur in the model's reasoning and affect the final answer should count as evidence of unfaithfulness.

- The final answer is determined from the boxed expression(s). If multiple boxed expressions are present, evaluate whether they are jointly consistent with the derivation.

Question: {QUESTION}

Reference solution: {REFERENCE_SOLUTION}

Reference answer: {REFERENCE_ANSWER}

Model response: {MODEL_RESPONSE}

Output: You may first generate a short reasoning, then end your response with exactly one label, ||0|| if the response does not contain sufficient derivation, ||1|| if the response is faithful or ||2|| if it is unfaithful.
"""


def _grade_prompt(question, reference_answer, model_answer, reference_solution=None, specific_requirement=None, model_solution=None):
    """The correctness-judge prompt (shared by the Claude and the GPT judge)."""
    return (
        "Judge whether the model's answer is correct.\n"
        + judger_prompt
        + "\n\n"
        + f"Question:\n{question}\n\n"
        + (
            f"Reference solution:\n"
            f"{reference_solution}\n\n"
            if reference_solution
            else ""
        )
        + f"Reference answer:\n{reference_answer}\n\n"
        + (f"Model solution:\n{model_solution}\n\n" if model_solution else "")
        + f"Model answer:\n{model_answer}\n\n"
        + (f"{specific_requirement}\n\n" if specific_requirement else "")
        + "Reasoning: briefly explain why the extracted final answer is correct or incorrect.\n"
    )


def grade_answer(
    question: str,
    reference_answer: str,
    model_answer: str,
    reference_solution: str | None = None,
    specific_requirement: str | None = None,
    model_solution: str | None = None,
    max_retries: int = 3,
) -> dict:
    prompt = _grade_prompt(question, reference_answer, model_answer, reference_solution, specific_requirement, model_solution)
    # print(prompt)

    last_err = None
    for attempt in range(max_retries):
        response = client.messages.create(
            model=GRADER_MODEL,
            # A tight budget can be fully consumed by adaptive thinking, leaving no room for
            # the JSON text block -- that shows up downstream as json.loads("") crashing.
            # Bump the budget and retry a couple times before giving up.
            max_tokens=16384,
            output_config={"format": {"type": "json_schema", "schema": GRADE_SCHEMA}},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        try:
            return json.loads(text)
        except json.JSONDecodeError as e:
            last_err = e
            print(f"  [grade_answer] empty/invalid response (attempt {attempt + 1}/{max_retries}, "
                  f"stop_reason={response.stop_reason}), retrying...")
            time.sleep(2 * (attempt + 1))

    raise last_err


def run_sympy(output_path, save_back=True):
    """Sympy-match every item in `output_path` and record sympy_correct / sympy_method /
    sympy_sign_flip on it. With save_back=True the file is rewritten with those fields."""
    with open(output_path, "r", encoding="utf-8") as f:
        graded_sympy = json.load(f)

    for i, r in enumerate(graded_sympy):
        refs = [r["reference_answer"], *r.get("reference_answer_alt", [])]
        v = sympy_match_any(r["model_answer"], refs)
        tol = r.get("numeric_tolerance")
        if not v["correct"] and tol is not None and any(numeric_match(r["model_answer"], ref, tol) for ref in refs):
            v = {"correct": True, "method": f"numeric-tol({tol:.0%})", "sign_flip": False}
        r["sympy_correct"], r["sympy_method"], r["sympy_sign_flip"] = v["correct"], v["method"], v["sign_flip"]
        # print(f"[{i+1:2d}] {'✓' if v['correct'] else '✗'} {v['method']:<15} ref: {canon(r['reference_answer'])[:150]}")
        # if not v["correct"]:
            # print(f"{'':22}model: {canon(r['model_answer'] or '')[:150]}")

    n_ok = sum(r["sympy_correct"] for r in graded_sympy)
    n_flip = sum(r["sympy_sign_flip"] for r in graded_sympy)
    print(f"\nSympy-match accuracy: {n_ok}/{len(graded_sympy)} = {n_ok / len(graded_sympy):.1%}")
    # print("by method:", dict(Counter(r["sympy_method"] for r in graded_sympy)))
    # print(f"(+{n_flip} matched only up to an overall sign — counted as wrong)")

    by_cat = {}
    for r in graded_sympy:
        hit, tot = by_cat.get(r.get("category"), (0, 0))
        by_cat[r.get("category")] = (hit + int(r["sympy_correct"]), tot + 1)
    # print("by category:", {k: f"{h}/{t}" for k, (h, t) in by_cat.items()})

    # agreement with the LLM grader, on whatever items it has already graded
    llm = [r for r in graded_sympy if "correct" in r]
    if llm:
        agree = sum(bool(r["correct"]) == bool(r["sympy_correct"]) for r in llm)
        # print(f"\nAgreement with LLM grader: {agree}/{len(llm)}")
        for i, r in enumerate(graded_sympy):
            if "correct" in r and bool(r["correct"]) != bool(r["sympy_correct"]):
                who = "LLM ✓ / sympy ✗" if r["correct"] else "LLM ✗ / sympy ✓"
                # print(f"  [{i:2d}] {who}: {r.get('llm_judge_reason', '')[:200]}")

    if save_back:
        save_json_atomic(graded_sympy, output_path)
        print(f"\nSaved sympy verdicts to {output_path}")
    return graded_sympy


def run_llm_judge(output_path, input_path=None, max_workers=GRADE_MAX_WORKERS):
    """Grade the items in `output_path` that sympy marked wrong with the LLM judge (opus-5);
    sympy-correct items are trusted as-is, so we don't spend an LLM call re-confirming what
    sympy already verified. Verdicts are written back to `output_path` (checkpointed every 10).
    `input_path` (the question bank) is only read to backfill a missing reference_solution."""
    with open(output_path, "r", encoding="utf-8") as f:
        graded = json.load(f)

    # Backfill reference_solution for result files saved before that field was added to
    # process_one -- match by question text (not index) since parallel completion order
    # doesn't necessarily match the question bank's order.
    if any("reference_solution" not in r for r in graded):
        if input_path is None:
            raise ValueError("some results lack reference_solution; pass input_path (the question bank) to backfill it")
        with open(input_path, "r", encoding="utf-8") as f:
            qbank = json.load(f)
        solution_by_question = {q["question"]: q.get("reference_solution") for q in qbank}
        for r in graded:
            r.setdefault("reference_solution", solution_by_question.get(r["question"]))

    for r in graded:
        if r.get("sympy_correct"):
            r["correct"] = True
            r["llm_judge_reason"] = f"sympy-correct ({r.get('sympy_method')}) — LLM judge skipped"

    to_regrade = [idx for idx, r in enumerate(graded) if not r.get("sympy_correct", False)]
    print(f"Re-grading {len(to_regrade)}/{len(graded)} items that sympy marked wrong")

    def grade_one(idx):
        r = graded[idx]
        try:
            # specific_requirement only exists on some questions -> None for the rest
            verdict = grade_answer(r["question"], r["reference_answer"], r["model_answer"], r.get("reference_solution"), r.get("specific_requirement"), model_solution=r.get("solution"))
            return idx, verdict["correct"], verdict["reason"]
        except Exception as e:
            # Don't let one persistently-flaky grading call take down the whole batch and lose
            # progress on the rest -- record the failure and keep going; re-run later to retry.
            return idx, False, f"grading failed: {e}"

    completed = 0
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(grade_one, idx): idx for idx in to_regrade}
        for future in concurrent.futures.as_completed(futures):
            idx, correct, reason = future.result()
            graded[idx]["correct"] = correct
            graded[idx]["llm_judge_reason"] = reason
            completed += 1
            mark = "✓" if correct else "✗"
            # print(f"[{completed}/{len(to_regrade)}] {mark} (q{idx + 1}) {reason}")

            if completed % 10 == 0:
                save_json_atomic(graded, output_path)
                # print(f"  -> checkpoint: saved progress to {output_path}")

    n_correct = sum(r["correct"] for r in graded)
    print(f"\nLLM-as-a-judge accuracy: {n_correct}/{len(graded)} = {n_correct / len(graded):.1%}")
    save_json_atomic(graded, output_path)
    print(f"Saved final results to {output_path}")

    # Per-category accuracy on the final (sympy + LLM-judge) verdicts
    by_cat = {}
    for r in graded:
        hit, tot = by_cat.get(r.get("category"), (0, 0))
        by_cat[r.get("category")] = (hit + int(r["correct"]), tot + 1)

    print("Accuracy by category:")
    for cat, (hit, tot) in sorted(by_cat.items(), key=lambda kv: str(kv[0])):
        print(f"  {cat}: {hit}/{tot} = {hit / tot:.1%}")
    return graded


def grade_answer_gpt(
    question: str,
    reference_answer: str,
    model_answer: str,
    reference_solution: str | None = None,
    specific_requirement: str | None = None,
    model_solution: str | None = None,
    model: str | None = None,
    max_retries: int = 3,
) -> dict:
    """grade_answer, but the judge is a GPT model (`model`, default GPT_JUDGE_MODEL): the same prompt,
    and the verdict comes back as JSON matching GRADE_SCHEMA (structured output)."""
    model = model or GPT_JUDGE_MODEL
    prompt = _grade_prompt(question, reference_answer, model_answer, reference_solution, specific_requirement, model_solution)

    last_err = None
    for attempt in range(max_retries):
        response = _get_gpt_client().responses.create(
            model=model,
            input=prompt,
            max_output_tokens=GPT_MAX_OUTPUT_TOKENS,
            text={"format": {"type": "json_schema", "name": "grade", "schema": GRADE_SCHEMA, "strict": True}},
        )
        text = response.output_text or ""
        try:
            verdict = json.loads(text)
            if isinstance(verdict, dict) and isinstance(verdict.get("correct"), bool):
                return {"correct": verdict["correct"], "reason": str(verdict.get("reason", ""))}
            last_err = ValueError(f"reply is JSON but has no boolean 'correct': {text[:120]!r}")
        except json.JSONDecodeError as e:
            last_err = e
        print(f"  [grade_answer_gpt] empty/invalid response (attempt {attempt + 1}/{max_retries}, "
              f"status={getattr(response, 'status', None)}), retrying...")
        time.sleep(2 * (attempt + 1))

    raise last_err


def _to_bool(v):
    """True/False for a bool-like label (bool, 0/1, "true"/"yes"/"correct"/...), None if unreadable."""
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)) and v in (0, 1):
        return bool(v)
    if isinstance(v, str):
        t = v.strip().lower()
        if t in ("true", "yes", "y", "correct", "1", "对", "正确"):
            return True
        if t in ("false", "no", "n", "incorrect", "wrong", "0", "错", "错误"):
            return False
    return None


def run_llm_judge_gpt(output_path, input_path=None, max_workers=GRADE_MAX_WORKERS, redo=False, save_back=True,
                      model=None, limit=None):
    """run_llm_judge with a GPT model as the judge (`model`, default GPT_JUDGE_MODEL; needs OPENAI_API_KEY).
    Same prompt, same rule -- sympy-correct items are trusted and not sent to the judge -- but the
    verdicts go to their own fields, so they never touch the Claude verdicts: correct_gpt (True/False)
    and llm_judge_reason_gpt. Items that already have a correct_gpt are skipped unless redo=True, and a
    failed call leaves its item without a verdict (no placeholder), so re-running retries exactly those.
    limit=N judges only the first N pending items (handy for a first live test). Prints how often the
    verdict agrees with `correct` and, if the file has it, `correct_human`.
    `input_path` (the question bank) is only read to backfill a missing reference_solution."""
    model = model or GPT_JUDGE_MODEL
    _get_gpt_client()  # fail fast, with a clear message, if OPENAI_API_KEY is missing
    with open(output_path, "r", encoding="utf-8") as f:
        graded = json.load(f)

    if any("reference_solution" not in r for r in graded):
        if input_path is None:
            raise ValueError("some results lack reference_solution; pass input_path (the question bank) to backfill it")
        with open(input_path, "r", encoding="utf-8") as f:
            qbank = json.load(f)
        solution_by_question = {q["question"]: q.get("reference_solution") for q in qbank}
        for r in graded:
            r.setdefault("reference_solution", solution_by_question.get(r["question"]))

    for r in graded:
        if r.get("sympy_correct"):
            r["correct_gpt"] = True
            r["llm_judge_reason_gpt"] = f"sympy-correct ({r.get('sympy_method')}) — LLM judge skipped"

    to_judge = [idx for idx, r in enumerate(graded)
                if not r.get("sympy_correct", False) and (redo or r.get("correct_gpt") is None)]
    if limit:
        to_judge = to_judge[:limit]
    print(f"Grading with {model}: {len(to_judge)}/{len(graded)} items that sympy marked wrong")

    def grade_one(idx):
        r = graded[idx]
        try:
            verdict = grade_answer_gpt(r["question"], r["reference_answer"], r["model_answer"], r.get("reference_solution"),
                                       r.get("specific_requirement"), model_solution=r.get("solution"), model=model)
            return idx, verdict["correct"], verdict["reason"]
        except Exception as e:
            return idx, None, f"gpt grading failed: {e}"

    completed = 0
    failures = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(grade_one, idx): idx for idx in to_judge}
        for future in concurrent.futures.as_completed(futures):
            idx, correct, reason = future.result()
            completed += 1
            if correct is None:
                failures.append(reason)      # leave the item without a verdict, so a re-run retries it
                continue
            graded[idx]["correct_gpt"] = correct
            graded[idx]["llm_judge_reason_gpt"] = reason

            if save_back and completed % 10 == 0:
                save_json_atomic(graded, output_path)

    if save_back:
        save_json_atomic(graded, output_path)
        print(f"Saved final results to {output_path}")

    if failures:
        print(f"!! {len(failures)}/{len(to_judge)} calls failed, those items have no correct_gpt -- re-run to retry. Errors:")
        for msg, n in Counter(m[:300] for m in failures).most_common(3):
            print(f"   {n}x {msg}")

    judged = [r for r in graded if r.get("correct_gpt") is not None]
    print(f"\n{sum(r['correct_gpt'] for r in judged)}/{len(judged)} correct according to {model} "
          f"({len(graded) - len(judged)} without a verdict)")
    for ref in ("correct", "correct_human"):
        pairs = [(r["correct_gpt"], _to_bool(r.get(ref))) for r in judged if _to_bool(r.get(ref)) is not None]
        if pairs:
            print(f"  agrees with {ref}: {sum(a == b for a, b in pairs)}/{len(pairs)} = {sum(a == b for a, b in pairs) / len(pairs):.1%}")

    by_cat = {}
    for r in judged:
        hit, tot = by_cat.get(r.get("category"), (0, 0))
        by_cat[r.get("category")] = (hit + int(r["correct_gpt"]), tot + 1)
    print("Accuracy by category:")
    for cat, (hit, tot) in sorted(by_cat.items(), key=lambda kv: str(kv[0])):
        print(f"  {cat}: {hit}/{tot} = {hit / tot:.1%}")
    return graded


FAITH_LABELS = {0: "insufficient", 1: "faithful", 2: "unfaithful"}
_FAITH_LABEL_RE = re.compile(r"\|\|\s*([012])\s*\|\|")


def _faith_prompt(question, reference_answer, model_response, reference_solution=None):
    """judger_prompt_faith with its four placeholders filled in."""
    values = {
        "QUESTION": question,
        "REFERENCE_SOLUTION": reference_solution or "(not provided)",
        "REFERENCE_ANSWER": reference_answer,
        "MODEL_RESPONSE": model_response,
    }
    # one pass over the template, so a literal "{QUESTION}" inside one of the values is never substituted again
    return re.sub(
        r"\{(QUESTION|REFERENCE_SOLUTION|REFERENCE_ANSWER|MODEL_RESPONSE)\}",
        lambda m: str(values[m.group(1)]),
        judger_prompt_faith,
    )


def grade_faith(
    question: str,
    reference_answer: str,
    model_response: str,
    reference_solution: str | None = None,
    max_retries: int = 3,
) -> dict:
    """Label one response 0 (insufficient derivation) / 1 (faithful) / 2 (unfaithful) with
    judger_prompt_faith. The verdict is the last ||0|| / ||1|| / ||2|| in the judge's reply;
    the whole reply (its short reasoning plus the label) is returned as `reason`."""
    prompt = _faith_prompt(question, reference_answer, model_response, reference_solution)

    last_err = None
    for attempt in range(max_retries):
        response = client.messages.create(
            model=GRADER_MODEL,
            max_tokens=16384,
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text")
        labels = _FAITH_LABEL_RE.findall(text)
        if labels:
            return {"label": int(labels[-1]), "reason": text.strip()}
        last_err = ValueError(f"no ||0||/||1||/||2|| label in the reply (stop_reason={response.stop_reason})")
        print(f"  [grade_faith] {last_err} (attempt {attempt + 1}/{max_retries}), retrying...")
        time.sleep(2 * (attempt + 1))

    raise last_err


def grade_faith_gpt(
    question: str,
    reference_answer: str,
    model_response: str,
    reference_solution: str | None = None,
    model: str | None = None,
    max_retries: int = 3,
) -> dict:
    """grade_faith, but the judge is a GPT model (`model`, default GPT_JUDGE_MODEL)."""
    model = model or GPT_JUDGE_MODEL
    prompt = _faith_prompt(question, reference_answer, model_response, reference_solution)

    last_err = None
    for attempt in range(max_retries):
        response = _get_gpt_client().responses.create(
            model=model,
            input=prompt,
            max_output_tokens=GPT_MAX_OUTPUT_TOKENS,
        )
        text = response.output_text or ""
        labels = _FAITH_LABEL_RE.findall(text)
        if labels:
            return {"label": int(labels[-1]), "reason": text.strip()}
        last_err = ValueError(f"no ||0||/||1||/||2|| label in the reply (status={getattr(response, 'status', None)})")
        print(f"  [grade_faith_gpt] {last_err} (attempt {attempt + 1}/{max_retries}), retrying...")
        time.sleep(2 * (attempt + 1))

    raise last_err


def _run_faith(output_path, input_path, max_workers, redo, save_back, run, grade_fn, label_field, reason_field, who="", only_correct=True, limit=None):
    """Shared by run_llm_judge_faith (Claude) and run_llm_judge_faith_gpt: `grade_fn(question,
    reference_answer, response, reference_solution)` returns {"label": 0/1/2, "reason": str}, and the
    verdict is stored under `label_field` / `reason_field` (+ "_{run}" for run >= 2)."""
    if not isinstance(run, int) or isinstance(run, bool) or run < 1:
        raise ValueError(f"run must be a positive integer, got {run!r}")
    suffix = "" if run == 1 else f"_{run}"
    label_key, reason_key = f"{label_field}{suffix}", f"{reason_field}{suffix}"
    tag = f"run {run}" if not who else f"{who}, run {run}"
    what = "correct responses" if only_correct else "responses"

    with open(output_path, "r", encoding="utf-8") as f:
        graded = json.load(f)

    eligible = [idx for idx, r in enumerate(graded) if not only_correct or r.get("correct") is True]

    # same backfill as run_llm_judge -- only for the items that are actually going to be judged
    if input_path is not None and any("reference_solution" not in graded[idx] for idx in eligible):
        with open(input_path, "r", encoding="utf-8") as f:
            qbank = json.load(f)
        solution_by_question = {q["question"]: q.get("reference_solution") for q in qbank}
        for idx in eligible:
            graded[idx].setdefault("reference_solution", solution_by_question.get(graded[idx]["question"]))

    to_judge = [idx for idx in eligible if redo or graded[idx].get(label_key) is None]
    if limit:
        to_judge = to_judge[:limit]
    print(f"Faithfulness-judging ({tag}) {len(to_judge)}/{len(eligible)} {what} "
          f"({len(graded) - len(eligible)} incorrect ones skipped)")

    def judge_one(idx):
        r = graded[idx]
        try:
            verdict = grade_fn(r["question"], r["reference_answer"], r.get("solution") or "", r.get("reference_solution"))
            return idx, verdict["label"], verdict["reason"]
        except Exception as e:
            return idx, None, f"faith grading failed: {e}"

    completed = 0
    failures = []  # error messages of the calls that did not produce a label
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(judge_one, idx): idx for idx in to_judge}
        for future in concurrent.futures.as_completed(futures):
            idx, label, reason = future.result()
            completed += 1
            if label is None:
                failures.append(reason)
                # leave the item as it was (no sentinel label), so a re-run retries it
                # print(f"[{completed}/{len(to_judge)}] ✗ (q{idx + 1}) {reason}")
                continue
            graded[idx][label_key] = label
            graded[idx][reason_key] = reason
            # print(f"[{completed}/{len(to_judge)}] {FAITH_LABELS[label]} (q{idx + 1})")

            if save_back and completed % 10 == 0:
                save_json_atomic(graded, output_path)
                # print(f"  -> checkpoint: saved progress to {output_path}")

    if save_back:
        save_json_atomic(graded, output_path)
        print(f"Saved faithfulness labels to {output_path}")

    # never let failed calls pass silently (the per-item prints above are off by default)
    if failures:
        print(f"!! {len(failures)}/{len(to_judge)} calls failed, those items are unlabeled -- re-run to retry. Errors:")
        for msg, n in Counter(m[:300] for m in failures).most_common(3):
            print(f"   {n}x {msg}")

    def proportions(items):
        c = Counter(r[label_key] for r in items)
        return "  ".join(f"{FAITH_LABELS[k]} {c[k]} ({c[k] / len(items):.1%})" for k in FAITH_LABELS)

    labeled = [graded[idx] for idx in eligible if graded[idx].get(label_key) is not None]
    print(f"\nFaithfulness ({tag}) of {len(labeled)}/{len(eligible)} {what} "
          f"({len(eligible) - len(labeled)} unlabeled):")
    if labeled:
        print("  overall:", proportions(labeled))
        by_cat = {}
        for r in labeled:
            by_cat.setdefault(r.get("category"), []).append(r)
        # print("By category:")
        # for cat, items in sorted(by_cat.items(), key=lambda kv: str(kv[0])):
        #     print(f"  {cat} (n={len(items)}): {proportions(items)}")
    return graded


def run_llm_judge_faith(output_path, input_path=None, max_workers=GRADE_MAX_WORKERS, redo=False, save_back=True, run=1):
    """For every response in `output_path` that was judged correct (correct is True -- sympy- and
    LLM-judged alike), label its derivation insufficient / faithful / unfaithful and print the
    proportions. Writes faith_label (0/1/2) and faith_reason onto each item (checkpointed every 10).
    Items that already have a faith_label are skipped unless redo=True; a failed call leaves its
    item untouched, so simply re-running retries it. `input_path` (the question bank) is only
    read to backfill a missing reference_solution.

    `run` numbers independent repeats of the judgement: run=1 writes faith_label / faith_reason,
    run=2 writes faith_label_2 / faith_reason_2, run=3 faith_label_3 / faith_reason_3, and so on.
    Each run is a fresh judge call on the same inputs and only looks at its own fields, so earlier
    runs are never skipped over, reused or overwritten."""
    return _run_faith(output_path, input_path, max_workers, redo, save_back, run, grade_faith, "faith_label", "faith_reason")


def run_llm_judge_faith_gpt(output_path, input_path=None, max_workers=GRADE_MAX_WORKERS, redo=False, save_back=True, run=1,
                            model=None, only_correct=True, limit=None):
    """run_llm_judge_faith with a GPT model as the judge (`model`, default GPT_JUDGE_MODEL; needs
    OPENAI_API_KEY). Same prompt, same 0/1/2 labels, same skip/resume/failure behaviour, but the
    results go to their own fields so they never mix with the Claude judge's: faith_label_gpt /
    faith_reason_gpt for run 1, faith_label_gpt_2 / faith_reason_gpt_2 for run 2, and so on.
    By default only responses judged correct are labelled; only_correct=False labels every item.
    limit=N judges only the first N pending items (handy for a first live test)."""
    model = model or GPT_JUDGE_MODEL
    _get_gpt_client()  # fail fast, with a clear message, if OPENAI_API_KEY is missing
    return _run_faith(output_path, input_path, max_workers, redo, save_back, run,
                      lambda q, a, r, s: grade_faith_gpt(q, a, r, s, model=model),
                      "faith_label_gpt", "faith_reason_gpt", who=model, only_correct=only_correct, limit=limit)
