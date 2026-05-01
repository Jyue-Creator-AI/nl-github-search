"""
Eval pipeline: tests 3 models on 30 natural language → GitHub query cases.
Usage:
  python3 eval.py --prompt v2
  python3 eval.py --prompt v2 --model claude   (single model for faster iteration)
Results saved to results/eval_<model>_<prompt>.json
"""

import os
import json
import time
import argparse
from datetime import date, timedelta
from pathlib import Path
from dotenv import load_dotenv
import anthropic
from openai import OpenAI
from groq import Groq

load_dotenv()

VALID_QUALIFIERS = """
VALID qualifiers ONLY (do not use any other qualifier):
- language:<name>          e.g. language:python
- stars:>N or stars:N..M   e.g. stars:>1000
- forks:>N
- topic:<name>             e.g. topic:machine-learning
- created:>YYYY-MM-DD
- pushed:>YYYY-MM-DD
- org:<name>               e.g. org:microsoft
- user:<name>
- archived:true/false
- is:public / is:private
- license:<spdx-id>        e.g. license:mit
- size:<N (kb)
- in:name / in:description / in:readme
"""

PROMPTS = {
    "v1": """You are a GitHub search query generator.
Convert the user's natural language request into a valid GitHub repository search query string.

GitHub search syntax examples:
- language:python stars:>1000
- topic:machine-learning created:>2024-01-01
- language:javascript stars:>500 pushed:>2024-06-01
- org:microsoft language:python

Rules:
- Output ONLY the query string, nothing else
- No explanation, no markdown, no quotes
- Use GitHub's official search qualifiers
- If the request is ambiguous, make a reasonable interpretation
""",

    "v2": None,  # built dynamically — includes date injection + qualifier whitelist
    "v3": None,  # built dynamically — adds strict rules + topic mappings + no-stars rule
    "v4": None,  # built dynamically — strict archived:false + AI topic + updated=pushed
    "v5": None,  # built dynamically — active=pushed enforced, no pushed when explicit created:, framework compound noun fix
}


def build_v5_prompt() -> str:
    today = date.today()
    three_months_ago = today - timedelta(days=90)
    six_months_ago = today - timedelta(days=180)
    one_month_ago = today - timedelta(days=30)
    one_week_ago = today - timedelta(days=7)
    with open("prompts/v5_system_prompt.txt") as f:
        template = f.read()
    lines = [l for l in template.split("\n") if not l.startswith("#")]
    template = "\n".join(lines)
    return template.format(
        today=today.isoformat(),
        three_months_ago=three_months_ago.isoformat(),
        six_months_ago=six_months_ago.isoformat(),
        one_month_ago=one_month_ago.isoformat(),
        one_week_ago=one_week_ago.isoformat(),
        this_year=today.year,
    )


def build_v4_prompt() -> str:
    today = date.today()
    three_months_ago = today - timedelta(days=90)
    six_months_ago = today - timedelta(days=180)
    one_month_ago = today - timedelta(days=30)
    one_week_ago = today - timedelta(days=7)
    with open("prompts/v4_system_prompt.txt") as f:
        template = f.read()
    lines = [l for l in template.split("\n") if not l.startswith("#")]
    template = "\n".join(lines)
    return template.format(
        today=today.isoformat(),
        three_months_ago=three_months_ago.isoformat(),
        six_months_ago=six_months_ago.isoformat(),
        one_month_ago=one_month_ago.isoformat(),
        one_week_ago=one_week_ago.isoformat(),
        this_year=today.year,
    )


def build_v3_prompt() -> str:
    today = date.today()
    three_months_ago = today - timedelta(days=90)
    six_months_ago = today - timedelta(days=180)
    one_month_ago = today - timedelta(days=30)
    one_week_ago = today - timedelta(days=7)
    with open("prompts/v3_system_prompt.txt") as f:
        template = f.read()
    lines = [l for l in template.split("\n") if not l.startswith("#")]
    template = "\n".join(lines)
    return template.format(
        today=today.isoformat(),
        three_months_ago=three_months_ago.isoformat(),
        six_months_ago=six_months_ago.isoformat(),
        one_month_ago=one_month_ago.isoformat(),
        one_week_ago=one_week_ago.isoformat(),
        this_year=today.year,
    )


def build_v2_prompt() -> str:
    today = date.today()
    three_months_ago = today - timedelta(days=90)
    six_months_ago = today - timedelta(days=180)
    return f"""You are a GitHub search query generator.
Convert the user's natural language request into a valid GitHub repository search query string.

Today's date is {today.isoformat()}. Use this when interpreting relative time expressions:
- "recent" / "lately" = last 3 months (after {three_months_ago.isoformat()})
- "new" = last 6 months (after {six_months_ago.isoformat()})
- "this year" = after {today.year}-01-01
- "brand new" = last 1 month (after {(today - timedelta(days=30)).isoformat()})
- "this week" = last 7 days (after {(today - timedelta(days=7)).isoformat()})

GitHub search syntax examples:
- language:python stars:>1000
- topic:machine-learning created:>{three_months_ago.isoformat()}
- language:javascript stars:>500 pushed:>{six_months_ago.isoformat()}
- org:microsoft language:python

{VALID_QUALIFIERS}

Rules:
- Output ONLY the query string, nothing else
- No explanation, no markdown, no quotes
- ONLY use qualifiers from the VALID list above
- If the request is ambiguous, make a reasonable interpretation
- popular/good = stars:>1000
- lots of = >100 (for forks/stars when no threshold given)
"""


def get_prompt(version: str) -> str:
    if version == "v2":
        return build_v2_prompt()
    if version == "v3":
        return build_v3_prompt()
    if version == "v4":
        return build_v4_prompt()
    if version == "v5":
        return build_v5_prompt()
    return PROMPTS[version]


def normalize(query: str) -> str:
    return " ".join(sorted(query.strip().lower().split()))


def query_claude(prompt: str, user_input: str) -> str:
    client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
    msg = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system=prompt,
        messages=[{"role": "user", "content": user_input}],
    )
    return msg.content[0].text.strip()


def query_openai(prompt: str, user_input: str) -> str:
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        max_tokens=200,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
    )
    return resp.choices[0].message.content.strip()


def query_llama(prompt: str, user_input: str) -> str:
    client = Groq(api_key=os.getenv("GROQ_API_KEY"))
    resp = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        max_tokens=200,
        messages=[
            {"role": "system", "content": prompt},
            {"role": "user", "content": user_input},
        ],
    )
    return resp.choices[0].message.content.strip()


MODELS = {
    "claude": ("Claude Haiku 4.5", query_claude),
    "openai": ("GPT-4o mini", query_openai),
    "llama": ("Llama 3.3 70B", query_llama),
}


def run_eval(model_key: str, cases: list, prompt: str, prompt_version: str) -> dict:
    model_name, query_fn = MODELS[model_key]
    results = []
    correct = 0

    print(f"\n{'='*50}")
    print(f"Model: {model_name} | Prompt: {prompt_version}")
    print(f"{'='*50}")

    for case in cases:
        try:
            output = query_fn(prompt, case["input"])
        except Exception as e:
            output = f"ERROR: {e}"

        norm_output = normalize(output)
        norm_truth = normalize(case["ground_truth"])
        is_correct = norm_output == norm_truth
        if is_correct:
            correct += 1

        status = "✓" if is_correct else "✗"
        print(f"[{status}] #{case['id']:02d} ({case['category']}) {case['input'][:50]}")
        if not is_correct:
            print(f"     expected: {norm_truth}")
            print(f"     got:      {norm_output}")

        results.append({
            "id": case["id"],
            "category": case["category"],
            "input": case["input"],
            "ground_truth": case["ground_truth"],
            "output": output,
            "normalized_output": norm_output,
            "normalized_truth": norm_truth,
            "correct": is_correct,
        })
        time.sleep(1)

    accuracy = correct / len(cases)
    print(f"\nAccuracy: {correct}/{len(cases)} = {accuracy:.1%}")

    result_data = {
        "model": model_name,
        "prompt_version": prompt_version,
        "accuracy": round(accuracy, 4),
        "correct": correct,
        "total": len(cases),
        "results": results,
    }

    out_path = Path("results") / f"eval_{model_key}_{prompt_version}.json"
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(result_data, f, indent=2, ensure_ascii=False)
    print(f"Saved to {out_path}")

    return result_data


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prompt", default="v2", choices=["v1", "v2", "v3", "v4", "v5"])
    parser.add_argument("--model", default="all", choices=["all", "claude", "openai", "llama"])
    args = parser.parse_args()

    with open("data/test_cases.json") as f:
        cases = json.load(f)["cases"]

    prompt = get_prompt(args.prompt)
    model_keys = list(MODELS.keys()) if args.model == "all" else [args.model]

    summary = []
    for model_key in model_keys:
        result = run_eval(model_key, cases, prompt, args.prompt)
        summary.append((result["model"], result["accuracy"]))
        if args.model == "all":
            time.sleep(2)

    print(f"\n{'='*50}")
    print("SUMMARY")
    print(f"{'='*50}")
    for model_name, acc in summary:
        status = "✓" if acc >= 0.85 else "✗ (<85%)"
        print(f"{status}  {model_name}: {acc:.1%}")


if __name__ == "__main__":
    main()
