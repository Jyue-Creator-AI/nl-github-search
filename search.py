"""
GitHub Repo Search CLI
Usage: python3 search.py "找最多星的 Python 機器學習專案"
"""

import os
import argparse
import requests
from datetime import date, timedelta
from dotenv import load_dotenv
import anthropic

load_dotenv()

ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY")
GITHUB_TOKEN = os.getenv("GITHUB_TOKEN")

# Exhaustive list of valid GitHub search qualifiers — prevents LLM from hallucinating
# non-existent qualifiers like "good-first-issue:>0" or "contributors:>50"
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

SYSTEM_PROMPT_TEMPLATE = """You are a GitHub search query generator.
Convert the user's natural language request into a valid GitHub repository search query string.

Today's date is {today}. Use this when interpreting relative time expressions like "recent" (last 3 months), "new" (last 6 months), or "lately" (last 3 months).

GitHub search syntax examples:
- language:python stars:>1000
- topic:machine-learning created:>{three_months_ago}
- language:javascript stars:>500 pushed:>{six_months_ago}
- org:microsoft language:python

{valid_qualifiers}

Rules:
- Output ONLY the query string, nothing else
- No explanation, no markdown, no quotes
- ONLY use qualifiers from the VALID list above
- If the request is ambiguous, make a reasonable interpretation
"""


def build_system_prompt() -> str:
    today = date.today()
    three_months_ago = today - timedelta(days=90)
    six_months_ago = today - timedelta(days=180)
    return SYSTEM_PROMPT_TEMPLATE.format(
        today=today.isoformat(),
        three_months_ago=three_months_ago.isoformat(),
        six_months_ago=six_months_ago.isoformat(),
        valid_qualifiers=VALID_QUALIFIERS,
    )


def nl_to_github_query(user_input: str) -> str:
    client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model="claude-haiku-4-5",
        max_tokens=200,
        system=build_system_prompt(),
        messages=[{"role": "user", "content": user_input}],
    )
    return message.content[0].text.strip()


def search_github(query: str, max_results: int = 5) -> list:
    headers = {
        "Authorization": f"Bearer {GITHUB_TOKEN}",
        "Accept": "application/vnd.github+json",
    }
    params = {"q": query, "sort": "stars", "order": "desc", "per_page": max_results}
    response = requests.get(
        "https://api.github.com/search/repositories",
        headers=headers,
        params=params,
        timeout=10,
    )
    response.raise_for_status()
    return response.json().get("items", [])


def format_results(repos: list, query: str) -> str:
    if not repos:
        return (
            f"No results found for query: {query}\n"
            "Possible reasons:\n"
            "  - Constraints may be too strict or contradictory\n"
            "  - Try relaxing filters (e.g. lower star count, wider date range)"
        )
    lines = []
    for i, repo in enumerate(repos, 1):
        lines.append(f"{i}. {repo['full_name']} ★{repo['stargazers_count']:,}")
        if repo.get("description"):
            lines.append(f"   {repo['description']}")
        lines.append(f"   {repo['html_url']}")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Search GitHub repos with natural language")
    parser.add_argument("query", help="Natural language search query")
    parser.add_argument("--results", type=int, default=5, help="Number of results (default: 5)")
    args = parser.parse_args()

    print(f"\nInput:  {args.query}")

    github_query = nl_to_github_query(args.query)
    print(f"Query:  {github_query}")
    print(f"{'─' * 50}")

    repos = search_github(github_query, args.results)
    print(format_results(repos, github_query))


if __name__ == "__main__":
    main()
