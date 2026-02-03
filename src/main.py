"""BugBot: AI Code Reviewer.

Run:
    genkit start -- python src/main.py

Test:
    curl localhost:8080/review -d '{"code": "password = \"admin123\""}'
    curl localhost:8080/review-pr -d '{"diff": "+API_KEY = \"secret\"", "filename": "config.py"}'

Traces:
    Open http://localhost:4000
"""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

# New typed output imports!
from genkit import Genkit, Output
from genkit.plugins.google_genai import GoogleAI

load_dotenv()


# =============================================================================
# Setup
# =============================================================================

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.0-flash',
    prompt_dir=Path(__file__).parent.parent / 'prompts',
)


# =============================================================================
# Types - These get runtime validation via response.output!
# =============================================================================


class Issue(BaseModel):
    """A code issue found by the analyzer."""

    line: int
    title: str
    severity: Literal['critical', 'warning', 'info']
    category: Literal['security', 'bug', 'style']
    explanation: str
    suggestion: str


class Analysis(BaseModel):
    """Analysis result - response.output returns THIS type, not a dict!"""

    issues: list[Issue] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Complete review result."""

    filename: str = ''
    summary: str
    issues: list[Issue]
    verdict: Literal['approve', 'request_changes', 'needs_discussion']


# Register schema for dotprompt files
ai.define_schema('Analysis', Analysis)


# =============================================================================
# Flows
# =============================================================================


def summarize(issues: list[Issue], filename: str = '') -> ReviewResult:
    """Create a summary from issues."""
    if not issues:
        return ReviewResult(filename=filename, summary='LGTM!', issues=[], verdict='approve')

    counts = {'security': 0, 'bug': 0, 'style': 0}
    for i in issues:
        counts[i.category] += 1

    has_critical = any(i.severity == 'critical' for i in issues)
    has_warning = any(i.severity == 'warning' for i in issues)

    verdict: Literal['approve', 'request_changes', 'needs_discussion']
    if has_critical:
        verdict = 'request_changes'
    elif has_warning:
        verdict = 'needs_discussion'
    else:
        verdict = 'approve'

    summary = f"Found {len(issues)} issue(s): {counts['security']} security, {counts['bug']} bugs, {counts['style']} style"
    return ReviewResult(filename=filename, summary=summary, issues=sorted(issues, key=lambda i: i.line), verdict=verdict)


@ai.flow()
async def review_code(code: str, language: str = 'python') -> ReviewResult:
    """Review code for security, bugs, and style issues."""
    # Get typed prompts - Output(schema=Analysis) gives us static + runtime typing!
    security = ai.prompt('analyze_security', output=Output(schema=Analysis))
    bugs = ai.prompt('analyze_bugs', output=Output(schema=Analysis))
    style = ai.prompt('analyze_style', output=Output(schema=Analysis))

    # Run in parallel
    results = await asyncio.gather(
        security(input={'code': code, 'language': language}),
        bugs(input={'code': code, 'language': language}),
        style(input={'code': code, 'language': language}),
    )

    # response.output is now a real Analysis instance, not a dict!
    all_issues: list[Issue] = []
    seen: set[tuple[int, str]] = set()
    for response in results:
        for issue in response.output.issues:  # <- Fully typed!
            key = (issue.line, issue.title[:20])
            if key not in seen:
                seen.add(key)
                all_issues.append(issue)

    return summarize(all_issues)


@ai.flow()
async def review_diff(diff: str, filename: str, language: str = 'python') -> ReviewResult:
    """Review a git diff - only analyzes changed lines."""
    # Typed prompt with Output(schema=...)
    prompt = ai.prompt('analyze_diff', output=Output(schema=Analysis))
    response = await prompt(input={'diff': diff, 'filename': filename, 'language': language})

    # response.output is Analysis, not dict!
    return summarize(response.output.issues, filename=filename)


# =============================================================================
# API
# =============================================================================

app = FastAPI(title='BugBot', description='AI code reviewer with typed outputs')


class CodeRequest(BaseModel):
    code: str
    language: str = 'python'


class DiffRequest(BaseModel):
    diff: str
    filename: str = 'unknown'
    language: str = 'python'


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.post('/review')
async def review(req: CodeRequest) -> ReviewResult:
    return await review_code(req.code, req.language)


@app.post('/review-pr')
async def review_pr(req: DiffRequest) -> ReviewResult:
    return await review_diff(req.diff, req.filename, req.language)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
