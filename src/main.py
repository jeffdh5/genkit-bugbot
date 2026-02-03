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

from genkit import Genkit, Output
from genkit.plugins.google_genai import GoogleAI

load_dotenv()


# =============================================================================
# Setup
# =============================================================================


PROMPTS_DIR = Path(__file__).parent.parent / 'prompts'

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.0-flash',
    prompt_dir=PROMPTS_DIR,
)


# =============================================================================
# Types
# =============================================================================


class Issue(BaseModel):
    """A code issue found by the analyzer."""

    line: int = Field(description='Line number')
    title: str = Field(description='Brief title')
    severity: Literal['critical', 'warning', 'info']
    category: Literal['security', 'bug', 'style']
    explanation: str = Field(description='Why this is a problem')
    suggestion: str = Field(description='How to fix it')


class Analysis(BaseModel):
    """Analysis result containing issues."""

    issues: list[Issue] = Field(default_factory=list)


class ReviewResult(BaseModel):
    """Complete review result."""

    filename: str = ''
    summary: str
    issues: list[Issue]
    verdict: Literal['approve', 'request_changes', 'needs_discussion']


# Register for dotprompt files
ai.define_schema('Analysis', Analysis)


# =============================================================================
# Helpers
# =============================================================================


def create_result(issues: list[Issue], filename: str = '') -> ReviewResult:
    """Create a ReviewResult from a list of issues."""
    if not issues:
        summary = 'LGTM! No issues found.'
        verdict: Literal['approve', 'request_changes', 'needs_discussion'] = 'approve'
    else:
        counts = {'security': 0, 'bug': 0, 'style': 0}
        has_critical = False
        has_warning = False

        for issue in issues:
            counts[issue.category] += 1
            if issue.severity == 'critical':
                has_critical = True
            elif issue.severity == 'warning':
                has_warning = True

        summary = f"Found {len(issues)} issue(s): {counts['security']} security, {counts['bug']} bugs, {counts['style']} style"

        if has_critical:
            verdict = 'request_changes'
        elif has_warning:
            verdict = 'needs_discussion'
        else:
            verdict = 'approve'

    return ReviewResult(
        filename=filename,
        summary=summary,
        issues=sorted(issues, key=lambda i: i.line),
        verdict=verdict,
    )


# =============================================================================
# Flows
# =============================================================================


@ai.flow()
async def review_code(code: str, language: str = 'python') -> ReviewResult:
    """Review code for security, bugs, and style issues."""
    # Run analyzers in parallel
    security_prompt = ai.prompt('analyze_security', output=Output(schema=Analysis))
    bugs_prompt = ai.prompt('analyze_bugs', output=Output(schema=Analysis))
    style_prompt = ai.prompt('analyze_style', output=Output(schema=Analysis))

    input_data = {'code': code, 'language': language}

    security, bugs, style = await asyncio.gather(
        security_prompt(input=input_data),
        bugs_prompt(input=input_data),
        style_prompt(input=input_data),
    )

    # Combine and dedupe issues
    all_issues: list[Issue] = []
    seen: set[tuple[int, str]] = set()

    for response in [security, bugs, style]:
        for issue in response.output.issues:
            key = (issue.line, issue.title[:20])
            if key not in seen:
                seen.add(key)
                all_issues.append(issue)

    return create_result(all_issues)


@ai.flow()
async def review_diff(diff: str, filename: str, language: str = 'python') -> ReviewResult:
    """Review a git diff - only analyzes changed lines."""
    prompt = ai.prompt('analyze_diff', output=Output(schema=Analysis))
    response = await prompt(input={
        'diff': diff,
        'filename': filename,
        'language': language,
    })

    return create_result(response.output.issues, filename=filename)


# =============================================================================
# API
# =============================================================================


app = FastAPI(
    title='BugBot',
    description='AI code reviewer',
    version='1.0.0',
)


class CodeRequest(BaseModel):
    """Request to review code."""

    code: str = Field(description='Code to review')
    language: str = Field(default='python', description='Programming language')


class DiffRequest(BaseModel):
    """Request to review a diff."""

    diff: str = Field(description='Git diff in unified format')
    filename: str = Field(default='unknown', description='Filename')
    language: str = Field(default='python', description='Programming language')


@app.get('/health')
async def health() -> dict[str, str]:
    """Health check endpoint."""
    return {'status': 'ok'}


@app.post('/review')
async def review(request: CodeRequest) -> ReviewResult:
    """Review code for security, bugs, and style issues."""
    return await review_code(request.code, request.language)


@app.post('/review-pr')
async def review_pr(request: DiffRequest) -> ReviewResult:
    """Review a git diff (for GitHub Actions integration)."""
    return await review_diff(request.diff, request.filename, request.language)


# =============================================================================
# Main
# =============================================================================


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8080)
