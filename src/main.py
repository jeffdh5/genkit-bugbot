# Copyright 2025 Google LLC
# SPDX-License-Identifier: Apache-2.0

"""BugBot: AI Code Reviewer with Full Observability.

Run:
    genkit start -- uv run src/main.py

Try it:
    curl localhost:8080/review -d '{"code": "def login(user, pw): return db.query(f\"SELECT * FROM users WHERE password={pw}\")"}'

Then open http://localhost:4000 - see exactly WHY it flagged the SQL injection.
"""

import asyncio
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from genkit import Genkit, Output
from genkit.plugins.google_genai import GoogleAI  # pyright: ignore[reportMissingImports]

_ = load_dotenv()

# =============================================================================
# Genkit
# =============================================================================

PROMPTS_DIR = Path(__file__).parent.parent / 'prompts'

ai = Genkit(
    plugins=[GoogleAI()],
    model='googleai/gemini-2.0-flash',
    prompt_dir=PROMPTS_DIR,
)


# =============================================================================
# Models
# =============================================================================

Severity = Literal['critical', 'warning', 'info']
Category = Literal['security', 'bug', 'style']


class Issue(BaseModel):
    """A single issue found in the code."""

    line: int = Field(description='Line number where the issue occurs')
    title: str = Field(description='Brief title like "SQL Injection Risk"')
    severity: Severity
    category: Category = Field(description='Type of issue')
    explanation: str = Field(description='Why this is a problem')
    suggestion: str = Field(description='How to fix it')


class Analysis(BaseModel):
    """Analysis results from any analyzer."""

    issues: list[Issue] = Field(default_factory=list)


class ReviewRequest(BaseModel):
    """API request."""

    code: str = Field(..., min_length=1, max_length=10000, description='Code to review')
    language: str = Field(default='python', description='Programming language')


class ReviewResponse(BaseModel):
    """Complete code review."""

    summary: str = Field(description='One-line summary')
    issues: list[Issue] = Field(description='All issues found')
    verdict: Literal['approve', 'request_changes', 'needs_discussion']


class CodeInput(BaseModel):
    """Input for code analysis."""

    code: str
    language: str = 'python'


class DiffInput(BaseModel):
    """Input for diff-based code analysis."""

    diff: str = Field(description='Git diff in unified format')
    filename: str = Field(description='Name of the file being changed')
    language: str = 'python'


class PRReviewRequest(BaseModel):
    """API request for PR review."""

    diff: str = Field(..., min_length=1, max_length=50000, description='Git diff in unified format')
    filename: str = Field(default='unknown', description='Filename')
    language: str = Field(default='python', description='Programming language')


class PRReviewResponse(BaseModel):
    """PR review response."""

    filename: str
    summary: str
    issues: list[Issue]
    verdict: Literal['approve', 'request_changes', 'needs_discussion']


# Register schema for use in dotprompt files
_ = ai.define_schema('AnalyzerIssues', Analysis)


# =============================================================================
# Flow - Each analyzer runs in parallel, all traced in Dev UI
# =============================================================================

def dedupe_issues(issues: list[Issue]) -> list[Issue]:
    """Remove duplicate issues (same line + similar title)."""
    seen: set[tuple[int, str]] = set()
    unique: list[Issue] = []
    for issue in issues:
        key = (issue.line, issue.title.lower().split()[0])
        if key not in seen:
            seen.add(key)
            unique.append(issue)
    return unique


@ai.flow()
async def review_code(input: CodeInput) -> ReviewResponse:
    """BugBot code review flow. Open localhost:4000 to see each analyzer's trace."""
    security, bugs, style = await asyncio.gather(
        analyze_security(input),
        analyze_bugs(input),
        analyze_style(input),
    )

    all_issues = dedupe_issues(security.issues + bugs.issues + style.issues)
    all_issues.sort(key=lambda i: i.line)

    has_critical = any(i.severity == 'critical' for i in all_issues)
    has_warnings = any(i.severity == 'warning' for i in all_issues)

    if has_critical:
        verdict: Literal['approve', 'request_changes', 'needs_discussion'] = 'request_changes'
    elif has_warnings:
        verdict = 'needs_discussion'
    else:
        verdict = 'approve'

    if not all_issues:
        summary = 'No issues found. Code looks good!'
    else:
        by_cat = {'security': 0, 'bug': 0, 'style': 0}
        for i in all_issues:
            by_cat[i.category] += 1
        summary = f"Found {len(all_issues)} issue(s): {by_cat['security']} security, {by_cat['bug']} bugs, {by_cat['style']} style"

    return ReviewResponse(summary=summary, issues=all_issues, verdict=verdict)


@ai.flow()
async def analyze_security(input: CodeInput) -> Analysis:
    """Check for security vulnerabilities only."""
    prompt = ai.prompt('analyze_security', output=Output(schema=Analysis))
    response = await prompt(input={'code': input.code, 'language': input.language})
    return response.output  # Fully typed - both static AND runtime!


@ai.flow()
async def analyze_bugs(input: CodeInput) -> Analysis:
    """Check for bugs and logic errors only."""
    prompt = ai.prompt('analyze_bugs', output=Output(schema=Analysis))
    response = await prompt(input={'code': input.code, 'language': input.language})
    return response.output  # Fully typed - both static AND runtime!


@ai.flow()
async def analyze_style(input: CodeInput) -> Analysis:
    """Check for style and readability issues only."""
    prompt = ai.prompt('analyze_style', output=Output(schema=Analysis))
    response = await prompt(input={'code': input.code, 'language': input.language})
    return response.output  # Fully typed - both static AND runtime!


# =============================================================================
# Diff-based Review (for PR reviews)
# =============================================================================


@ai.flow()
async def review_diff(input: DiffInput) -> PRReviewResponse:
    """Review a git diff for issues. Focuses only on changed lines."""
    prompt = ai.prompt('analyze_diff', output=Output(schema=Analysis))
    response = await prompt(input={
        'diff': input.diff,
        'filename': input.filename,
        'language': input.language,
    })
    
    issues = response.output.issues
    issues.sort(key=lambda i: i.line)
    
    has_critical = any(i.severity == 'critical' for i in issues)
    has_warnings = any(i.severity == 'warning' for i in issues)
    
    if has_critical:
        verdict: Literal['approve', 'request_changes', 'needs_discussion'] = 'request_changes'
    elif has_warnings:
        verdict = 'needs_discussion'
    else:
        verdict = 'approve'
    
    if not issues:
        summary = 'No issues found in changed code. LGTM!'
    else:
        by_cat = {'security': 0, 'bug': 0, 'style': 0}
        for i in issues:
            by_cat[i.category] += 1
        summary = f"Found {len(issues)} issue(s) in diff: {by_cat['security']} security, {by_cat['bug']} bugs, {by_cat['style']} style"
    
    return PRReviewResponse(
        filename=input.filename,
        summary=summary,
        issues=issues,
        verdict=verdict,
    )


# =============================================================================
# FastAPI
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    yield


app = FastAPI(
    title='BugBot',
    description='AI code reviewer. See /docs for API, localhost:4000 for traces.',
    version='1.0.0',
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


@app.get('/health')
async def health() -> dict[str, str]:
    return {'status': 'ok'}


@app.get('/demo')
async def demo() -> ReviewResponse:
    """One-click demo - reviews intentionally bad code."""
    bad_code = '''def login(username, password):
    query = f"SELECT * FROM users WHERE user='{username}' AND pass='{password}'"
    return db.execute(query)
'''
    return await review_code(CodeInput(code=bad_code, language='python'))


@app.post('/review', response_model=ReviewResponse)
async def review_endpoint(request: ReviewRequest) -> ReviewResponse:
    """
    Review code for security issues, bugs, and style problems.

    Run with `genkit start` and check localhost:4000 to see why each issue was flagged.
    """
    try:
        return await review_code(CodeInput(code=request.code, language=request.language))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post('/review-pr', response_model=PRReviewResponse)
async def review_pr_endpoint(request: PRReviewRequest) -> PRReviewResponse:
    """
    Review a git diff (for PR reviews).
    
    Send a unified diff and get back issues found in the CHANGED lines only.
    Perfect for GitHub Actions integration.
    
    Example:
        curl -X POST /review-pr -d '{"diff": "...", "filename": "src/main.py"}'
    """
    try:
        return await review_diff(DiffInput(
            diff=request.diff,
            filename=request.filename,
            language=request.language,
        ))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(app, host='0.0.0.0', port=8080)
