"""BugBot: AI Code Reviewer.

Run locally:
    genkit start -- python src/main.py

Then try:
    curl localhost:8080/review -d '{"code": "password = \"admin123\""}'
    curl localhost:8080/review-pr -d '{"diff": "+API_KEY = \"secret\"", "filename": "config.py"}'

Open localhost:4000 to see traces.
"""

import asyncio
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import FastAPI
from pydantic import BaseModel, Field

from genkit import Genkit, Output
from genkit.plugins.google_genai import GoogleAI  # pyright: ignore[reportMissingImports]

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
# Types
# =============================================================================

class Issue(BaseModel):
    line: int
    title: str
    severity: Literal['critical', 'warning', 'info']
    category: Literal['security', 'bug', 'style']
    explanation: str
    suggestion: str


class Analysis(BaseModel):
    issues: list[Issue] = Field(default_factory=list)


class ReviewResult(BaseModel):
    filename: str = ''
    summary: str
    issues: list[Issue]
    verdict: Literal['approve', 'request_changes', 'needs_discussion']


# Register for dotprompt files
ai.define_schema('Analysis', Analysis)


# =============================================================================
# Flows
# =============================================================================

def summarize(issues: list[Issue]) -> ReviewResult:
    """Create a summary from issues."""
    has_critical = any(i.severity == 'critical' for i in issues)
    has_warning = any(i.severity == 'warning' for i in issues)
    
    verdict: Literal['approve', 'request_changes', 'needs_discussion']
    if has_critical:
        verdict = 'request_changes'
    elif has_warning:
        verdict = 'needs_discussion'
    else:
        verdict = 'approve'
    
    if not issues:
        summary = 'LGTM! No issues found.'
    else:
        counts = {'security': 0, 'bug': 0, 'style': 0}
        for i in issues:
            counts[i.category] += 1
        summary = f"Found {len(issues)} issue(s): {counts['security']} security, {counts['bug']} bugs, {counts['style']} style"
    
    return ReviewResult(summary=summary, issues=issues, verdict=verdict)


@ai.flow()
async def review_code(code: str, language: str = 'python') -> ReviewResult:
    """Review code for security, bugs, and style issues."""
    # Run all analyzers in parallel
    prompts = await asyncio.gather(
        ai.prompt('analyze_security', output=Output(schema=Analysis))(input={'code': code, 'language': language}),
        ai.prompt('analyze_bugs', output=Output(schema=Analysis))(input={'code': code, 'language': language}),
        ai.prompt('analyze_style', output=Output(schema=Analysis))(input={'code': code, 'language': language}),
    )
    
    # Combine and dedupe
    all_issues = []
    seen = set()
    for response in prompts:
        for issue in response.output.issues:
            key = (issue.line, issue.title[:20])
            if key not in seen:
                seen.add(key)
                all_issues.append(issue)
    
    all_issues.sort(key=lambda i: i.line)
    return summarize(all_issues)


@ai.flow()
async def review_diff(diff: str, filename: str, language: str = 'python') -> ReviewResult:
    """Review a git diff - only analyzes changed lines."""
    prompt = ai.prompt('analyze_diff', output=Output(schema=Analysis))
    response = await prompt(input={'diff': diff, 'filename': filename, 'language': language})
    
    result = summarize(response.output.issues)
    result.filename = filename
    return result


# =============================================================================
# API
# =============================================================================

app = FastAPI(title='BugBot', description='AI code reviewer')


class CodeRequest(BaseModel):
    code: str
    language: str = 'python'


class DiffRequest(BaseModel):
    diff: str
    filename: str = 'unknown'
    language: str = 'python'


@app.get('/health')
async def health():
    return {'status': 'ok'}


@app.post('/review')
async def review(req: CodeRequest) -> ReviewResult:
    """Review code for issues."""
    return await review_code(req.code, req.language)


@app.post('/review-pr')
async def review_pr(req: DiffRequest) -> ReviewResult:
    """Review a git diff (for GitHub Actions)."""
    return await review_diff(req.diff, req.filename, req.language)


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8080)
