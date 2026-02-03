"""BugBot Firebase Functions.

Deploy:
    firebase functions:secrets:set GEMINI_API_KEY
    firebase deploy --only functions
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any, Literal

from firebase_admin import initialize_app
from firebase_functions import https_fn, options
from firebase_functions.params import SecretParam
from pydantic import BaseModel, Field

from genkit import Genkit, Output
from genkit.plugins.google_genai import GoogleAI

# Initialize Firebase
initialize_app()

# Secret stored in Firebase
GEMINI_API_KEY = SecretParam('GEMINI_API_KEY')

# Prompts directory (relative to this file)
PROMPTS_DIR = Path(__file__).parent / 'prompts'


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

    filename: str
    summary: str
    issues: list[dict[str, Any]]
    verdict: Literal['approve', 'request_changes', 'needs_discussion']


# =============================================================================
# Helpers
# =============================================================================


def create_genkit(api_key: str) -> Genkit:
    """Create a configured Genkit instance."""
    return Genkit(
        plugins=[GoogleAI(api_key=api_key)],
        model='googleai/gemini-2.0-flash',
        prompt_dir=PROMPTS_DIR,
    )


def json_response(data: dict[str, Any], status: int = 200) -> https_fn.Response:
    """Create a JSON response with CORS headers."""
    return https_fn.Response(
        response=json.dumps(data),
        status=status,
        headers={
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        },
    )


def handle_cors(request: https_fn.Request) -> https_fn.Response | None:
    """Handle CORS preflight requests."""
    if request.method == 'OPTIONS':
        return https_fn.Response(
            response='',
            status=204,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Max-Age': '3600',
            },
        )
    return None


def create_summary(issues: list[dict[str, Any]]) -> tuple[str, str]:
    """Create summary and verdict from issues."""
    if not issues:
        return 'LGTM! No issues found.', 'approve'

    counts = {'security': 0, 'bug': 0, 'style': 0}
    has_critical = False
    has_warning = False

    for issue in issues:
        category = issue.get('category', 'style')
        severity = issue.get('severity', 'info')
        if category in counts:
            counts[category] += 1
        if severity == 'critical':
            has_critical = True
        elif severity == 'warning':
            has_warning = True

    summary = f"Found {len(issues)} issue(s): {counts['security']} security, {counts['bug']} bugs, {counts['style']} style"

    if has_critical:
        verdict = 'request_changes'
    elif has_warning:
        verdict = 'needs_discussion'
    else:
        verdict = 'approve'

    return summary, verdict


# =============================================================================
# Cloud Function
# =============================================================================


@https_fn.on_request(
    memory=options.MemoryOption.GB_1,
    timeout_sec=120,
    secrets=[GEMINI_API_KEY],
)
def review_pr(request: https_fn.Request) -> https_fn.Response:
    """Review a git diff for security, bugs, and style issues.

    POST /review_pr
    Body: {"diff": "...", "filename": "...", "language": "python"}

    Returns: {"filename", "summary", "issues", "verdict"}
    """
    # Handle CORS preflight
    if cors_response := handle_cors(request):
        return cors_response

    # Require POST
    if request.method != 'POST':
        return json_response({'error': 'POST method required'}, status=405)

    # Parse request
    try:
        body: dict[str, Any] = request.get_json(silent=True) or {}
    except Exception:
        return json_response({'error': 'Invalid JSON'}, status=400)

    diff = str(body.get('diff', ''))
    filename = str(body.get('filename', 'unknown'))
    language = str(body.get('language', 'python'))

    if not diff:
        return json_response({'error': 'diff is required'}, status=400)

    # Analyze the diff
    try:
        ai = create_genkit(GEMINI_API_KEY.value)

        # Register schema for dotprompt
        ai.define_schema('Analysis', Analysis)

        async def analyze_diff() -> Analysis:
            prompt = ai.prompt('analyze_diff', output=Output(schema=Analysis))
            response = await prompt(input={
                'diff': diff,
                'filename': filename,
                'language': language,
            })
            return response.output

        result = asyncio.run(analyze_diff())
        issues = [issue.model_dump() for issue in result.issues]
        summary, verdict = create_summary(issues)

        return json_response({
            'filename': filename,
            'summary': summary,
            'issues': issues,
            'verdict': verdict,
        })

    except Exception as e:
        return json_response({'error': str(e)}, status=500)
