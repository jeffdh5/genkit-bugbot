"""BugBot Firebase Functions.

Deploy:
    firebase functions:secrets:set GEMINI_API_KEY
    firebase deploy --only functions

Test:
    curl https://YOUR-PROJECT.cloudfunctions.net/review_pr \
      -d '{"diff": "+API_KEY = \"secret\"", "filename": "config.py"}'
"""

import asyncio
import json
import re
from pathlib import Path
from typing import Literal

from firebase_admin import initialize_app
from firebase_functions import https_fn, options
from firebase_functions.params import SecretParam
from pydantic import BaseModel, Field

initialize_app()
GEMINI_API_KEY = SecretParam('GEMINI_API_KEY')


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


# =============================================================================
# Helpers
# =============================================================================

def json_response(data: dict, status: int = 200) -> https_fn.Response:
    return https_fn.Response(
        json.dumps(data),
        status=status,
        headers={'Content-Type': 'application/json', 'Access-Control-Allow-Origin': '*'},
    )


def handle_cors(req: https_fn.Request) -> https_fn.Response | None:
    if req.method == 'OPTIONS':
        return https_fn.Response('', status=204, headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        })
    return None


def init_genkit(api_key: str):
    from genkit import Genkit, Output
    from genkit.plugins.google_genai import GoogleAI
    
    return Genkit(
        plugins=[GoogleAI(api_key=api_key)],
        model='googleai/gemini-2.0-flash',
        prompt_dir=Path(__file__).parent / 'prompts',
    )


# =============================================================================
# Main Endpoint
# =============================================================================

@https_fn.on_request(memory=options.MemoryOption.GB_1, timeout_sec=120, secrets=[GEMINI_API_KEY])
def review_pr(req: https_fn.Request) -> https_fn.Response:
    """Review a git diff for security, bugs, and style issues."""
    if cors := handle_cors(req):
        return cors
    
    if req.method != 'POST':
        return json_response({'error': 'POST required'}, 405)
    
    try:
        data = req.get_json(silent=True) or {}
        diff = str(data.get('diff', ''))
        filename = str(data.get('filename', 'unknown'))
        language = str(data.get('language', 'python'))
        
        if not diff:
            return json_response({'error': 'diff required'}, 400)
        
        ai = init_genkit(GEMINI_API_KEY.value)
        
        async def analyze():
            from genkit import Output
            prompt = ai.prompt('analyze_diff', output=Output(schema=Analysis))
            response = await prompt(input={'diff': diff, 'filename': filename, 'language': language})
            return response.output
        
        result = asyncio.run(analyze())
        issues = [i.model_dump() for i in result.issues]
        
        # Determine verdict
        has_critical = any(i['severity'] == 'critical' for i in issues)
        has_warning = any(i['severity'] == 'warning' for i in issues)
        
        if has_critical:
            verdict = 'request_changes'
        elif has_warning:
            verdict = 'needs_discussion'
        else:
            verdict = 'approve'
        
        # Summary
        if not issues:
            summary = 'LGTM! No issues found.'
        else:
            counts = {'security': 0, 'bug': 0, 'style': 0}
            for i in issues:
                counts[i['category']] += 1
            summary = f"Found {len(issues)} issue(s): {counts['security']} security, {counts['bug']} bugs, {counts['style']} style"
        
        return json_response({
            'filename': filename,
            'summary': summary,
            'issues': issues,
            'verdict': verdict,
        })
    
    except Exception as e:
        return json_response({'error': str(e)}, 500)
