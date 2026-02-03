"""BugBot Firebase Function.

This is deployed at: https://us-central1-aim-testing.cloudfunctions.net/review_pr

Note: This uses the released genkit package. For the typed output features,
see src/main.py which requires the development version.
"""

from __future__ import annotations

import asyncio
import json
import re
from typing import Any

from firebase_admin import initialize_app
from firebase_functions import https_fn, options
from firebase_functions.params import SecretParam

initialize_app()
GEMINI_API_KEY = SecretParam('GEMINI_API_KEY')


def json_response(data: dict[str, Any], status: int = 200) -> https_fn.Response:
    return https_fn.Response(
        response=json.dumps(data),
        status=status,
        headers={
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
        },
    )


@https_fn.on_request(
    memory=options.MemoryOption.GB_1,
    timeout_sec=120,
    secrets=[GEMINI_API_KEY],
)
def review_pr(request: https_fn.Request) -> https_fn.Response:
    """Review a git diff for security, bugs, and style issues."""
    if request.method == 'OPTIONS':
        return https_fn.Response('', status=204, headers={
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'POST',
            'Access-Control-Allow-Headers': 'Content-Type',
        })

    if request.method != 'POST':
        return json_response({'error': 'POST required'}, 405)

    body: dict[str, Any] = request.get_json(silent=True) or {}
    diff = str(body.get('diff', ''))
    filename = str(body.get('filename', 'unknown'))
    language = str(body.get('language', 'python'))

    if not diff:
        return json_response({'error': 'diff required'}, 400)

    try:
        from genkit.ai import Genkit
        from genkit.plugins.google_genai import GoogleAI

        ai = Genkit(
            plugins=[GoogleAI(api_key=GEMINI_API_KEY.value)],
            model='googleai/gemini-2.0-flash',
        )

        async def analyze() -> dict[str, Any]:
            response = await ai.generate(prompt=f'''Analyze this git diff for security vulnerabilities, bugs, and style issues.

**File:** {filename}
**Language:** {language}

```diff
{diff}
```

RULES:
1. Only analyze ADDED lines (starting with +)
2. Return JSON: {{"issues": [{{line, title, severity, category, explanation, suggestion}}]}}
3. severity: "critical" | "warning" | "info"
4. category: "security" | "bug" | "style"
5. If safe, return {{"issues": []}}''')

            match = re.search(r'\{[\s\S]*\}', response.text)
            if match:
                return json.loads(match.group())
            return {'issues': []}

        result = asyncio.run(analyze())
        issues = result.get('issues', [])

        # Determine verdict
        has_critical = any(i.get('severity') == 'critical' for i in issues)
        has_warning = any(i.get('severity') == 'warning' for i in issues)

        if has_critical:
            verdict = 'request_changes'
        elif has_warning:
            verdict = 'needs_discussion'
        else:
            verdict = 'approve'

        # Summary
        if not issues:
            summary = 'LGTM!'
        else:
            c = {'security': 0, 'bug': 0, 'style': 0}
            for i in issues:
                cat = i.get('category', 'style')
                if cat in c:
                    c[cat] += 1
            summary = f"Found {len(issues)}: {c['security']} security, {c['bug']} bugs, {c['style']} style"

        return json_response({
            'filename': filename,
            'summary': summary,
            'issues': issues,
            'verdict': verdict,
        })

    except Exception as e:
        return json_response({'error': str(e)}, 500)
