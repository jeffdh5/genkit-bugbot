# Copyright 2025 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# SPDX-License-Identifier: Apache-2.0

"""Firebase Functions for Genkit AI endpoints.

This module provides Firebase Functions endpoints for AI-powered features.
Each endpoint is a separate function optimized for serverless deployment.

Deploy with:
    cd firebase
    firebase deploy --only functions

IMPORTANT NUANCES:
1. Each function is independent - no shared state between invocations
2. Genkit is initialized per-request (adds to cold start)
3. Use lightweight functions for production (not full FastAPI wrapper)
4. Streaming is NOT supported in Firebase Functions (use Cloud Run instead)
5. Set appropriate memory and timeout for AI workloads
"""

from __future__ import annotations

# Firebase packages have incomplete type stubs - suppress related warnings
# pyright: reportMissingTypeStubs=false
# pyright: reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false
# pyright: reportUnknownParameterType=false
# pyright: reportPrivateImportUsage=false
# pyright: reportMissingImports=false

import asyncio
import json
from typing import cast

# Firebase imports
from firebase_admin import initialize_app
from firebase_functions import https_fn, options
from firebase_functions.params import SecretParam
from genkit.ai import Genkit


# Initialize Firebase Admin SDK (only once per cold start)
_ = initialize_app()

# Define secret for API key - stored securely in Firebase
GEMINI_API_KEY: SecretParam = SecretParam('GEMINI_API_KEY')


def _init_genkit(api_key: str) -> Genkit:
    """Initialize Genkit with Google AI plugin.

    Args:
        api_key: The Gemini API key for authentication.

    Returns:
        Configured Genkit instance.
    """
    from genkit.ai import Genkit
    from genkit.plugins.google_genai import GoogleAI

    return Genkit(
        plugins=[GoogleAI(api_key=api_key)],
        model='googleai/gemini-2.0-flash',
    )


def _json_response(data: dict[str, object], status: int = 200) -> https_fn.Response:
    """Create a JSON response with proper headers."""
    return https_fn.Response(
        json.dumps(data),
        status=status,
        headers={
            'Content-Type': 'application/json',
            'Access-Control-Allow-Origin': '*',
            'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
            'Access-Control-Allow-Headers': 'Content-Type',
        },
    )


def _handle_cors_preflight(req: https_fn.Request) -> https_fn.Response | None:
    """Handle CORS preflight requests."""
    if req.method == 'OPTIONS':
        return https_fn.Response(
            '',
            status=204,
            headers={
                'Access-Control-Allow-Origin': '*',
                'Access-Control-Allow-Methods': 'GET, POST, OPTIONS',
                'Access-Control-Allow-Headers': 'Content-Type',
                'Access-Control-Max-Age': '3600',
            },
        )
    return None

# =============================================================================
# Chat Endpoint - Main AI chat function
# =============================================================================


@https_fn.on_request(
    memory=options.MemoryOption.MB_512,  # AI needs more memory
    timeout_sec=120,  # AI generation can take time
    secrets=[GEMINI_API_KEY],
)
def chat(req: https_fn.Request) -> https_fn.Response:
    """Chat with the AI model.

    GET:  /chat?message=Hello
    POST: /chat with JSON body {"message": "Hello"}

    NOTE: Streaming is NOT supported in Firebase Functions.
    Use Cloud Run for streaming responses.
    """
    if cors := _handle_cors_preflight(req):
        return cors

    try:
        # Get message from request
        if req.method == 'GET':
            message: str = str(req.args.get('message', 'Hello'))
        else:
            data: dict[str, object] = req.get_json(silent=True) or {}
            message = str(data.get('message', 'Hello'))

        if not message:
            return _json_response({'error': 'message is required'}, status=400)

        # Initialize Genkit and run
        ai = _init_genkit(GEMINI_API_KEY.value)

        async def chat_flow(msg: str) -> str:
            response = await ai.generate(prompt=msg)
            return response.text

        # Run async flow synchronously
        result: str = asyncio.run(chat_flow(message))

        return _json_response({
            'response': result,
            'model': 'googleai/gemini-2.0-flash',
        })

    except Exception as e:
        return _json_response({'error': str(e)}, status=500)


# =============================================================================
# Joke Endpoint - Structured output example
# =============================================================================


@https_fn.on_request(
    memory=options.MemoryOption.MB_512,
    timeout_sec=60,
    secrets=[GEMINI_API_KEY],
)
def joke(req: https_fn.Request) -> https_fn.Response:
    """Generate a structured joke.

    GET:  /joke?topic=programming
    POST: /joke with JSON body {"topic": "cats"}
    """
    if cors := _handle_cors_preflight(req):
        return cors

    try:
        # Get topic from request
        if req.method == 'GET':
            topic: str = str(req.args.get('topic', 'programming'))
        else:
            data: dict[str, object] = req.get_json(silent=True) or {}
            topic = str(data.get('topic', 'programming'))

        ai = _init_genkit(GEMINI_API_KEY.value)

        async def joke_flow(t: str) -> str:
            response = await ai.generate(
                prompt=f'Tell me a funny joke about {t}. Format it as: Setup: [setup] Punchline: [punchline]',
            )
            return response.text

        result: str = asyncio.run(joke_flow(topic))

        return _json_response({'joke': result, 'topic': topic})

    except Exception as e:
        return _json_response({'error': str(e)}, status=500)


# =============================================================================
# Summarize Endpoint - Text processing example
# =============================================================================


# =============================================================================
# Support Agent - RAG + Tools example (simplified for serverless)
# =============================================================================

# Simulated knowledge base (in production, use Firestore or a vector DB)
KNOWLEDGE_BASE = [
    {'content': 'To reset your password: Go to Settings > Security > Reset Password.', 'source': 'password-guide.md'},
    {'content': 'Billing issues: Contact support@example.com. Refunds take 5-7 days.', 'source': 'billing-faq.md'},
    {'content': 'API rate limits: Free=100/min, Pro=1000/min, Enterprise=unlimited.', 'source': 'api-limits.md'},
]


def _simple_retrieve(query: str, limit: int = 3) -> list[dict[str, str]]:
    """Simple keyword-based retrieval (replace with vector search in prod)."""
    query_lower = query.lower()
    scored = []
    for doc in KNOWLEDGE_BASE:
        score = sum(1 for word in query_lower.split() if word in doc['content'].lower())
        if score > 0:
            scored.append((score, doc))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in scored[:limit]]


@https_fn.on_request(
    memory=options.MemoryOption.GB_1,  # More memory for RAG
    timeout_sec=120,
    secrets=[GEMINI_API_KEY],
)
def support(req: https_fn.Request) -> https_fn.Response:
    """Support agent with RAG - simplified for Firebase Functions.

    POST: /support with JSON body {"question": "How do I reset my password?"}

    This is a simplified version. For full RAG + tools, use Cloud Run.
    """
    if cors := _handle_cors_preflight(req):
        return cors

    if req.method != 'POST':
        return _json_response({'error': 'POST method required'}, status=405)

    try:
        data: dict[str, object] = req.get_json(silent=True) or {}
        question: str = str(data.get('question', ''))

        if not question:
            return _json_response({'error': 'question is required'}, status=400)

        # Step 1: Retrieve relevant docs (RAG)
        docs = _simple_retrieve(question, limit=3)
        context = '\n'.join([f"[{d['source']}]: {d['content']}" for d in docs])

        # Step 2: Generate with context
        ai = _init_genkit(GEMINI_API_KEY.value)

        async def support_flow(q: str, ctx: str) -> str:
            response = await ai.generate(
                prompt=f"""You are a helpful support agent. Answer using the documentation.

DOCUMENTATION:
{ctx}

QUESTION: {q}

Be concise and helpful.""",
            )
            return response.text

        result: str = asyncio.run(support_flow(question, context))

        return _json_response({
            'answer': result,
            'sources': [d['source'] for d in docs],
        })

    except Exception as e:
        return _json_response({'error': str(e)}, status=500)


@https_fn.on_request(
    memory=options.MemoryOption.MB_512,
    timeout_sec=60,
    secrets=[GEMINI_API_KEY],
)
def summarize(req: https_fn.Request) -> https_fn.Response:
    """Summarize text.

    POST: /summarize with JSON body {"text": "...", "max_sentences": 2}
    """
    if cors := _handle_cors_preflight(req):
        return cors

    if req.method != 'POST':
        return _json_response({'error': 'POST method required'}, status=405)

    try:
        data: dict[str, object] = req.get_json(silent=True) or {}
        text: str = str(data.get('text', ''))
        max_sentences: int = cast(int, data.get('max_sentences', 3))

        if not text:
            return _json_response({'error': 'text is required'}, status=400)

        ai = _init_genkit(GEMINI_API_KEY.value)

        async def summary_flow(t: str, max_s: int) -> str:
            response = await ai.generate(
                prompt=f'Summarize the following text in {max_s} sentences or less:\n\n{t}',
            )
            return response.text

        result: str = asyncio.run(summary_flow(text, max_sentences))

        return _json_response({
            'summary': result,
            'original_length': len(text),
            'summary_length': len(result),
        })

    except Exception as e:
        return _json_response({'error': str(e)}, status=500)


# =============================================================================
# Review PR - Diff-based code review for GitHub Actions
# =============================================================================


@https_fn.on_request(
    memory=options.MemoryOption.GB_1,  # More memory for code analysis
    timeout_sec=120,
    secrets=[GEMINI_API_KEY],
)
def review_pr(req: https_fn.Request) -> https_fn.Response:
    """Review a git diff for security, bugs, and style issues.

    POST: /review-pr with JSON body {"diff": "...", "filename": "...", "language": "python"}
    
    Perfect for GitHub Actions integration - reviews only CHANGED lines.
    """
    if cors := _handle_cors_preflight(req):
        return cors

    if req.method != 'POST':
        return _json_response({'error': 'POST method required'}, status=405)

    try:
        data: dict[str, object] = req.get_json(silent=True) or {}
        diff: str = str(data.get('diff', ''))
        filename: str = str(data.get('filename', 'unknown'))
        language: str = str(data.get('language', 'python'))

        if not diff:
            return _json_response({'error': 'diff is required'}, status=400)

        ai = _init_genkit(GEMINI_API_KEY.value)

        async def review_diff_flow(d: str, f: str, lang: str) -> dict[str, object]:
            prompt = f"""Analyze this git diff for security vulnerabilities, bugs, and style issues.

**File:** {f}
**Language:** {lang}

```diff
{d}
```

IMPORTANT RULES:
1. ONLY analyze lines that are ADDED (lines starting with `+`)
2. Ignore removed lines (starting with `-`) - they're being deleted
3. Line numbers should reference the NEW file (after the change)
4. Focus on issues INTRODUCED by this change

For each issue, provide:
- line: The line number in the NEW file
- title: Brief description
- severity: "critical" for security, "warning" for bugs, "info" for style
- category: "security", "bug", or "style"
- explanation: Why this is a problem
- suggestion: How to fix it

Respond with JSON: {{"issues": [...]}}
If the diff is safe, return {{"issues": []}}"""

            response = await ai.generate(prompt=prompt)
            
            # Parse the response
            import json
            import re
            text = response.text
            # Extract JSON from response
            json_match = re.search(r'\{[\s\S]*\}', text)
            if json_match:
                result = json.loads(json_match.group())
                return cast(dict[str, object], result)
            return {'issues': []}

        result = asyncio.run(review_diff_flow(diff, filename, language))
        issues = cast(list[dict[str, object]], result.get('issues', []))
        
        # Determine verdict
        has_critical = any(i.get('severity') == 'critical' for i in issues)
        has_warnings = any(i.get('severity') == 'warning' for i in issues)
        
        if has_critical:
            verdict = 'request_changes'
        elif has_warnings:
            verdict = 'needs_discussion'
        else:
            verdict = 'approve'
        
        # Generate summary
        if not issues:
            summary = 'No issues found in changed code. LGTM!'
        else:
            by_cat: dict[str, int] = {'security': 0, 'bug': 0, 'style': 0}
            for i in issues:
                cat = str(i.get('category', 'style'))
                if cat in by_cat:
                    by_cat[cat] += 1
            summary = f"Found {len(issues)} issue(s): {by_cat['security']} security, {by_cat['bug']} bugs, {by_cat['style']} style"

        return _json_response({
            'filename': filename,
            'summary': summary,
            'issues': issues,
            'verdict': verdict,
        })

    except Exception as e:
        return _json_response({'error': str(e)}, status=500)
