# BugBot 🤖

AI code reviewer that comments on PRs with security issues, bugs, and style problems.

Built with [Google Genkit](https://github.com/firebase/genkit).

## How it works

1. GitHub Action triggers on PR
2. Sends diff to BugBot API
3. AI analyzes only changed lines
4. Posts comment with issues found

## Quick Start

### Deploy

```bash
cd firebase
firebase login
firebase functions:secrets:set GEMINI_API_KEY
firebase deploy --only functions
```

### Add to any repo

1. Copy `.github/workflows/bugbot.yml` to your repo
2. Add repo variable: `BUGBOT_URL` = your deployed function URL
3. Open a PR and watch BugBot review it!

## Local Development

```bash
pip install -r requirements.txt
genkit start -- python src/main.py
```

Try it:
```bash
# Review code
curl localhost:8080/review -d '{"code": "password = \"admin123\""}'

# Review a diff
curl localhost:8080/review-pr -d '{"diff": "+API_KEY = \"secret\"", "filename": "config.py"}'
```

Open http://localhost:4000 to see traces.

## API

**POST /review-pr**

```json
{
  "diff": "@@ -1,3 +1,5 @@\n+API_KEY = \"secret123\"",
  "filename": "config.py",
  "language": "python"
}
```

Response:
```json
{
  "filename": "config.py",
  "summary": "Found 1 issue(s): 1 security, 0 bugs, 0 style",
  "issues": [{
    "line": 1,
    "title": "Hardcoded Secret",
    "severity": "critical",
    "category": "security",
    "explanation": "API key exposed in source code",
    "suggestion": "Use environment variables"
  }],
  "verdict": "request_changes"
}
```

## License

Apache 2.0
