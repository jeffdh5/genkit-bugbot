# BugBot 🤖

AI-powered code reviewer that automatically comments on PRs with security vulnerabilities, bugs, and style issues.

**Powered by [Google Genkit](https://github.com/firebase/genkit)**

## Features

- **Security scanning**: SQL injection, XSS, hardcoded secrets, command injection
- **Bug detection**: Null errors, logic issues, resource leaks
- **Style suggestions**: Missing error handling, naming conventions
- **Diff-aware**: Only reviews changed lines, not the entire file
- **GitHub Actions integration**: Auto-comments on PRs

## Quick Start

### 1. Deploy to Firebase

```bash
cd firebase
firebase login
firebase functions:secrets:set GEMINI_API_KEY
firebase deploy --only functions
```

### 2. Add to Your Repo

Create `.github/workflows/bugbot.yml`:

```yaml
name: BugBot PR Review

on:
  pull_request:
    types: [opened, synchronize]

jobs:
  review:
    runs-on: ubuntu-latest
    permissions:
      pull-requests: write
      contents: read
    steps:
      - uses: actions/checkout@v4
        with:
          fetch-depth: 0
      
      - name: Review PR
        env:
          BUGBOT_URL: ${{ vars.BUGBOT_URL }}
          GH_TOKEN: ${{ github.token }}
        run: |
          for file in $(git diff --name-only origin/${{ github.base_ref }}...HEAD -- '*.py' '*.js' '*.ts' | head -20); do
            DIFF=$(git diff origin/${{ github.base_ref }}...HEAD -- "$file")
            RESPONSE=$(curl -s -X POST "$BUGBOT_URL" \
              -H "Content-Type: application/json" \
              -d "$(jq -n --arg diff "$DIFF" --arg filename "$file" '{diff: $diff, filename: $filename}')")
            
            VERDICT=$(echo "$RESPONSE" | jq -r '.verdict')
            if [ "$VERDICT" != "approve" ]; then
              gh pr comment ${{ github.event.pull_request.number }} \
                --body "$(echo "$RESPONSE" | jq -r '"## 🤖 BugBot: `\(.filename)`\n\(.summary)\n\n" + (.issues | map("- **[\(.severity)]** \(.title): \(.suggestion)") | join("\n"))')"
            fi
          done
```

### 3. Set Repository Variable

Go to **Settings → Secrets and variables → Actions → Variables** and add:
- `BUGBOT_URL`: Your deployed Firebase function URL

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run with Genkit dev UI
genkit start -- python src/main.py

# Open http://localhost:4000 to see traces
```

## API

### POST /review-pr

Review a git diff for issues.

```bash
curl -X POST https://your-function-url/review_pr \
  -H "Content-Type: application/json" \
  -d '{
    "diff": "@@ -1,3 +1,5 @@\n+API_KEY = \"secret123\"",
    "filename": "config.py",
    "language": "python"
  }'
```

Response:
```json
{
  "filename": "config.py",
  "summary": "Found 1 issue(s): 1 security, 0 bugs, 0 style",
  "issues": [{
    "line": 1,
    "title": "Hardcoded API Key",
    "severity": "critical",
    "category": "security",
    "explanation": "API key is exposed in source code",
    "suggestion": "Use environment variables"
  }],
  "verdict": "request_changes"
}
```

## License

Apache 2.0
