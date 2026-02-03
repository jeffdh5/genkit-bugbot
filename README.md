# BugBot 🤖

AI code reviewer that comments on PRs with security issues, bugs, and style problems.

Built with [Google Genkit](https://github.com/firebase/genkit) - showcasing **typed outputs**.

## The Key Feature: Typed Outputs

```python
# Define your schema
class Analysis(BaseModel):
    issues: list[Issue]

# Get a typed prompt
prompt = ai.prompt('analyze_diff', output=Output(schema=Analysis))

# Execute it
response = await prompt(input={'code': code})

# response.output is a real Analysis instance, not a dict!
for issue in response.output.issues:  # <- Full IDE autocomplete!
    print(issue.title)  # <- Type checked!
```

**Before:** `response.output` was `Any` - no autocomplete, no type checking  
**After:** `response.output` is your actual Pydantic model with full typing!

## Quick Start

```bash
# Install genkit (dev version with typed outputs)
pip install -e /path/to/genkit/py/packages/genkit

# Run
genkit start -- python src/main.py
```

Try it:
```bash
curl localhost:8080/review -d '{"code": "API_KEY = \"secret123\""}'
```

## How It Works

1. Define Pydantic models for your output schema
2. Use `ai.prompt('name', output=Output(schema=YourModel))`
3. `response.output` is now fully typed - both statically AND at runtime!

## GitHub Actions Integration

BugBot can automatically review PRs. See `.github/workflows/bugbot.yml`.

The deployed function is at: `https://us-central1-aim-testing.cloudfunctions.net/review_pr`

## API

**POST /review-pr**
```json
{"diff": "+API_KEY = \"secret\"", "filename": "config.py", "language": "python"}
```

**Response:**
```json
{
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
