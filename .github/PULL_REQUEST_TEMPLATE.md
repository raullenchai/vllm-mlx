## What does this PR do?

<!-- Brief description of the change -->

## Checklist

- [ ] Tests pass locally (`python3 -m pytest tests/ -x`)
- [ ] Lint passes (`ruff check && ruff format --check`)
- [ ] Self-validated with `python3 -m scripts.pr_validate.pr_validate <PR#>` — see [CONTRIBUTING.md](../CONTRIBUTING.md#self-validating-your-pr) (opt out heavy steps with `PR_VALIDATE_NO_DEEPSEEK=1 PR_VALIDATE_NO_STRESS=1` if you don't have the hardware/keys)
- [ ] Updated README/docs if applicable
- [ ] No breaking changes to existing API
