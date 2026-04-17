Describe briefly what this PR changes and why.



Delete the guidance below before opening the PR.

- Use `pre-commit` locally before pushing to catch formatting issues such as `clang-format`, as the same checks are enforced in CI.
- Add SPDX headers or matching REUSE metadata for new files. License compliance is checked in CI.

### PR approval workflow

- The automatic prechecks must pass.
- A repository member must trigger the physics checks.
- For approval, either the physics results must remain unchanged, or the full validation must pass.
- If the physics drift shows differences, explain them clearly in the PR description.

---
