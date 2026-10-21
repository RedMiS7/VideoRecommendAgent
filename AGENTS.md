# Repository Guidelines

修改代码之前询问用户

Coding After Asking the User

修改代码之前询问用户

修改代码之前询问用户

## Project Structure & Module Organization

Application code is in `src/video_agent/`: model contracts and adapters live in `models/`, while memory, RAG, recommendations, API, and CLI have separate modules. Tests are in `tests/`, evaluation fixtures in `evals/`, Windows task scripts in `scripts/`, and feature documentation in `docs/`. Local databases and reports are ignored by Git.

## Build, Test, and Development Commands

Run commands from the repository root with Python 3.10+ and uv:

- `uv sync --locked`: install project-local dependencies.
- `uv run video-agent serve`: start the local FastAPI service.
- `uv run video-agent demo --days 8`: run a synthetic offline demonstration.
- `uv run video-agent eval --offline`: verify the evaluation pipeline with test doubles.
- `uv run pytest -q`: run tests.
- `uv run ruff check src tests`: lint Python files.

## Coding Style & Naming Conventions

Use four-space indentation, snake_case functions and modules, and PascalCase classes. Ruff checks syntax and unused imports. Business code uses provider-neutral model contracts; provider selection belongs in application assembly. Keep changes focused and remove code made unused by your changes.

## Testing Guidelines

Use pytest with `test_*.py` files and behavior-focused `test_*` functions. Cover model contracts, persistence, feedback consolidation, citation ownership, and daily deduplication. Prefer a failing regression test before fixes. No coverage percentage is mandated. Real-service checks and AI quality evaluations remain separate from offline tests; never present test-double scores as model quality.

## Commit & Pull Request Guidelines

This workspace has no Git history, so no commit convention can be inferred. Use concise, imperative commit subjects, such as `Add video input validation`. Pull requests should explain the change, its purpose, verification performed, and any limitations. Link relevant issues and include screenshots for visible UI changes.

## Feature Documentation

After completing each feature, write a document in `docs/` covering:

- **What was added:** describe the new functionality and its scope.
- **Why it exists:** explain the problem it solves and its purpose.
- **How to test or use it:** provide actionable steps, commands, and expected results.

## Contributor & Agent Practices

State material assumptions before implementation. Resolve correctness-critical ambiguity before editing. Prefer small, reversible changes. Obtain explicit authorization before deleting files, rewriting Git history, changing secrets or production configuration, or upgrading dependencies broadly.
