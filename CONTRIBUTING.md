# Contributing to scoringrules

Thanks for contributing! `scoringrules` implements every score once and runs it across
multiple array backends (numpy, numba, jax, torch), so most of the conventions below exist
to keep that multi-backend guarantee intact.

## Development setup

The project uses [uv](https://docs.astral.sh/uv/) for environments and packaging.

```bash
uv sync --all-extras --dev          # all backends (jax/torch are extras) + dev tools
uv run pytest tests/                # full suite, every installed backend
uv run pytest tests/ --backend numpy        # one backend
uv run pytest tests/ --backend numpy,numba  # a subset
uvx pre-commit run --all-files      # ruff, codespell, whitespace — the hooks CI runs
```

`--backend` (see `tests/conftest.py`) filters which backends a run exercises. Without it,
tests parametrize over whichever of numpy/numba/jax/torch are importable — so to genuinely
cover all four locally you must `uv sync --all-extras` first.

## CI overview

`.github/workflows/ci.yaml` runs on every push and PR to `main` (and on demand via
**Run workflow**):

- **`test`** — a matrix of `{macos, ubuntu} × py{3.12,3.13,3.14} × {numpy,numba,jax,torch}`.
  Each leg installs only its backend and runs `pytest --backend <b>`. `fail-fast: false`
  keeps one backend's failure from cancelling the others. A folded **`all`** leg
  (ubuntu / 3.13 / all extras) runs the full suite with coverage and uploads to Codecov.
- **`lint`** — `pre-commit` (ruff, codespell, whitespace).

A `concurrency` group cancels superseded runs on rapid pushes, and workflows run with
least-privilege `permissions: contents: read`.

**Repo automation:**

- **Dependabot** (`.github/dependabot.yml`) opens **one grouped PR per week** for GitHub
  Actions version bumps — not one PR per action. It doesn't auto-merge.
- **Labeler** (`.github/labeler.yml` + `labeler.yml` workflow) auto-labels PRs by the paths
  they touch (`core`, `docs`, `ci`, `backend:jax`, `backend:torch`). These labels drive the
  release notes below.
- **Issue / PR templates** live in `.github/` — the bug form asks for backend, Python and
  package versions, and a reproducer.

**Merging:** `main` requires **one approving review**. CI (the `test` matrix + `lint`) runs
on every PR; please get it green before merging. (Branch protection does not currently pin
specific required status checks.)

## Making a release

Releases publish to PyPI via **Trusted Publishing (OIDC)** — there is no API token to store
or rotate. `.github/workflows/release.yaml` builds the distribution once, validates it with
`twine check`, and publishes that exact artifact.

### One-time setup (maintainers, PyPI side)

In the **PyPI** and **TestPyPI** web UIs, add a **Trusted Publisher** for the project:

| Field | Value |
|-------|-------|
| Owner | `frazane` |
| Repository | `scoringrules` |
| Workflow | `release.yaml` |
| Environment | `pypi` (on PyPI) / `testpypi` (on TestPyPI) |

Also create the matching GitHub **Environments** (`pypi`, `testpypi`) under repo
**Settings → Environments**. A required-reviewer rule on `pypi` is recommended so a real
publish always pauses for a human.

### Tag convention

Tags are **bare** `X.Y.Z` — **no `v` prefix** — matching the `version` in `pyproject.toml`.
The release workflow compares the tag to `pyproject.toml` literally and fails on a mismatch.
(Historical tags `v0.3.0`…`v0.10.0` keep their `v`; the convention switched to bare starting
after `0.10.0`.)

### Pre-flight: rehearse on TestPyPI

Before the first release, and any time `release.yaml` changes, do a dry run:

```bash
gh workflow run release.yaml -f target=testpypi
gh run watch
```

`build` → `publish-testpypi` should both go green and the version should appear at
<https://test.pypi.org/project/scoringrules/>. (TestPyPI won't accept re-uploading a version
that already exists — bump or use a local `.devN` build if you need to rehearse the same
version again.)

### Cut a release (the supported path)

1. Ensure `main` CI is green.
2. Bump `version` in `pyproject.toml`; merge it to `main`.
3. Create a GitHub **Release** with a tag equal to that version (bare, e.g. `0.11.0`).
4. Publishing the release runs `release.yaml`: it checks `tag == pyproject version`, builds,
   runs `twine check`, and publishes to PyPI via OIDC. Approve the `pypi` deployment if you
   enabled the reviewer rule.
5. Verify: `pip install --upgrade scoringrules` and check the version, or
   <https://pypi.org/project/scoringrules/>.

### Release notes

GitHub auto-generates notes, categorized by label via `.github/release.yml`
(**Breaking changes**, **New features**, **Bug fixes**, **Backend-specific**,
**Documentation**, **CI / tooling**). Categorization keys off the labels on the *merged* PRs,
so label PRs as they merge — especially `breaking` for removals or behavior changes.

### Manual fallback (Actions down / workflow misbehaving)

```bash
uv build
uvx twine check dist/*
uvx twine upload dist/*          # needs PyPI upload credentials
```

Then create the matching tag/Release on GitHub so repo state stays consistent.

### Pre-release checklist

- [ ] CI green on `main`
- [ ] `version` bumped in `pyproject.toml`
- [ ] Merged PRs labeled so release notes categorize correctly (`breaking` where relevant)
- [ ] (first release / after a `release.yaml` change) TestPyPI rehearsal passed
