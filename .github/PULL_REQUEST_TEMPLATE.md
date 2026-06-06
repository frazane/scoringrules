## Summary

<!-- What does this PR do and why? -->

## Checklist

- [ ] Tests pass across all backends — `uv sync --all-extras --dev` then `uv run pytest tests/` (exercises every installed backend: numpy, numba, jax, torch)
- [ ] New/changed numerics touch both the array-API and numba paths where applicable
- [ ] Docs / docstrings updated if public behavior changed
- [ ] A release-note label is applied (`breaking`, `enhancement`, `bug`, `backend`, `documentation`, or `ci`)
