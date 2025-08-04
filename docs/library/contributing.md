# Contributing

We welcome contributions! You can help improve the library in many ways. For example, by:

- Reporting issues or proposing enhancements via the [GitHub issue tracker](https://github.com/frazane/scoringrules/issues)
- Improving or extending the codebase
- Improving or extending the documentation

## Getting started

To make changes to the library, fork the repository on GitHub and clone it to your computer:

```
git clone https://github.com/<your-username>/scoringrules.git
```

We use [uv](https://docs.astral.sh/uv/) for project management and packaging. Install it by running

```
curl -LsSf https://astral.sh/uv/install.sh | sh
```

You should then be able to install the library and all dependencies (including development dependencies), and install the pre-commit hooks:

```
uv install
uv run pre-commit install
```

From here, you can work on your changes. Once you're satisfied with your changes, and have followed the additional instructions below, push everything to your repository and open a pull request on GitHub.


### Contributing to the codebase

Don't forget to include new tests if necessary. Make sure that all tests are passing with

```
uv run pytest tests/
```

### Contributing to the documentation

You can work on the documentation by modifying files in `docs/`. The most convenient way to do this is to run

```
uvx --with-requirements docs/requirements.txt sphinx-autobuild docs/ docs/_build/
```

and open the locally hosted documentation on your browser. It will be updated automatically every time you make changes and save. If you edit or add pieces of LaTeX math, please make sure they are rendered correctly.
