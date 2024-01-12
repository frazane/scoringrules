# Contributing

We welcome contributions! You can help improve the library in many ways:

- Report issues or propose enhancements using on the [GitHub issue tracker](https://github.com/frazane/scoringrules/issues)
- Improve or extend the codebase
- Improve or extend the documentation

## Getting started

Fork the repository on GitHub and clone it to your computer:

```
git clone https://github.com/<your-username>/scoringrules.git
```

We use [poetry](https://python-poetry.org/) for dependencies management and packaging. Once you've set it up, you can install the library and all dependencies (including development dependencies), install the pre-commit hooks and activate the environment:

```
poetry install
poetry run pre-commit install
poetry shell
```

From here you can work on your changes! Once you're satisfied with your changes (and followed the additional instructions below), push everything to your repository and open a pull request on GitHub.


### Contributing to the codebase
Don't forget to include new tests if necessary, then make sure that all tests are passing with

```
pytest tests/
```

### Contributing to the documentation

You can work on the documentation by modifying `mkdocs.yaml` and files in `docs/`. The most convenient way to do it is to run

```
mkdocs serve
```

and open the locally hosted documentation on your browser. It will be updated automatically every time you make changes and save.
