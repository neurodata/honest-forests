# honest-forests package

## Overview

Honest decision forests and trees implemented efficiently and scikit-learn compliant.

Honest trees and forests use sample splitting to unbias the estimates made in leaves.
This leads to asytmptotic convergence guarantees and empirically better calibration
(e.g. more accurate posterior probabilities).

An example can be seen here, comparing an honest forest to the traditional random forest
and two other ad-hoc calibration approaches.

![overlapping_gaussians.pdf](examples/figures/overlapping_gaussians.pdf)

## Install from Github

```console
git clone https://github.com/neurodata/honest-forests.git
cd honest_forests
pip install -e .
```

## Contributing

### Git workflow

The preferred workflow for contributing to hyppo is to fork the main repository on GitHub, clone, and develop on a
branch. Steps:

1. Fork the [project repository](https://github.com/neurodata/honest-forests) by clicking on the ‘Fork’ button near the top
   right of the page. This creates a copy of the code under your GitHub user account. For more details on how to
   fork a repository see [this guide](https://help.github.com/articles/fork-a-repo/).

2. Clone your fork of the hyppo repo from your GitHub account to your local disk:

   ```sh
   git clone git@github.com:YourGithubAccount/honest-forests.git
   cd honest-forests
   ```

3. Create a feature branch to hold your development changes:

   ```sh
   git checkout -b my-feature
   ```

   Always use a `feature` branch. Pull requests directly to either `dev` or `main` will be rejected
   until you create a feature branch based on `dev`.

4. Develop the feature on your feature branch. Add changed files using `git add` and then `git commit` files:

   ```sh
   git add modified_files
   git commit
   ```

   After making all local changes, you will want to push your changes to your fork:

   ```sh
   git push -u origin my-feature
   ```

### Pull Request Checklist

We recommended that your contribution complies with the following rules before you submit a pull request:

- Follow the [coding-guidelines](#guidelines).
- Give your pull request a helpful title that summarizes what your contribution does.
- Link your pull request to the issue (see:
  [closing keywords](https://docs.github.com/en/github/managing-your-work-on-github/linking-a-pull-request-to-an-issue)
  for an easy way of linking your issue)
- All public methods should have informative docstrings with sample usage presented as doctests when appropriate.
- At least one paragraph of narrative documentation with links to references in the literature (with PDF links when
  possible) and the example.
- If your feature is complex enough that a doctest is insufficient to fully showcase the utility, consider creating a
  Jupyter notebook to illustrate use instead
- All functions and classes must have unit tests. These should include, at the very least, type checking and ensuring
  correct computation/outputs.
- All code should be automatically formatted by `black`. You can run this formatter by calling:

  ```sh
  pip install black
  black path/to/your_module.py
  ```

<!-- - Ensure all tests are passing locally using `pytest`. Install the necessary
  packages by:

  ```sh
  pip install pytest pytest-cov
  pytest
  ``` -->

### Coding Guidelines

Uniformly formatted code makes it easier to share code ownership. hyppo package closely follows the official
Python guidelines detailed in [PEP8](https://www.python.org/dev/peps/pep-0008/) that detail how code should be
formatted and indented. Please read it and follow it.

### Docstring Guidelines

Properly formatted docstrings are required for documentation generation by Sphinx. The hyppo package closely
follows the numpydoc guidelines. Please read and follow the
[numpydoc](https://numpydoc.readthedocs.io/en/latest/format.html#overview) guidelines. Refer to the
[example.py](https://numpydoc.readthedocs.io/en/latest/example.html#example) provided by numpydoc.
