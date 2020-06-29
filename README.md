# [McKinsey Prohack Competition](https://prohack.org/)

## Setup python project

First clone the project

```bash
git clone git@github.com:omariyassine/ProHack.git
```

We're using the Pipenv as an environment manager for our development workflow. 

```bash
pip install pipenv
```

If you're on macOs and you prefer brew :

```bash
brew install pipenv
```

To setup the environment and the dependencies:

```bash
pipenv install --dev
```

The `--dev` flag is important to install the developement requirements

## Linting

We use [black](https://github.com/psf/black) as our default linter.

To *automatically trigger* black before each commit, init [pre-commit](https://pre-commit.com/):

```bash
pipenv run pre-commit install
```
