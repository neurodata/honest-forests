from setuptools import setup, find_packages

VERSION = 1.0
PACKAGE_NAME = "honest_forests"
DESCRIPTION = "Scikit-learn compliant honest decision tree and forest implementations with additional utilities."
with open("README.md", "r") as f:
    LONG_DESCRIPTION = f.read()
AUTHOR = ("Ronan Perry",)
AUTHOR_EMAIL = "rflperry@gmail.com"
with open('requirements.txt') as f:
    REQUIRED_PACKAGES = f.read().splitlines()

setup(
    name=PACKAGE_NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    install_requires=REQUIRED_PACKAGES,
    license="MIT",
    packages=find_packages(),
)
