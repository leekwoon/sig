from setuptools import find_packages
from setuptools import setup

setup(
    name="sig",
    version="1.0.0",
    description="Spectral Integrated Gradients for Coarse-to-Fine Feature Attribution (KDD 2026).",
    url="https://github.com/leekwoon/sig",
    license="MIT",
    packages=find_packages(where="."),
    package_dir={"": "."},
    python_requires=">=3.9",
)
