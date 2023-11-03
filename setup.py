#!/usr/bin/env python

import ast
import os

from setuptools import setup  # type: ignore[import]


def read(*relpath, **kwargs):
    with open(os.path.join(os.path.dirname(__file__), *relpath),
              encoding=kwargs.get("encoding", "utf8")) as fh:
        return fh.read()

# Extract __version__ from the package __init__.py
# (since it's not a good idea to actually run __init__.py during the build process).
#
# http://stackoverflow.com/questions/2058802/how-can-i-get-the-version-defined-in-setup-py-setuptools-in-my-package
init_py_path = os.path.join("randomthought", "__init__.py")
version = None
try:
    with open(init_py_path) as f:
        for line in f:
            if line.startswith("__version__"):
                module = ast.parse(line, filename=init_py_path)
                expr = module.body[0]
                assert isinstance(expr, ast.Assign)
                v = expr.value
                if type(v) is ast.Constant:
                    # mypy understands `isinstance(..., ...)` but not `type(...) is ...`,
                    # and we want to match on the exact type, not any subclass that might be
                    # added in some future Python version.
                    assert isinstance(v, ast.Constant)
                    version = v.value
                break
except FileNotFoundError:
    pass
if not version:
    raise RuntimeError(f"Version information not found in {init_py_path}")

setup(
    name="randomthought",
    version=version,
    packages=["randomthought"],
    provides=["randomthought"],
    keywords=["artificial-intelligence", "variational-autoencoder", "differentiation", "partial-differential-equations", "numerics", "utilities"],
    install_requires=["matplotlib>=3.3.3", "numpy>=1.22.0", "unpythonic>=0.15.1", "tensorflow>=2.12", "tensorflow_probability>=0.19.0",
                      "visualkeras>=0.0.2", "typeguard>=2.13.3", "imageio>=2.23.0", "scikit-learn>=1.3.0", "opentsne>=1.0.0", "umap-learn>=0.5.3"],
    python_requires=">=3.8",
    author="Juha Jeronen",
    author_email="juha.m.jeronen@gmail.com",
    url="https://github.com/Technologicat/randomthought",
    description="Development package for an AI accelerator for partial differential equations in 2D.",
    long_description=read("README.md"),
    long_description_content_type="text/markdown",
    license="BSD",
    platforms=["Any"],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3 :: Only",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: Implementation :: CPython",
        "Topic :: Software Development :: Libraries",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Utilities"
    ],
    entry_points={},
    zip_safe=True
)
