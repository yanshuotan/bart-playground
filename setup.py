from setuptools import setup, find_packages

runtime_deps = [
    "numpy",
    "scipy",
    "pandas",
    "scikit-learn",
    "numba",
    "arviz",
    "pydantic",
    "tqdm",
    "graphviz",
    "ray[default]",
]

test_extras = [
    "pytest",
    "xgboost",
    "bartz",
    "pmlb",
]

setup(
    name="bart_playground",
    version="0.1.1",
    packages=find_packages(),
    install_requires=runtime_deps,
    extras_require={
        "test": test_extras,
    },
    python_requires=">=3.7",
    description="A fast and modular implementation of BART",
    author="Yan Shuo Tan",
    author_email="yanshuo@nus.edu.sg",
    url="https://github.com/yanshuotan/bart-playground",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
