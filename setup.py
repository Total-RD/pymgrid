from setuptools import setup, find_packages

setup(
    name="pymgrid",
    package_dir={"": "pymgrid"},
    packages=find_packages("pymgrid"),
    version="0.1.0",
    python_requires=">=3.7",
    install_requires=[
        "requests",
        "pandas",
        "numpy",
        "cvxpy",
        "statsmodels",
        "matplotlib",
        "plotly",
        "cufflinks",
        "gym",
    ],
)
