from setuptools import setup, find_packages

setup(
    name="pymgrid",
    package_dir={"": "src"},
    packages=find_packages("src"),
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
