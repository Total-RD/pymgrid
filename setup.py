from pathlib import Path
from setuptools import setup, find_packages

VERSION = (Path(__file__).parent / "version.txt").read_text()
DESCRIPTION = "A simulator for tertiary control of electrical microgrids"
DOWNLOAD_URL = f"https://github.com/Total-RD/pymgrid/archive/refs/tags/v{VERSION}.tar.gz"
MAINTAINER = "Avishai Halev"
MAINTAINER_EMAIL = "avishaihalev@gmail.com"
LICENSE = "GNU LGPL 3.0"
PROJECT_URLS = {"Source Code": "https://github.com/Total-RD/pymgrid"}

EXTRAS = dict()
EXTRAS["genset_mpc"] = ["Mosek", "cvxopt"]
EXTRAS["dev"] = ["pytest", "flake8", *EXTRAS["genset_mpc"]]
EXTRAS["all"] = list(set(sum(EXTRAS.values(), [])))


setup(
    name="pymgrid",
    package_dir={"": "src"},
    packages=find_packages("src"),
    python_requires=">=3.6",
    version=VERSION,
    maintainer=MAINTAINER,
    maintainer_email=MAINTAINER_EMAIL,
    download_url=DOWNLOAD_URL,
    project_urls=PROJECT_URLS,
    description=DESCRIPTION,
    license=LICENSE,
    long_description=(Path(__file__).parent / "README.md").read_text(),
    long_description_content_type="text/markdown",
    include_package_data=True,
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
        "tqdm",
        "pyyaml"
    ],
    extras_require=EXTRAS
)
