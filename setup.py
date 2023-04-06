from pathlib import Path
from setuptools import setup, find_packages

v = {}
exec(open('src/pymgrid/version.py').read(), v)  # read __version__
VERSION = v['__version__']
DESCRIPTION = "A simulator for tertiary control of electrical microgrids"
DOWNLOAD_URL = f"https://github.com/Total-RD/pymgrid/archive/refs/tags/v{VERSION}.tar.gz"
MAINTAINER = "Avishai Halev"
MAINTAINER_EMAIL = "avishaihalev@gmail.com"
LICENSE = "GNU LGPL 3.0"
PROJECT_URLS = {"Source Code": "https://github.com/ahalev/python-microgrid",
                "Documentation": "https://python-microgrid.readthedocs.io/"}

EXTRAS = dict()
EXTRAS["genset_mpc"] = ["Mosek", "cvxopt"]
EXTRAS["dev"] = [
    "pytest",
    "pytest-subtests",
    "flake8",
    "sphinx",
    "pydata_sphinx_theme",
    "numpydoc",
    "nbsphinx",
    "nbsphinx-link",
    *EXTRAS["genset_mpc"]]

EXTRAS["rtd"] = ["ipython"]

EXTRAS["all"] = list(set(sum(EXTRAS.values(), [])))


setup(
    name="python-microgrid",
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
