from setuptools import setup, find_packages

setup(
    name='rxn_ca',
    version='1.0.0',
    url='https://github.com/mypackage.git',
    author='Max Gallant',
    author_email='maxg@lbl.gov',
    description='A cellular automaton for solid state reactions',
    packages=find_packages(),

)

from pathlib import Path

from setuptools import find_packages, setup

module_dir = Path(__file__).resolve().parent

with open(module_dir / "README.md") as f:
    long_desc = f.read()

setup(
    name="pylatca",
    description="A library for implementing lattice automata",
    use_scm_version={"version_scheme": "python-simplified-semver"},
    setup_requires=["setuptools_scm"],
    long_description_content_type="text/markdown",
    author="Max Gallant",
    author_email="mcgallant72@gmail.com",
    license="modified BSD",
    packages=find_packages("src"),
    package_dir={"": "src"},
    package_data={"jobflow": ["py.typed"]},
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        'numpy >= 1.21.5',
        'matplotlib >= 3.5.1',
        'scipy >= 1.8.0',
        'tqdm >= 4.63.0',
        'reaction-network @ git+https://git@github.com/GENESIS-EFRC/reaction-network@dev#egg=reaction-network',
        'pymatgen >= 2022.3.29',
        'plotly >= 5.6.0',
        'jobflow >= 0.1.8'
    ],
    extras_require={
        "docs": [
            "sphinx==5.2.3",
            "furo==2022.9.29",
            "m2r2==0.3.3",
            "ipython==8.5.0",
            "nbsphinx==0.8.9",
            "nbsphinx-link==1.3.0",
            "autodoc_pydantic==1.7.2",
        ],
        "tests": [
            "pytest==7.1.3",
            "pytest-cov==4.0.0",
            "matplotlib==3.6.1",
            "pydot==1.4.2",
            "moto==4.0.7",
        ],
        "dev": ["pre-commit>=2.12.1"],
        "vis": ["matplotlib", "pydot"],
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Intended Audience :: System Administrators",
        "Intended Audience :: Information Technology",
        "Operating System :: OS Independent",
        "Topic :: Other/Nonlisted Topic",
        "Topic :: Scientific/Engineering",
    ],
    python_requires=">=3.8",
    tests_require=["pytest"],
)