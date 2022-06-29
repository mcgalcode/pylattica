from setuptools import setup, find_packages

setup(
    name='rxn_ca',
    version='1.0.0',
    url='https://github.com/mypackage.git',
    author='Max Gallant',
    author_email='maxg@lbl.gov',
    description='A cellular automaton for solid state reactions',
    packages=find_packages(),
    install_requires=[
        'numpy >= 1.21.5',
        'matplotlib >= 3.5.1',
        'scipy >= 1.8.0',
        'tqdm >= 4.63.0',
        'reaction-network >= 5.0.0',
        'pymatgen >= 2022.3.29',
        'plotly >= 5.6.0'
    ],
)