from setuptools import find_packages, setup

setup(
    name='cprnn',
    packages=find_packages(),
    package_dir={'cprnn': 'src'},
    version='0.1.0',
    description='Exploring multiplicative interactions in CP-RNNs',
    author='Marawan Gamal, Maude Lizaire',
    license='MIT',
)
