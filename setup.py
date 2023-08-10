from setuptools import setup, find_packages

with open('requirements.txt', 'r') as fp:
    requirements = fp.read().splitlines()

setup(
    name='fx_signals',
    version='0.0.1',
    description='FX Signals Generator',
    packages=find_packages(),
    install_requires=requirements
)

