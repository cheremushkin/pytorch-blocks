from setuptools import setup, find_packages

with open('requirements.txt', 'r') as f:
    install_requires = f.read().split('\n')

__version__ = '0.0.3'

setup(
    name='pytorch_blocks',
    version=__version__,
    packages=find_packages(),
    url='',
    license='Apache License 2.0',
    author='Ilya Cheremushkin',
    author_email='',
    description='',
    install_requires=install_requires,
    package_data={},
    dependency_links=[],
    include_package_data=True,
)
