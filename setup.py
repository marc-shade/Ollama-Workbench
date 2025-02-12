from setuptools import setup, find_packages

# Setup pyarrow
setup(
    name='OllamaWorkbench',
    version='0.1.0',
    packages=find_packages(),
    include_package_data=True,
    install_requires=[
        'pyarrow'
    ],
)
