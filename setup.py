from setuptools import setup, find_packages

# Specify the required dependencies for the package
install_requires = [
    'numpy==1.21.2',
    'pandas==1.3.2',
]

setup(
    name='protein_universe_annotate',
    version='0.1',
    packages=find_packages("src"),
    package_dir={"": "src"},
    description='Predict the function of protein domains, based on the PFam dataset.',
    install_requires=install_requires
)


