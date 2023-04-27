from setuptools import setup, find_packages

# Specify the required dependencies for the package
install_requires = [
    'scipy==1.7.0',
    'numpy==1.22.4',
    'pandas==1.5.3',
    'pytest==7.2.2',
    'seaborn==0.12.2',
    'scikit-learn==1.0.2.',
    'tokenizers==0.13.3',
    'tqdm==4.65.0',
    'transformers==4.28.1',
    'tensorflow==2.12.0',
    'numpy==1.21.2',
    'pandas==1.3.2'
]

setup(
    name='protein_universe_annotate',
    version='0.1',
    packages=find_packages("src"),
    package_dir={"": "src"},
    description='Predict the function of protein domains, based on the PFam dataset.',
    install_requires=install_requires
)


