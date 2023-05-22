from setuptools import setup, find_packages

# Specify the required dependencies for the package
install_requires = [
    'numpy==1.22.4',
    'pandas==1.5.3',
    'scikit-learn==1.2.2',
    'matplotlib==3.7.1',
    'seaborn==0.12.2',
    'pytest==7.2.2',
    'tqdm==4.65.0',
    'pytorch==2.0.1',
    'torchvision==0.15.2'
]

setup(
    name='protein_universe_annotate',
    version='0.1',
    packages=find_packages("src"),
    package_dir={"": "src"},
    description='Predict the function of protein domains, based on the PFam dataset.',
    install_requires=install_requires
)
