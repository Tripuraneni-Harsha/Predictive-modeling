from setuptools import setup, find_packages

setup(
    name="src.traffic_insights",
    version="0.1",
    packages=find_packages(),
    description="A Python package for analyzing Traffic Data",
    author="Harsha",
    author_email="harshatripuraneni@gmail.com",
    url="https://github.com/Tripuraneni-Harsha/Predictive-modeling",
    install_requires=["pandas", "numpy", "matplotlib", "seaborn", "scikit-learn"],
)
