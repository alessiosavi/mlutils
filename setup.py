from setuptools import setup, find_packages

setup(
    name="mlutils",
    version="0.0.1-rc12",
    author="alessiosavi",
    author_email="alessiosavibtc@gmail.com",
    maintainer="alessiosavi",
    maintainer_email="alessiosavibtc@gmail.com",
    description="A minimal utility for deal with machine learning",
    install_requires=[
        "tensorflow==2.11.1",
        "pandas==1.3.4",
        "numpy==1.21.4"
    ],
    url='https://github.com/alessiosavi/mlutils',
    packages=find_packages()
)
