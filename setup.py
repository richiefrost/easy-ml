import setuptools

with open("README.md", "r") as f:
    long_description = f.read()

setuptools.setup(
    name="easy-ml-rsfrost",
    version="0.0.4",
    author="Richie Frost",
    author_email="richard.scott.frost@gmail.com",
    description="Simplified machine learning training with scikit-learn and pandas",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/richiefrost/easy-ml",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)