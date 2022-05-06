import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="housing_price",
    version="0.3",
    author="Anubhav Yadav",
    author_email="anubhav.yadav@tigeranalytics.com",
    description="Package made for assignment 4.1",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/tigerAnubhav/mle-training",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
)
