import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vmixer",
    version="1.0.0",
    author="MillenniumEarl",
    description="POC for merging similar videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MillenniumEarl/vmixer",
    packages=setuptools.find_packages(),
    license="MIT",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
