import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vmixer",
    version="0.0.1",
    author="MillenniumEarl",
    description="POC for merging similar videos",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/MillenniumEarl/vmixer",
    packages=setuptools.find_packages(),
    install_requires=[],
    license="MIT",
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
)
