from setuptools import setup, find_packages

setup(
    name="ENDIGate",
    version="0.1.0",
    author="Sidharth",
    author_email="sidharth.iiserm@gmail.com",
    description="End-to-End Neural Differentiable Indicator Gating for Feature Selection and Neural Architecture Discovery",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/sidhu2690/ENDIGate",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "matplotlib>=3.7.0",
        "numpy>=1.24.0",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
