from setuptools import setup, find_packages

def read_requirements(filename):
    with open(filename) as f:
        return [line.strip() for line in f if line.strip() and not line.startswith('#')]

setup(
    name="agentarium",
    version="0.1.0",
    packages=find_packages(),
    install_requires=read_requirements('requirements.txt'),
    author="thytu",
    description="A framework for managing and orchestrating AI agents",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/thytu/Agentarium",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.10",
)
