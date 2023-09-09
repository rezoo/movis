import pathlib

from setuptools import find_packages, setup

BASE_DIR = pathlib.Path(__file__).resolve().parent


def parse_requirements(name: str = "requirements.txt"):
    with open(BASE_DIR / name) as f:
        return f.read().splitlines()


setup(
    name="movis",
    version="0.5",
    author="Masaki Saito",
    author_email="msaito@preferred.jp",
    description="A video editing library",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "movis": ["py.typed"],
    },
    python_requires=">=3.9.0",
    install_requires=parse_requirements(),
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
