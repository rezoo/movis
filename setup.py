from setuptools import find_packages, setup

setup(
    name="zunda",
    version="0.3",
    author="Masaki Saito",
    author_email="msaito@preferred.jp",
    description="A video editing library",
    packages=find_packages(),
    include_package_data=True,
    package_data={
        "zunda": ["assets/*", "py.typed"],
    },
    python_requires=">3.9.0",
    install_requires=[
        "pandas>=1.0.1",
        "numpy>=1.18.1",
        "pydub>=0.25.1",
        "Pillow>=8.2.0",
        "pdf2image>=1.16.3",
        "ffmpeg-python>=0.2.0",
        "imageio>=2.31.1",
        "imageio-ffmpeg>=0.4.8",
        "tqdm>=4.46.0",
        "cachetools>=4.2.2",
    ],
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
    ],
)
