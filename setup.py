from setuptools import setup

setup(
    name='zunda',
    version='0.0.1',
    author='Masaki Saito',
    author_email='msaito@preferred.jp',
    description='A simple python package for creating simple zunda videos',
    packages=['zunda'],
    install_requires=[
        'numpy>=1.18.1',
        'pandas>=1.0.1',
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
    ],
)