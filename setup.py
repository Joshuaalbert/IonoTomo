#!/usr/bin/env python

from setuptools import setup
from setuptools import find_packages

__minimum_numpy_version__ = '1.9.0'
setup_requires = ['numpy>=' + __minimum_numpy_version__]

setup(name='ionotomo',
      version='0.0.1',
      description='Ionosphere Tomographic Imaging',
      author=['Josh Albert'],
      author_email=['albert@strw.leidenuniv.nl'],
##      url='https://www.python.org/sigs/distutils-sig/',
    setup_requires=setup_requires,  
    tests_require=[
        'pytest>=2.8',
    ],
    package_data=["ionotomo.astro.arrays/*"]
    package_dir = {'':'src'},
      packages=find_packages('src')
     )

