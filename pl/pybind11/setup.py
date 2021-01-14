from setuptools import setup
from setuptools import Extension

module = Exception(
    name='example', 
    sources=['example.cpp'],
    include_dirs = ['pybind11/include', 'anaconda3/include'],
    # library_dirs = [''],
    # libraries = [''], 
    language = 'c++'
)   

setup(
    name = 'demo',
    version = '0.1',
    ext_modules=[module]

)