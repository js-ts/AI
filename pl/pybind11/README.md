# [pybind11](https://pybind11.readthedocs.io/en/stable/index.html)

- git clone https://github.com/pybind/pybind11.git
- pip install pytest
- mkdir build
- cd build
- cmake ..
- cmake --build . --config Release --target check

- python setup.py build_ext --inspace