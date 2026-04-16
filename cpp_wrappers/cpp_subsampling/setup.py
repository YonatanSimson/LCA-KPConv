from numpy import get_include
from setuptools import Extension, setup

# Adding OpenCV to project
# ************************

# Adding sources of the project
# *****************************

SOURCES = ["../cpp_utils/cloud/cloud.cpp",
             "grid_subsampling/grid_subsampling.cpp",
             "wrapper.cpp"]

module = Extension(
    name="grid_subsampling",
    sources=SOURCES,
    extra_compile_args=["-std=c++11"],
)


setup(ext_modules=[module], include_dirs=[get_include()])








