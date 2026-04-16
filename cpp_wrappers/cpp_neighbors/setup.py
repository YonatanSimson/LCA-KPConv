from numpy import get_include
from setuptools import Extension, setup

# Adding OpenCV to project
# ************************

# Adding sources of the project
# *****************************

SOURCES = ["../cpp_utils/cloud/cloud.cpp",
             "radius_neighbors/neighbors.cpp",
             "wrapper.cpp"]

module = Extension(
    name="radius_neighbors",
    sources=SOURCES,
    # Default libstdc++ ABI matches modern PyTorch Linux wheels; avoid forcing old ABI=0.
    extra_compile_args=["-std=c++11"],
)


setup(ext_modules=[module], include_dirs=[get_include()])








