import os

import numpy
from numpy.distutils.misc_util import Configuration
import platform


def configuration(parent_package="", top_path=None):
    config = Configuration("compiledtrees", parent_package, top_path)
    libraries = []
    if os.name == 'posix':
        libraries.append('m')
    if platform.python_implementation() != "PyPy":
        config.add_extension("_compiled",
                            sources=["_compiled.c"],
                            include_dirs=[numpy.get_include()],
                            libraries=libraries,
                            extra_link_args=["-Wl,--allow-multiple-definition"],
                            extra_compile_args=["-O3", "-Wno-unused-function"])
    config.add_subpackage("tests")
    return config

if __name__ == "__main__":
    from numpy.distutils.core import setup
    setup(**configuration(top_path='').todict())
