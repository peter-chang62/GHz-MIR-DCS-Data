"""
Simple function to search surrounding folders for a package and make that
package importable by python by adding the directory to sys.path.

usage: from packfind import find_package; find_package('packagename')

author: rjw

last modified: 2019-02-19 rjw
"""

import os, sys

if sys.version_info >= (3, 0):
    from importlib import find_loader
else:
    from pkgutil import find_loader
    range = xrange

def find_package(package_name, search_levels=3):
    if find_loader(package_name) is not None:
        return
    pkg_init = os.path.join(package_name, '__init__.py')
    path = os.getcwd()
    for i in range(search_levels):
        path = os.path.abspath(os.path.join(path, os.pardir))
        for f in os.listdir(path):
            test_path = os.path.join(path, f)
            if os.path.isdir(test_path):
                if os.path.isfile(os.path.join(test_path, pkg_init)):
                    if not test_path in sys.path:
                        sys.path.append(test_path)
                    return
    raise ImportError('Failed to find package ' + package_name)