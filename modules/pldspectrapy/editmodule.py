# TODO: right now this assumes a .py file exists; how to change that?

import imp, sys, os
import numpy as np
from copy import deepcopy
from six import iteritems
if sys.version_info >= (3, 0):
    def execfile(filename, globals=None, locals=None):
         with open(filename, 'r', encoding='utf8') as f:
             exec(f.read(), globals, locals)


class ImmutableDict(dict):
    # Suclassing dict may not be ideal, but it seems to work as an immutable namespace
    # http://www.kr41.net/2016/03-23-dont_inherit_python_builtin_dict_type.html
    def __init__(self, immutables):
        self.immutables = tuple(immutables.keys())
        self.return_copies = False
        super(ImmutableDict, self).__init__(immutables)
    def __getitem__(self, y):
        v = super(ImmutableDict, self).__getitem__(y)
        if self.return_copies and y in self.immutables:
            return deepcopy(v)
        return v
    def __setitem__(self, i, y):
        if i not in self.immutables:
            super(ImmutableDict, self).__setitem__(i, y)

class PrintBlocker(object):
    STDOUT_DEFAULT = sys.stdout
    def __enter__(self):
        sys.stdout = open(os.devnull, 'w')
    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            sys.stdout.close()
        finally:
            sys.stdout = self.STDOUT_DEFAULT

PRINT_BLOCKER = PrintBlocker()


def _load_module(name, path, locals_dict):
    try:
        module = sys.modules[name]
    except KeyError:
        fp, pathname, description = imp.find_module(name, [path])
        try:
            with PRINT_BLOCKER:
                module =  imp.load_module(name, fp, pathname, description)
        finally:
            # Since we may exit via an exception, close fp explicitly.
            if fp:
                fp.close()
    for k, v in iteritems(module.__dict__):
        if not k.startswith('_'):
            locals_dict[k] = v
    module_path = os.path.join(os.path.split(module.__file__)[0], module.__name__ + '.py')
    old_locals = []
    for k, v in iteritems(locals_dict):
        if k.startswith('_'):
            continue
        try:
            old_locals.append((k, deepcopy(v)))
        except:
            old_locals.append((k, v))
    old_locals = dict(old_locals)
    return (module_path, old_locals)


def _finalize_modifications(module_path, old_locals, locals_dict):
    modified  = {}
    for k, v in iteritems(locals_dict):
        if k.startswith('_'):
            continue
        try:
            v_old = old_locals[k]
        except KeyError:
            modified[k] = v
        else:
            try:
                if v_old != v:
                    modified[k] = v
            except ValueError:
                # Thrown if v is numpy array (ValueError: The truth value of an array with more than one element is ambiguous. Use a.any() or a.all())
                # TODO: can this be thrown in other cases? Corner cases that could cause failure?
                if isinstance(v, dict):
                    try:
                        for subk, subv in iteritems(v):
                            if not np.array_equal(subv, v_old[subk]):
                                modified[k] = v
                                break
                    except KeyError:
                        modified[k] = v
                else:
                    if not np.array_equal(v_old, v):
                        modified[k] = v

    namespace = ImmutableDict(modified)
    with PRINT_BLOCKER:
        execfile(module_path, namespace)

    for k, v in iteritems(namespace):
        if not k.startswith('_') and not k in modified:
            locals_dict[k] = v
    return modified