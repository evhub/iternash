#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xa228539a

# Compiled with Coconut version 1.5.0-post_dev24 [Fish License]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import *
from __coconut__ import _coconut_call_set_names, _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match, _coconut_reiterable
if _coconut_sys.version_info >= (3,):
    _coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

from math import log

sys = _coconut_sys


Str = (str, bytes)


def clean_env(env):
    """Make a copy of env without game."""
    new_env = env.copy()
    del new_env["game"]
    return new_env


def printret(obj):
    """Print then return _obj_."""
    print(obj)
    return obj


def printerr(*args):
    """Print to standard error."""
    print(*args, file=sys.stderr)


def clip(x, m=sys.float_info.epsilon, M=1 - sys.float_info.epsilon):
    """Clip x into [m, M] (defaults to [eps, 1-eps])."""
    if m is not None and x <= m:
        return m
    elif M is not None and x >= M:
        return M
    else:
        return x


def safe_log(x):
    """Safe log allows calling log on floats that could be zero."""
    return log(x if x != 0 else sys.float_info.epsilon)


def real(x):
    """Get only the real part of x."""
    return x.real if isinstance(x, complex) else x


def repeat(iterable):
    """Infinitely repeat the given iterable."""
    while True:
        _coconut_yield_from = _coconut.iter(iterable)
        while True:
            try:
                yield _coconut.next(_coconut_yield_from)
            except _coconut.StopIteration as _coconut_yield_err:
                _coconut_yield_from_0 = _coconut_yield_err.args[0] if _coconut.len(_coconut_yield_err.args) > 0 else None
                break

        _coconut_yield_from_0
