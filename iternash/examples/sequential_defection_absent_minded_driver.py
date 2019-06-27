#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xf8169469

# Compiled with Coconut version 1.4.0-post_dev40 [Ernest Scribbler]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.dirname(_coconut_os_path.abspath(__file__)))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import *
from __coconut__ import _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_back_pipe, _coconut_star_pipe, _coconut_back_star_pipe, _coconut_dubstar_pipe, _coconut_back_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert
if _coconut_sys.version_info >= (3,):
    _coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

import random

from iternash import expr_agent
from iternash import Game


p_agent = expr_agent(name="p", expr="""(
    (n * p_mod - (d-1)/(1-p))
    / (n * p_mod + m - (d-1)/(1-p))
    * (r_m - r_n)/(r_m - r_f)
)^(1/(m-d))""", default=0.9)


n_agent = expr_agent(name="n", expr="m/p_mod * (1-eps)/eps")


PC_agent = expr_agent(name="PC", expr="p^(-d) * (1-p)^(d-1) * (p^d - p^(m+1))")


game = Game(d=1, p_mod=0.5, m=100, eps=0.01, n=n_agent, r_n=0, r_m=1, r_f=0, p=p_agent, PC=PC_agent)


if __name__ == "__main__":
    game.attach(lambda env: print("p = {p}; PC = {PC}".format(**env)))
    (print)(game.run())
