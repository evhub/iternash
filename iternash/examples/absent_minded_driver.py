#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xb8efd3b0

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
from math import log

from iternash import expr_agent
from iternash import bbopt_agent
from iternash import Game


common_params = dict(m=100, eps=0.01, p_mod=0.9, r_n=0, r_m=1, r_f=0, min_n_m=0.1, max_n_m=10000)


# conservative estimate of required training episodes
conservative_n_agent = expr_agent(name="n", expr="m/p_mod * (1-eps)/eps", default=common_params["m"])


# optimal defection probability in the sequential defection game
seq_d_p_agent = expr_agent(name="p", expr="""(
    (n * p_mod - (d-1)/(1-p))
    / (n * p_mod + m - (d-1)/(1-p))
    * (r_m - r_n)/(r_m - r_f)
)^(1/(m-d))""", default=0.9)


# probability of catastrophe in the sequential defection game
seq_d_PC_agent = expr_agent(name="PC", expr="(1-p)^(d-1) * (1 - p^(m-d+1))", default=0.1)


# optimal defection probability in the non-sequential two defection game
nonseq_2d_p_agent = expr_agent(name="p", expr="""(
    (n * p_mod)
    / (n * p_mod + m^2 - m)
    * (r_m - r_n)/(r_m - r_f)
)^(1/(m-1))""", default=0.9)


# probability of catastrophe in the non-sequential two defection game
nonseq_2d_PC_agent = expr_agent(name="PC", expr="1 - (m-1)*(1-p)*p^(m-1) - p**m", default=0.1)


# black-box-optimized n agent that attempts to set PC to eps
bbopt_n_agent = bbopt_agent(name="n", tunable_actor=lambda bb, env: int(env["m"] * bb.loguniform("n/m", env["min_n_m"], env["max_n_m"])), util_func=lambda env: -abs(log(env["PC"]) - log(env["eps"])), file=__file__, default=common_params["m"])


# absent-minded driver game where catastrophe occurs if there are
#  ever d sequential defections during deployment
seq_d_game = Game(d=2, n=conservative_n_agent, p=seq_d_p_agent, PC=seq_d_PC_agent, **common_params)


# absent-minded driver game where catastrophe occurs upon the
#  second defection during deployment
nonseq_2d_game = Game(n=bbopt_n_agent, p=nonseq_2d_p_agent, PC=nonseq_2d_PC_agent, **common_params)



if __name__ == "__main__":
    print("Running sequential defection game...")
    seq_d_game.attach(lambda env: print("p = {p}; PC = {PC}".format(**env)))
    (print)(seq_d_game.run())

    print("Running non-sequential two defection game...")
    nonseq_2d_game.attach(lambda env: print("n = {n}; p = {p}; PC = {PC}".format(**env)))
    (print)(nonseq_2d_game.run())
