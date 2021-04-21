#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xbd1022ee

# Compiled with Coconut version 1.5.0-post_dev24 [Fish License]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.dirname(_coconut_os_path.abspath(__file__)))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import *
from __coconut__ import _coconut_call_set_names, _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match, _coconut_reiterable
if _coconut_sys.version_info >= (3,):
    _coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

import math
from pprint import pprint

from scipy import stats
from matplotlib import pyplot as plt

from itergame.game import Game
from itergame.agent import agent
from itergame.agent import bbopt_agent
from itergame.agent import debug_agent


# default values of all the parameters
default_params = dict(beta_N=10, beta_D=10, beta_mu=6, norm_mu=6, norm_sd=2, logistic_loc=5, logistic_scale=1, d_min=0, d_max=10, improvement=0.5)


# probability of difficulty using beta distribution
@agent(name="p_d")
def beta_p_d_agent(env):
    N = env["beta_N"]
    D = env["beta_D"]
    mu = env["beta_mu"]
    a = N * mu / D
    b = N - mu
    d = env["d"]
    return stats.beta.pdf(d, a, b, 0, D)


# probability of difficulty using normal distribution
@agent(name="p_d")
def norm_p_d_agent(env):
    mu = env["norm_mu"]
    sd = env["norm_sd"]
    d = env["d"]
    return stats.norm.pdf(d, mu, sd)


def sigma(x, m, w):
    """The logistic function."""
    return 1 / (1 + math.exp(-(x - m) / w))


# probability of success using logistic distribution
@agent(name="p_s_given_d")
def p_s_given_d_agent(env):
    loc = env["logistic_loc"]
    scale = env["logistic_scale"]
    d = env["d"]
    return 1 - sigma(d, loc, scale)


# probability of success using logistic distribution after intervention
@agent(name="p_s_given_d_post")
def p_s_given_d_post_agent(env):
    loc = env["logistic_loc"]
    scale = env["logistic_scale"]
    improvement = env["improvement"]
    d = env["d"]
    return 1 - sigma(d - improvement, loc, scale)


# increase in the probability of success
@agent(name="p_s_inc")
def p_s_inc_agent(env):
    p_d = env["p_d"]
    p_s_given_d = env["p_s_given_d"]
    p_s_given_d_post = env["p_s_given_d_post"]
    return (p_s_given_d_post - p_s_given_d) * p_d


# black-box-optimized d agent that attempts to maximize p_s_inc
bbopt_d_agent = bbopt_agent(name="d", tunable_actor=lambda bb, env: bb.uniform("d", env["d_min"], env["d_max"]), util_func=_coconut.operator.itemgetter(("p_s_inc")), file=__file__)


# debugger that periodically prints relevant info
periodic_debugger = debug_agent("\nd = {d}; P(d) = {p_d}; P(s|d) = {p_s_given_d}; increase in P(s|d) = {p_s_inc}", period=100)


# black-box-optimized d game with beta distributed d
beta_d_game = Game("beta_d", bbopt_d_agent, beta_p_d_agent, p_s_given_d_agent, p_s_given_d_post_agent, p_s_inc_agent, periodic_debugger, **default_params)


# black-box-optimized d game with normally distributed d
norm_d_game = Game("norm_d", bbopt_d_agent, norm_p_d_agent, p_s_given_d_agent, p_s_given_d_post_agent, p_s_inc_agent, periodic_debugger, **default_params)


def run_game(game, num_steps=500, **params):
    """Run the given game with the given parameters."""
    param_repr = (str)((list)((sorted)(params.items())))
    print("Running {_coconut_format_0} with {_coconut_format_1}...".format(_coconut_format_0=(game.name), _coconut_format_1=(param_repr)))
    env = game.clone(name="{_coconut_format_0}[{_coconut_format_1}]".format(_coconut_format_0=(game.name), _coconut_format_1=(param_repr)), **params).run(max_steps=num_steps, stop_at_equilibrium=True)
    pprint(env)
    return env["d"]


if __name__ == "__main__":
    run_game(beta_d_game)

    mus = range(10)
    ds = [run_game(norm_d_game, norm_mu=mu) for mu in mus]

# plt.plot(mus, ds)
# plt.show()
