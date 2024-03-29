#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xe7fbdb8

# Compiled with Coconut version 2.0.0-post_dev23 [How Not to Be Seen]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os as _coconut_os
_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.dirname(_coconut_os.path.abspath(__file__)))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os.path.dirname(_coconut_cached_module.__file__) != _coconut_file_dir:  # type: ignore
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_dir)
_coconut_module_name = _coconut_os.path.splitext(_coconut_os.path.basename(_coconut_file_dir))[0]
if _coconut_module_name and _coconut_module_name[0].isalpha() and all(c.isalpha() or c.isdigit() for c in _coconut_module_name) and "__init__.py" in _coconut_os.listdir(_coconut_file_dir):
    _coconut_full_module_name = str(_coconut_module_name + ".__coconut__")
    import __coconut__ as _coconut__coconut__
    _coconut__coconut__.__name__ = _coconut_full_module_name
    for _coconut_v in vars(_coconut__coconut__).values():
        if getattr(_coconut_v, "__module__", None) == str("__coconut__"):
            try:
                _coconut_v.__module__ = _coconut_full_module_name
            except AttributeError:
                _coconut_v_type = type(_coconut_v)
                if getattr(_coconut_v_type, "__module__", None) == str("__coconut__"):
                    _coconut_v_type.__module__ = _coconut_full_module_name
    _coconut_sys.modules[_coconut_full_module_name] = _coconut__coconut__
from __coconut__ import *
from __coconut__ import _coconut_call_set_names, _coconut_handle_cls_kwargs, _coconut_handle_cls_stargs, _namedtuple_of, _coconut, _coconut_super, _coconut_MatchError, _coconut_iter_getitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_raise, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec, _coconut_comma_op, _coconut_multi_dim_arr, _coconut_mk_anon_namedtuple, _coconut_matmul
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
