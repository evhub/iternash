#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x536cdbbb

# Compiled with Coconut version 1.4.3-post_dev11 [Ernest Scribbler]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.dirname(_coconut_os_path.abspath(__file__)))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import *
from __coconut__ import _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_back_pipe, _coconut_star_pipe, _coconut_back_star_pipe, _coconut_dubstar_pipe, _coconut_back_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match
if _coconut_sys.version_info >= (3,):
    _coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

from math import log
from math import exp
from pprint import pprint

from scipy.stats import linregress
from scipy.special import comb
from mpmath import hyp2f1
from matplotlib import pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn import linear_model

from iternash import Game
from iternash import agent
from iternash import expr_agent
from iternash import bbopt_agent
from iternash import debug_agent
from iternash import human_agent


common_params = dict(d=1, m=100, eps=0.01, p_mod=1.0, r_n=0, r_m=1, r_f=0)


# optimal training episodes in the one defection game
baseline_n_agent = expr_agent(name="n", expr="m/p_mod * (1-eps)/eps", default=common_params["m"])


# optimal defection probability in the one defection game
baseline_p_agent = expr_agent(name="p", expr="""(
        (n * p_mod)
        / (n * p_mod + m)
        * (r_m - r_n)/(r_m - r_f)
    )**(1/(m-1))""", default=0.9)


# probability of catastrophe in the one defection game
baseline_PC_agent = expr_agent(name="PC", expr="1 - p**m", default=0.1)


# generic expected reward calculator
ER_agent = expr_agent(name="ER", expr="""
    (1 - (p + (1-p)*(1-p_mod))**n) * r_n
    + (p + (1-p)*(1-p_mod))**n * (
        PC * r_m
        + (1-PC) * r_f
    )
    """)


# probability of catastrophe in the non-sequential two defection game
nonseq_2d_PC_agent = expr_agent(name="PC", expr="1 - p**m - m*(1-p)*p**(m-1)", default=0.1)

@agent(name="PC", default=0.1)
def nonseq_d_PC_agent(env):
    d = env["d"]
    if d == 2:
        return nonseq_2d_PC_agent(env)
    m = env["m"]
    p = env["p"]


# probability of catastrophe in the sequential defection game
    return comb(m, d) * p**(m - d) * (1 - p)**d * (_coconut_forward_compose(hyp2f1, float))(1, d - m, d + 1, (p - 1) / p)
@agent(name="PC", default=0.1)
def exact_seq_d_PC_agent(env):
    m = env["m"]
    d = env["d"]
    PC = 0
    for i in range(1, int(m / d) + 1):
        PC += expr_agent(None, """
        (-1)**(i+1) * (1-p + p*(m-i*d+1)/i) * comb(m-i*d, i-1) * (1-p)**(i*d) * p**(i-1)
        """, vars={"i": i, "comb": comb})(env)
    return PC

seq_d_PC_agent = expr_agent(name="PC", expr="(1 - p**(m-d+1)) * (1-p)**(d-1)", default=0.1)


# black-box-optimized p agent that attempts to maximize ER
bbopt_p_agent = bbopt_agent(name="p", tunable_actor=lambda bb, env: 1 - bb.loguniform("p", 0.00001, 1), util_func=_coconut.operator.itemgetter("ER"), file=__file__, default=0.9)


# black-box-optimized n agent that attempts to set PC to eps
bbopt_n_agent = bbopt_agent(name="n", tunable_actor=lambda bb, env: int(baseline_n_agent(env) * bb.loguniform("n/n_c", 0.001, 1000)), util_func=expr_agent(None, "-abs(log(PC) - log(eps))", vars={"log": log}), file=__file__, default=common_params["m"])


# empirical formula for PC in the conservative nonseq d game
nonseq_d_PC_guess = expr_agent(name="nonseq_d_PC_guess", expr="eps**d * 2**((d-1)*(d+6)/10)")


# empirical formula for PC in the conservative seq d game
seq_d_PC_guess = expr_agent(name="seq_d_PC_guess", expr="eps**d / m**(d-1) * 2**((d-1)*(d+4)/3)")


# agent that prints n, p, PC, ER every 100 steps
periodic_debugger = debug_agent("n = {n}; p = {p}; PC = {PC}; ER = {ER}", period=100)


# absent-minded driver game where catastrophe occurs on the first defection
baseline_game = Game("baseline", baseline_n_agent, baseline_p_agent, baseline_PC_agent, ER_agent, **common_params)


# absent-minded driver game where catastrophe occurs upon the
#  second defection during deployment with a conservative n
#  and p approximated by BBopt
conservative_nonseq_d_game = Game("conservative_nonseq_d", baseline_n_agent, bbopt_p_agent, nonseq_d_PC_agent, ER_agent, nonseq_d_PC_guess, periodic_debugger, default_run_steps=500, **common_params)


# absent-minded driver game where catastrophe occurs if there are ever
#  d sequential defections during deployment with a conservative n
#  and p approximated by BBopt
conservative_seq_d_game = Game("conservative_seq_d", baseline_n_agent, bbopt_p_agent, seq_d_PC_agent, ER_agent, seq_d_PC_guess, periodic_debugger, default_run_steps=500, **common_params)


# game for testing the impact of different p values in the non-sequential
#  vs. sequential cases
def _coconut_lambda_0(env):
    new_env = env.copy()
    new_env["PC"] = env["baseline_PC"]
    return ER_agent(new_env)
def _coconut_lambda_1(env):
    new_env = env.copy()
    new_env["PC"] = env["nonseq_d_PC"]
    return ER_agent(new_env)
def _coconut_lambda_2(env):
    new_env = env.copy()
    new_env["PC"] = env["seq_d_PC"]
    return ER_agent(new_env)
test_game = Game("test", baseline_n_agent, human_agent(name="p"), baseline_PC=baseline_PC_agent, baseline_ER=(_coconut_lambda_0), nonseq_d_PC=nonseq_d_PC_agent, nonseq_d_ER=(_coconut_lambda_1), seq_d_PC=seq_d_PC_agent, seq_d_ER=(_coconut_lambda_2), default_run_steps=100, **common_params)


def run_nonseq_game(d):
    """Run non-sequential d defection game and measure PC."""
    print("\nRunning conservative non-sequential {_coconut_format_0} defection game...".format(_coconut_format_0=(d)))
    env = conservative_nonseq_d_game.reset(name="conservative_nonseq_d{_coconut_format_0}_game".format(_coconut_format_0=(d)), d=d).run()
    pprint(env)
    return env["PC"]


def run_seq_game(d):
    """Run sequential d defection game and measure PC."""
    print("\nRunning conservative sequential {_coconut_format_0} defection game...".format(_coconut_format_0=(d)))
    env = conservative_seq_d_game.reset(name="conservative_seq_d{_coconut_format_0}_game".format(_coconut_format_0=(d)), d=d).run()
    pprint(env)
    return env["PC"]


def print_logregress(ds, logys, yname="y"):
    """Print linear and quadratic regression of ds and logys."""
    m, b, r, p, se = linregress(ds, logys)
    print("\nlog({_coconut_format_0}) = {_coconut_format_1} d + {_coconut_format_2}\t(r**2 = {_coconut_format_3})".format(_coconut_format_0=(yname), _coconut_format_1=(m), _coconut_format_2=(b), _coconut_format_3=(r**2)))
    print("{_coconut_format_0} = {_coconut_format_1} * 2**({_coconut_format_2} d))".format(_coconut_format_0=(yname), _coconut_format_1=(exp(b)), _coconut_format_2=(m / log(2))))
    poly = PolynomialFeatures(degree=2, include_bias=False)
    X = ((poly.fit_transform)((list)(map(lambda x: [x], ds))))
    clf = linear_model.LinearRegression()
    clf.fit(X, logys)
# a d**2 + b d + c
    b, a = clf.coef_
    c = clf.intercept_
    print("log({_coconut_format_0}) = {_coconut_format_1} d**2 + {_coconut_format_2} d + {_coconut_format_3}".format(_coconut_format_0=(yname), _coconut_format_1=(a), _coconut_format_2=(b), _coconut_format_3=(c)))
# (d - 1)(a d - c)
# a d**2 - a d - c d + c
# a d**2 - (a + c) d + c
    print("{_coconut_format_0} = exp((d - 1)({_coconut_format_1} d - {_coconut_format_2}) + {_coconut_format_3} d)".format(_coconut_format_0=(yname), _coconut_format_1=(a), _coconut_format_2=(c), _coconut_format_3=(b + a + c)))
    print("{_coconut_format_0} = 2**((d - 1)({_coconut_format_1} d - {_coconut_format_2}) + {_coconut_format_3} d)".format(_coconut_format_0=(yname), _coconut_format_1=(a / log(2)), _coconut_format_2=(c / log(2)), _coconut_format_3=((a + b + c) / log(2))))
    print("{_coconut_format_0} = exp({_coconut_format_1} ((d - 1)(d - {_coconut_format_2}) + {_coconut_format_3} d))".format(_coconut_format_0=(yname), _coconut_format_1=(a), _coconut_format_2=(c / a), _coconut_format_3=(1 + (b + c) / a)))
    print("{_coconut_format_0} = 2**({_coconut_format_1} ((d - 1)(d - {_coconut_format_2}) + {_coconut_format_3} d))".format(_coconut_format_0=(yname), _coconut_format_1=(a / log(2)), _coconut_format_2=(c / a), _coconut_format_3=(1 + (b + c) / a)))


if __name__ == "__main__":
    print("\nRunning baseline game...")
    baseline_env = baseline_game.run()
    pprint(baseline_env)

    ds = range(1, 5)
    nonseq_PCs = [baseline_env["PC"]] + [run_nonseq_game(d) for d in ds[1:]]
    seq_PCs = [baseline_env["PC"]] + [run_seq_game(d) for d in ds[1:]]

    eps, m = common_params["eps"], common_params["m"]
    nonseq_logys = [log(PC / eps**d) for d, PC in zip(ds, nonseq_PCs)]
    seq_logys = [log(PC / (eps**d / m**(d - 1))) for d, PC in zip(ds, seq_PCs)]

    print_logregress(ds, nonseq_logys, yname="PC_nonseq / eps**d")
    print_logregress(ds, seq_logys, yname="PC_seq / (eps**d / m**(d-1))")

# fig, axs = plt.subplots(2, 2)

# axs[0, 0].set(ylabel="non-sequential log P(C)", xlabel="required defections")
# axs[0, 0].plot(ds, list <| map(log, nonseq_PCs))

# axs[1, 0].set(ylabel="sequential log P(C)", xlabel="required defections")
# axs[1, 0].plot(ds, list <| map(log, seq_PCs))

# axs[0, 1].set(ylabel="non-sequential log(P(C)/eps**d)", xlabel="required defections")
# axs[0, 1].plot(ds, nonseq_logys)

# axs[1, 1].set(ylabel="sequential log(P(c)/(eps**d/m**(d-1)))", xlabel="required defections")
# axs[1, 1].plot(ds, seq_logys)

# plt.show()

# print("\nRunning test game...")
# test_game.run() |> pprint
