#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x95f1f33f

# Compiled with Coconut version 1.4.3-post_dev28 [Ernest Scribbler]

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

from iternash import Game
from iternash import agent
from iternash import hist_agent
from iternash import debug_all_agent

import numpy as np

from matplotlib import pyplot as plt


# = GENERIC UTILS =

C = 0
D = 1


def coop_with_prob(p):
    return np.random.binomial(1, 1 - p)


common_params = dict(INIT_C_PROB=1, PD_PAYOFFS=[[2, 3], [-1, 0],])


a_hist_1step = hist_agent("a_hist", "a", maxhist=1)


@agent(name="r")
def self_pd_reward(env):
    if env["a_hist"]:
        a1 = env["a_hist"][-1]
    else:
        a1 = coop_with_prob(env["INIT_C_PROB"])
    a2 = env["a"]
    return env["PD_PAYOFFS"][a1][a2]


# = POLICY GRADIENT GAME =

pol_grad_params = common_params.copy()
pol_grad_params.update(THETA_LR=0.001, CLIP_EPS=0.01)


@agent(name="a")
def pol_grad_act(env):
    return coop_with_prob(env["theta"])


@agent(name="theta", default=np.random.random())
def pol_grad_update(env):
    lr = env["THETA_LR"]
    eps = env["CLIP_EPS"]
    th = env["theta"]
# grad[th] E[r(a) | a~pi[th]]
# = sum[a] grad[th] p(a|pi[th]) r(a)
# = sum[a] r(a) grad[th] p(a|pi[th])
# = sum[a] r(a) p(a|pi[th]) grad[th] log(p(a|pi[th]))
# = E[r(a) grad[th] log(p(a|pi[th]) | a~pi[th]]
    if env["a"] == C:
# grad[th] log(p(C|pi[th]))
# = grad[th] log(th)
# = 1/th
        th += lr * env["r"] * (1 / th)
    elif env["a"] == D:
# grad[th] log(p(D|pi[th])
# = grad[th] log(1 - th)
# = -1/(1 - th)
        th += lr * env["r"] * (-1 / (1 - th))
    else:
        raise ValueError("got invalid action {_coconut_format_0}".format(_coconut_format_0=(a)))
    return np.clip(th, eps, 1 - eps)


pol_grad_game = Game("pol_grad", pol_grad_act, self_pd_reward, pol_grad_update, a_hist_1step, **pol_grad_params)


# = Q LEARNING GAME =

q_params = common_params.copy()
q_params.update(EXPLORE_EPS=0.2, BOLTZ_TEMP=1, Q_LR=0.001)


@agent(name="a")
def q_eps_greedy_act(env):
    eps = env["EXPLORE_EPS"]
    QC = env["qs"][C]
    QD = env["qs"][D]
    if QC == QD or np.random.random() <= eps:
        return coop_with_prob(0.5)
    else:
        return C if QC > QD else D


@agent(name="a")
def q_boltz_act(env):
    GUMB_C = env["BOLTZ_TEMP"]
    qs = np.array(env["qs"], dtype=float)
    zs = np.random.gumbel(size=qs.shape)
    return np.argmax(qs + GUMB_C * zs)


@agent(name="qs", default=[0, 0], extra_defaults=dict(q_sums=[0, 0], q_counts=[0, 0]))
def q_true_avg_update(env):
    a = env["a"]
    env["q_sums"][a] += env["r"]
    env["q_counts"][a] += 1
    env["qs"][a] = env["q_sums"][a] / env["q_counts"][a]
    return env["qs"]


@agent(name="qs", default=[0, 0])
def q_run_avg_update(env):
    lr = env["Q_LR"]
    a = env["a"]
    env["qs"][a] = (1 - lr) * env["qs"][a] + lr * env["r"]
    return env["qs"]


q_eps_greedy_true_avg_game = Game("q_eps_greedy_true_avg", q_eps_greedy_act, self_pd_reward, q_true_avg_update, a_hist_1step, **q_params)


q_boltz_run_avg_game = Game("q_boltz_run_avg", q_boltz_act, self_pd_reward, q_run_avg_update, a_hist_1step, **q_params)


# = MAIN =

if __name__ == "__main__":
# pol_grad_game.add_agents(
#     debug_all_agent(period=100)
# ).run(10000)

# q_eps_greedy_true_avg_game.add_agents(
#     debug_all_agent(period=100)
# ).run(10000)

# q_boltz_run_avg_game.add_agents(
#     debug_all_agent(period=100)
# ).run(10000)

    g = q_eps_greedy_true_avg_game.add_agents(hist_agent("qcs", lambda env: env["qs"][C]), hist_agent("qds", lambda env: env["qs"][D]))
    g.run(10000)

    fig, axs = plt.subplots(1)
    g.plot(axs, None, "qcs")
    g.plot(axs, None, "qds")
    axs.set(ylabel="qs")
    axs.legend()
# plt.show()
