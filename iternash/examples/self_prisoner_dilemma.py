#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xa6a69eb9

# Compiled with Coconut version 1.4.3-post_dev32 [Ernest Scribbler]

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
from math import ceil

from iternash.game import Game
from iternash.agent import agent
from iternash.agent import hist_agent
from iternash.util import repeat

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# = GENERIC UTILS =

C = 0
D = 1

SELF_PD_PAYOFFS = [[2, 3], [-1, 0],]

BUTTON_PAYOFFS = [[1, 1], [0, 0],]

PAY_FORWARD_PAYOFFS = [[1, 2], [-1, 0],]


def coop_with_prob(p):
    return np.random.binomial(1, 1 - p)


common_params = dict(INIT_C_PROB=0.5, PAYOFFS=SELF_PD_PAYOFFS, USE_STATE=False)


a_hist_1step = hist_agent("a_hist_1step", "a", maxhist=1)


def get_prev_a(env):
    if not env["a_hist_1step"]:
        env["a_hist_1step"].append(coop_with_prob(env["INIT_C_PROB"]))
    return env["a_hist_1step"][-1]


@agent(name="r")
def get_reward(env):
    a1, a2 = get_prev_a(env), env["a"]
    return env["PAYOFFS"][a1][a2]


@agent(name="s", default=C)
def get_state(env):
    if env["USE_STATE"]:
        return C
    else:
        return get_prev_a(env)


# = POLICY GRADIENT GAME =

pol_grad_params = common_params.copy()
pol_grad_params.update(POL_GRAD_LR=0.01, CLIP_EPS=0.01)


@agent(name="a")
def pol_grad_act(env):
    return coop_with_prob(env["pcs"][env["s"]])


@agent(name="pcs", default=[np.random.random(), np.random.random()])
def pol_grad_update(env):
    lr = env["POL_GRAD_LR"]
    eps = env["CLIP_EPS"]
    th = env["pcs"]
    a = env["a"]
    s = env["s"]
# grad[th] E[r(a) | a~pi[th]]
# = sum[a] grad[th] p(a|pi[th]) r(a)
# = sum[a] r(a) grad[th] p(a|pi[th])
# = sum[a] r(a) p(a|pi[th]) grad[th] log(p(a|pi[th]))
# = E[r(a) grad[th] log(p(a|pi[th]) | a~pi[th]]
    if a == C:
# grad[th] log(p(C|pi[th]))
# = grad[th] log(th)
# = 1/th
        th[s] += lr * env["r"] * (1 / th[s])
    elif a == D:
# grad[th] log(p(D|pi[th])
# = grad[th] log(1 - th)
# = -1/(1 - th)
        th[s] += lr * env["r"] * (-1 / (1 - th[s]))
    else:
        raise ValueError("got invalid action {_coconut_format_0}".format(_coconut_format_0=(a)))
    th[s] = np.clip(th[s], eps, 1 - eps)
    return th


@agent(name="pcs", default=[np.random.random(), np.random.random()])
def pol_grad_decoupled_update(env):
    new_env = env.copy()
    k = pol_grad_act(env)
    new_env["a"] = k
    return pol_grad_update(new_env)


pol_grad_game = Game("pol_grad", get_state, pol_grad_act, get_reward, pol_grad_update, a_hist_1step, **pol_grad_params)


pol_grad_decoupled_game = Game("pol_grad_decoupled", get_state, pol_grad_act, get_reward, pol_grad_decoupled_update, a_hist_1step, **pol_grad_params)


# = Q LEARNING GAME =

ql_params = common_params.copy()
ql_params.update(EXPLORE_EPS=0.1, BOLTZ_TEMP=1, QL_LR=0.01)


def get_eps_greedy_pc(env, s, eps=None):
    eps = env["EXPLORE_EPS"] if eps is None else eps
    QC = env["qs"][s][C]
    QD = env["qs"][s][D]

    prob_coop = eps / 2
    if QC == QD:
        prob_coop += (1 - eps) / 2
    elif QC > QD:
        prob_coop += 1 - eps
    return prob_coop


@agent(name="pcs")
def eps_greedy_pcs(env):
    return [get_eps_greedy_pc(env, i) for i in range(2)]


def get_boltz_pc(env, s, temp=None):
    temp = env["BOLTZ_TEMP"] if temp is None else temp
    QC = env["qs"][s][C]
    QD = env["qs"][s][D]
    zc = exp(QC / temp)
    zd = exp(QD / temp)
    return zc / (zc + zd)


@agent(name="pcs")
def boltz_pcs(env):
    return [get_boltz_pc(env, i) for i in range(2)]


@agent(name="pcs")
def eps_greedy_decay_pcs(env):
    new_env = env.copy()
    new_env["EXPLORE_EPS"] = 1 / env["M"][env["s"]]
    return eps_greedy_pcs(new_env)


@agent(name="a")
def ql_pcs_act(env):
    return coop_with_prob(env["pcs"][env["s"]])


@agent(name="qs", default=np.zeros((2, 2)), extra_defaults=dict(q_sums=np.zeros((2, 2)), q_counts=np.zeros((2, 2))))
def ql_true_avg_update(env):
    s = env["s"]
    a = env["a"]
    env["q_sums"][s, a] += env["r"]
    env["q_counts"][s, a] += 1
    env["qs"][s, a] = env["q_sums"][s, a] / env["q_counts"][s, a]
    return env["qs"]


@agent(name="qs", default=np.zeros((2, 2)))
def ql_run_avg_update(env):
    s = env["s"]
    a = env["a"]
    al = env["QL_LR"]
    if env.get("QL_LR_DECAY"):
        al /= env["M"][s]
    if env.get("QL_LR_CORRECTION"):
        prob_a = env["pcs"][s] if a == C else 1 - env["pcs"][s]
        al /= prob_a
    env["qs"][s, a] = (1 - al) * env["qs"][s, a] + al * env["r"]
    return env["qs"]


@agent(name="M", default=[1, 1])
def M_counter(env):
    env["M"][env["s"]] += 1
    return env["M"]


@agent(name="qs", default=np.zeros((2, 2)))
def ql_decoupled_update(env):
    al_init = env["QL_LR"]
    s = env["s"]

    num_actions = 2
    k_eps = max(1 / env["M"][s], al_init * num_actions)
    prob_k_coop = get_eps_greedy_pc(env, s, eps=k_eps)
    k = coop_with_prob(prob_k_coop)

    al = al_init
    if env.get("QL_LR_DECAY"):
        al /= env["M"][s]
    if env.get("QL_LR_CORRECTION"):
        if k == C:
            prob_k = prob_k_coop
        else:
            prob_k = 1 - prob_k_coop
        al /= prob_k
    env["qs"][s, k] = (1 - al) * env["qs"][s, k] + al * env["r"]
    return env["qs"]


ql_eps_greedy_true_avg_game = Game("ql_eps_greedy_true_avg", get_state, eps_greedy_pcs, ql_pcs_act, get_reward, ql_true_avg_update, a_hist_1step, **ql_params)


ql_eps_greedy_run_avg_game = Game("ql_eps_greedy_run_avg", get_state, eps_greedy_pcs, ql_pcs_act, get_reward, ql_run_avg_update, a_hist_1step, **ql_params)


ql_boltz_run_avg_game = Game("ql_boltz_run_avg", get_state, boltz_pcs, ql_pcs_act, get_reward, ql_run_avg_update, a_hist_1step, **ql_params)


ql_boltz_true_avg_game = Game("ql_boltz_true_avg", get_state, boltz_pcs, ql_pcs_act, get_reward, ql_true_avg_update, a_hist_1step, **ql_params)


ql_eps_greedy_decay_run_avg_decoupled_lr_decay_correction_game = Game("ql_eps_greedy_decay_run_avg_decoupled_lr_decay_correction", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_reward, ql_decoupled_update, a_hist_1step, M_counter, QL_LR_DECAY=True, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_decay_run_avg_decoupled_game = Game("ql_eps_greedy_decay_run_avg_decoupled", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_reward, ql_decoupled_update, a_hist_1step, M_counter, **ql_params)


ql_eps_greedy_decay_true_avg_game = Game("ql_eps_greedy_decay_true_avg", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_reward, ql_true_avg_update, a_hist_1step, M_counter, **ql_params)


ql_eps_greedy_decay_run_avg_game = Game("ql_eps_greedy_decay_run_avg", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_reward, ql_run_avg_update, a_hist_1step, M_counter, **ql_params)


ql_eps_greedy_decay_run_avg_lr_decay_correction_game = Game("ql_eps_greedy_decay_run_avg_lr_decay_correction", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_reward, ql_run_avg_update, a_hist_1step, M_counter, QL_LR_DECAY=True, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_run_avg_lr_decay_correction_game = Game("ql_eps_greedy_run_avg_lr_decay_correction", get_state, eps_greedy_pcs, ql_pcs_act, get_reward, ql_run_avg_update, a_hist_1step, M_counter, QL_LR_DECAY=True, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_run_avg_lr_decay_game = Game("ql_eps_greedy_run_avg_lr_decay", get_state, eps_greedy_pcs, ql_pcs_act, get_reward, ql_run_avg_update, a_hist_1step, M_counter, QL_LR_DECAY=True, **ql_params)


ql_eps_greedy_run_avg_lr_correction_game = Game("ql_eps_greedy_run_avg_lr_correction", get_state, eps_greedy_pcs, ql_pcs_act, get_reward, ql_run_avg_update, a_hist_1step, M_counter, QL_LR_CORRECTION=True, **ql_params)


# = MAIN =

def plot_pcs(game, num_steps=10000, **kwargs):
    """Plot pcs over time in the given game."""
    game = game.copy_with_agents(hist_agent("pcs_hist", "pcs"))
    game.run(num_steps)

    fig, axs = plt.subplots(1, 2)

    xs = range(1, len(game.env["pcs_hist"]) + 1)
    game.plot(axs[0], xs, lambda env: map(_coconut.operator.itemgetter((C)), env["pcs_hist"]), label="P(C|C)", **kwargs)
    game.plot(axs[0], xs, lambda env: map(_coconut.operator.itemgetter((D)), env["pcs_hist"]), label="P(C|D)", **kwargs)
    axs[0].set(xlabel="t")
    axs[0].legend()

    log_xs = (list)(map(log, xs))
    game.plot(axs[1], log_xs, lambda env: map(_coconut.operator.itemgetter((C)), env["pcs_hist"]), label="P(C|C)", **kwargs)
    game.plot(axs[1], log_xs, lambda env: map(_coconut.operator.itemgetter((D)), env["pcs_hist"]), label="P(C|D)", **kwargs)
    axs[1].set(xlabel="log(t)")
    axs[1].legend()

    plt.show()


def plot_qs(game, num_steps=10000, **kwargs):
    """Plot qs over time in the given game."""
    game = game.copy_with_agents(hist_agent("qs_hist", "qs"))
    game.run(num_steps)

    fig, axs = plt.subplots(1, 2)

    xs = range(1, len(game.env["qs_hist"]) + 1)
    game.plot(axs[0], xs, lambda env: map(_coconut.operator.itemgetter((C, C)), env["qs_hist"]), label="Q(C|C)", **kwargs)
    game.plot(axs[0], xs, lambda env: map(_coconut.operator.itemgetter((C, D)), env["qs_hist"]), label="Q(D|C)", **kwargs)
    game.plot(axs[0], xs, lambda env: map(_coconut.operator.itemgetter((D, C)), env["qs_hist"]), label="Q(C|D)", **kwargs)
    game.plot(axs[0], xs, lambda env: map(_coconut.operator.itemgetter((D, D)), env["qs_hist"]), label="Q(D|D)", **kwargs)
    axs[0].set(xlabel="t")
    axs[0].legend()

    log_xs = (list)(map(log, xs))
    game.plot(axs[1], log_xs, lambda env: map(_coconut.operator.itemgetter((C, C)), env["qs_hist"]), label="Q(C|C)", **kwargs)
    game.plot(axs[1], log_xs, lambda env: map(_coconut.operator.itemgetter((C, D)), env["qs_hist"]), label="Q(D|C)", **kwargs)
    game.plot(axs[1], log_xs, lambda env: map(_coconut.operator.itemgetter((D, C)), env["qs_hist"]), label="Q(C|D)", **kwargs)
    game.plot(axs[1], log_xs, lambda env: map(_coconut.operator.itemgetter((D, D)), env["qs_hist"]), label="Q(D|D)", **kwargs)
    axs[1].set(xlabel="log(t)")
    axs[1].legend()

    plt.show()


def run_experiment(game, num_iters=500, num_steps=5000, bucket_size=0.01, pc_calc_steps=500):
    """Measure limiting behavior for the given game."""
    game = game.copy_with_agents(hist_agent("a_hist", "a", maxhist=pc_calc_steps))
    buckets = [0] * int(1 / bucket_size)
    print("Running experiment for {_coconut_format_0}...".format(_coconut_format_0=(game.name)))
    for _ in tqdm(range(num_iters)):
        game.run(num_steps, use_tqdm=False)
        prop_coop = sum((a == C for a in game.env["a_hist"])) / pc_calc_steps
        bucket = int(prop_coop // bucket_size)
        if bucket == len(buckets):
            bucket -= 1
        buckets[bucket] += 1
        game.reset()
    for i in range(len(buckets)):
        buckets[i] /= num_iters
    return buckets


@_coconut_mark_as_match
def plot_experiments(*_coconut_match_to_args, **_coconut_match_to_kwargs):
    """Plot cooperation proportions for all the given games."""
    _coconut_match_check = False
    _coconut_FunctionMatchError = _coconut_get_function_match_error()
    games = _coconut_match_to_args[0:]
    _coconut_match_temp_0 = _coconut_match_to_kwargs.pop("linestyles") if "linestyles" in _coconut_match_to_kwargs else (":", "-.", "--", "-")
    _coconut_match_temp_1 = _coconut_match_to_kwargs.pop("alpha") if "alpha" in _coconut_match_to_kwargs else 0.6
    _coconut_match_temp_2 = _coconut_match_to_kwargs.pop("linewidth") if "linewidth" in _coconut_match_to_kwargs else 2.25
    linestyles = _coconut_match_temp_0
    alpha = _coconut_match_temp_1
    linewidth = _coconut_match_temp_2
    kwargs = _coconut_match_to_kwargs
    _coconut_match_check = True
    if not _coconut_match_check:
        _coconut_match_val_repr = _coconut.repr(_coconut_match_to_args)
        _coconut_match_err = _coconut_FunctionMatchError("pattern-matching failed for " '\'match def plot_experiments(*games, linestyles=(":", "-.", "--", "-"), alpha=0.6, linewidth=2.25, **kwargs):\'' " in " + (_coconut_match_val_repr if _coconut.len(_coconut_match_val_repr) <= 500 else _coconut_match_val_repr[:500] + "..."))
        _coconut_match_err.pattern = 'match def plot_experiments(*games, linestyles=(":", "-.", "--", "-"), alpha=0.6, linewidth=2.25, **kwargs):'
        _coconut_match_err.value = _coconut_match_to_args
        raise _coconut_match_err

    experiments = dict(((g.name), (run_experiment(g, **kwargs))) for g in games)
    fig, ax = plt.subplots(1)
    for (name, buckets), ls in zip(experiments.items(), repeat(linestyles)):
        bucket_xs = np.linspace(0, 1, num=len(buckets))
        ax.plot(bucket_xs, buckets, label=name, ls=ls, alpha=alpha, lw=linewidth)
    ax.set(xlabel="equilibrium cooperation probability", ylabel="probability of equilibrium")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run_experiment(pol_grad_decoupled_game)
# plot_pcs(pol_grad_decoupled_game)
# plot_qs(ql_eps_greedy_decay_run_avg_decoupled_lr_decay_correction_game)
# plot_experiments(
#     pol_grad_game,
#     pol_grad_decoupled_game,
#     ql_eps_greedy_true_avg_game,
#     ql_eps_greedy_run_avg_game,
#     ql_boltz_run_avg_game,
#     ql_boltz_true_avg_game,
#     ql_eps_greedy_decay_run_avg_decoupled_lr_decay_correction_game,
#     ql_eps_greedy_decay_run_avg_decoupled_game,
#     ql_eps_greedy_decay_run_avg_game,
#     ql_eps_greedy_decay_run_avg_lr_decay_correction_game,
#     ql_eps_greedy_decay_true_avg_game,
#     ql_eps_greedy_run_avg_lr_decay_correction_game,
#     ql_eps_greedy_run_avg_lr_decay_game,
#     ql_eps_greedy_run_avg_lr_correction_game,
# )
