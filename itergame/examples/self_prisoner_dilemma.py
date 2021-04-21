#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xc0f4c5df

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

from math import log
from math import exp
from math import ceil

from itergame.game import Game
from itergame.agent import agent
from itergame.agent import hist_agent
from itergame.util import repeat

import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm


# = GENERIC UTILS =

C = 0
D = 1

floatarr = _coconut.functools.partial(np.array, dtype=float)

NO_REWARD = np.zeros((2, 2), dtype=float)

COOP_PENALTY = (floatarr)([[-1, 0], [-1, 0],])
DEFECT_REWARD = COOP_PENALTY + 2

PREV_COOP_REWARD = (floatarr)([[1, 1], [0, 0],])
PREV_DEFECT_PENALTY = PREV_COOP_REWARD - 2

SELF_PD_DEL_C = COOP_PENALTY, 3 * PREV_COOP_REWARD
PAY_FORWARD_DEL_C = COOP_PENALTY, 2 * PREV_COOP_REWARD
COOKIE_DEL_C = DEFECT_REWARD, 2 * PREV_DEFECT_PENALTY
BUTTON_DEL_C = NO_REWARD, PREV_COOP_REWARD


def coop_with_prob(p):
    return np.random.binomial(1, 1 - p)


common_params = dict(INIT_C_PROB=0.5, DEL_C=SELF_PD_DEL_C, USE_STATE=False)


a_hist_1step = hist_agent("a_hist_1step", "a", maxhist=1)


def get_prev_a(env):
    if not env["a_hist_1step"]:
        env["a_hist_1step"].append(coop_with_prob(env["INIT_C_PROB"]))
    return env["a_hist_1step"][-1]


@agent(name="r")
def get_corrupted_feedback(env, s=None, a=None, k=None):
    s = get_prev_a(env) if s is None else s
    a = env["a"] if a is None else a
    k = a if k is None else k
    DEL, C = env["DEL_C"]
    feedback = DEL[s][k]
    corruption = C[s][a]
    return feedback + corruption


@agent(name="s", default=C)
def get_state(env):
    if env["USE_STATE"]:
        return get_prev_a(env)
    else:
        return C


# = POLICY GRADIENT GAME =

pol_grad_params = common_params.copy()
pol_grad_params.update(POL_GRAD_LR=0.01, CLIP_EPS=0.01)


@agent(name="a")
def pol_grad_act(env):
    return coop_with_prob(env["pcs"][env["s"]])


@agent(name="pcs", default=[np.random.random(), np.random.random()])
def pol_grad_update(env, a=None, r=None):
    a = env["a"] if a is None else a
    r = env["r"] if r is None else r

    lr = env["POL_GRAD_LR"]
    eps = env["CLIP_EPS"]
    th = env["pcs"]
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
    k = pol_grad_act(env)
    r = get_corrupted_feedback(env, k=k)
    return pol_grad_update(env, a=k, r=r)


pol_grad_game = Game("pol_grad", get_state, pol_grad_act, get_corrupted_feedback, pol_grad_update, a_hist_1step, **pol_grad_params)


pol_grad_decoupled_game = Game("pol_grad_decoupled", get_state, pol_grad_act, get_corrupted_feedback, pol_grad_decoupled_update, a_hist_1step, **pol_grad_params)


# = Q LEARNING GAME =

ql_params = common_params.copy()
ql_params.update(EXPLORE_EPS=0.1, BOLTZ_TEMP=1, QL_LR=0.01, QL_LR_DECAY_INIT=0.1)


def get_eps_greedy_pc(env, s, eps=None):
    eps = env["EXPLORE_EPS"] if eps is None else eps
    QC = env["qs"][s][C]
    QD = env["qs"][s][D]

    if QC == QD:
        return 1 / 2
    else:
        prob_coop = eps / 2
        if QC > QD:
            prob_coop += 1 - eps
        return prob_coop


@agent(name="pcs")
def eps_greedy_pcs(env, eps=None):
    return [get_eps_greedy_pc(env, i, eps) for i in range(2)]


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
    return eps_greedy_pcs(env, eps=1 / env["M"][env["s"]])


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
    if env.get("QL_LR_DECAY"):
        al = env["QL_LR_DECAY_INIT"] / env["M"][s]
    else:
        al = env["QL_LR"]
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
    if env.get("QL_LR_DECAY"):
        al_init = env["QL_LR_DECAY_INIT"]
    else:
        al_init = env["QL_LR"]
    s = env["s"]

    NUM_ACTS = 2
    k_eps = max(1 / env["M"][s], al_init * NUM_ACTS)
    prob_k_coop = get_eps_greedy_pc(env, s, eps=k_eps)
    k = coop_with_prob(prob_k_coop)
    r = get_corrupted_feedback(env, k=k)

    al = al_init
    if env.get("QL_LR_DECAY"):
        al /= env["M"][s]
    if env.get("QL_LR_CORRECTION"):
        if k == C:
            prob_k = prob_k_coop
        else:
            prob_k = 1 - prob_k_coop
        al /= prob_k

    env["qs"][s, k] = (1 - al) * env["qs"][s, k] + al * r
    return env["qs"]


ql_eps_greedy_true_avg_game = Game("ql_eps_greedy_true_avg", get_state, eps_greedy_pcs, ql_pcs_act, get_corrupted_feedback, ql_true_avg_update, a_hist_1step, **ql_params)


ql_eps_greedy_run_avg_game = Game("ql_eps_greedy_run_avg", get_state, eps_greedy_pcs, ql_pcs_act, get_corrupted_feedback, ql_run_avg_update, a_hist_1step, **ql_params)


ql_boltz_run_avg_game = Game("ql_boltz_run_avg", get_state, boltz_pcs, ql_pcs_act, get_corrupted_feedback, ql_run_avg_update, a_hist_1step, **ql_params)


ql_boltz_true_avg_game = Game("ql_boltz_true_avg", get_state, boltz_pcs, ql_pcs_act, get_corrupted_feedback, ql_true_avg_update, a_hist_1step, **ql_params)


ql_eps_greedy_decay_run_avg_decoupled_lr_decay_correction_game = Game("ql_eps_greedy_decay_run_avg_decoupled_lr_decay_correction", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_corrupted_feedback, ql_decoupled_update, a_hist_1step, M_counter, QL_LR_DECAY=True, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_decay_run_avg_decoupled_lr_correction_game = Game("ql_eps_greedy_decay_run_avg_decoupled_lr_correction", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_corrupted_feedback, ql_decoupled_update, a_hist_1step, M_counter, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_decay_run_avg_decoupled_game = Game("ql_eps_greedy_decay_run_avg_decoupled", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_corrupted_feedback, ql_decoupled_update, a_hist_1step, M_counter, **ql_params)


ql_eps_greedy_decay_true_avg_game = Game("ql_eps_greedy_decay_true_avg", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_corrupted_feedback, ql_true_avg_update, a_hist_1step, M_counter, **ql_params)


ql_eps_greedy_decay_run_avg_game = Game("ql_eps_greedy_decay_run_avg", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_corrupted_feedback, ql_run_avg_update, a_hist_1step, M_counter, **ql_params)


ql_eps_greedy_decay_run_avg_lr_decay_correction_game = Game("ql_eps_greedy_decay_run_avg_lr_decay_correction", get_state, eps_greedy_decay_pcs, ql_pcs_act, get_corrupted_feedback, ql_run_avg_update, a_hist_1step, M_counter, QL_LR_DECAY=True, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_run_avg_lr_decay_correction_game = Game("ql_eps_greedy_run_avg_lr_decay_correction", get_state, eps_greedy_pcs, ql_pcs_act, get_corrupted_feedback, ql_run_avg_update, a_hist_1step, M_counter, QL_LR_DECAY=True, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_run_avg_lr_decay_game = Game("ql_eps_greedy_run_avg_lr_decay", get_state, eps_greedy_pcs, ql_pcs_act, get_corrupted_feedback, ql_run_avg_update, a_hist_1step, M_counter, QL_LR_DECAY=True, **ql_params)


ql_eps_greedy_run_avg_lr_correction_game = Game("ql_eps_greedy_run_avg_lr_correction", get_state, eps_greedy_pcs, ql_pcs_act, get_corrupted_feedback, ql_run_avg_update, a_hist_1step, M_counter, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_run_avg_decoupled_lr_decay_correction_game = Game("ql_eps_greedy_run_avg_decoupled_lr_decay_correction", get_state, eps_greedy_pcs, ql_pcs_act, get_corrupted_feedback, ql_decoupled_update, a_hist_1step, M_counter, QL_LR_DECAY=True, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_run_avg_decoupled_lr_correction_game = Game("ql_eps_greedy_run_avg_decoupled_lr_correction", get_state, eps_greedy_pcs, ql_pcs_act, get_corrupted_feedback, ql_decoupled_update, a_hist_1step, M_counter, QL_LR_CORRECTION=True, **ql_params)


ql_eps_greedy_run_avg_decoupled_game = Game("ql_eps_greedy_run_avg_decoupled", get_state, eps_greedy_pcs, ql_pcs_act, get_corrupted_feedback, ql_decoupled_update, a_hist_1step, M_counter, **ql_params)


# = MAIN =

def plot_pcs(game, num_steps=10000, axs=None, **kwargs):
    """Plot pcs over time in the given game."""
    if num_steps is not None:
        game = game.copy_with_agents(hist_agent("pcs_hist", "pcs"))
        game.run(num_steps)

    show = axs is None
    if axs is None:
        fig, axs = plt.subplots(1, 2)

    xs = range(1, len(game.env["pcs_hist"]) + 1)
    game.plot(axs[0], xs, lambda env: (map)(_coconut.operator.itemgetter((C)), env["pcs_hist"]), label="P(C|C)", **kwargs)
    game.plot(axs[0], xs, lambda env: (map)(_coconut.operator.itemgetter((D)), env["pcs_hist"]), label="P(C|D)", **kwargs)
    axs[0].set(xlabel="t")
    axs[0].legend()

    log_xs = (list)((map)(log, xs))
    game.plot(axs[1], log_xs, lambda env: (map)(_coconut.operator.itemgetter((C)), env["pcs_hist"]), label="P(C|C)", **kwargs)
    game.plot(axs[1], log_xs, lambda env: (map)(_coconut.operator.itemgetter((D)), env["pcs_hist"]), label="P(C|D)", **kwargs)
    axs[1].set(xlabel="log(t)")
    axs[1].legend()

    if show:
        plt.show()


def plot_qs(game, num_steps=10000, axs=None, **kwargs):
    """Plot qs over time in the given game."""
    if num_steps is not None:
        game = game.copy_with_agents(hist_agent("qs_hist", "qs"))
        game.run(num_steps)

    show = axs is None
    if axs is None:
        fig, axs = plt.subplots(1, 2)

    xs = range(1, len(game.env["qs_hist"]) + 1)
    game.plot(axs[0], xs, lambda env: (map)(_coconut.operator.itemgetter((C, C)), env["qs_hist"]), label="Q(C|C)", **kwargs)
    game.plot(axs[0], xs, lambda env: (map)(_coconut.operator.itemgetter((C, D)), env["qs_hist"]), label="Q(D|C)", **kwargs)
    game.plot(axs[0], xs, lambda env: (map)(_coconut.operator.itemgetter((D, C)), env["qs_hist"]), label="Q(C|D)", **kwargs)
    game.plot(axs[0], xs, lambda env: (map)(_coconut.operator.itemgetter((D, D)), env["qs_hist"]), label="Q(D|D)", **kwargs)
    axs[0].set(xlabel="t")
    axs[0].legend()

    log_xs = (list)((map)(log, xs))
    game.plot(axs[1], log_xs, lambda env: (map)(_coconut.operator.itemgetter((C, C)), env["qs_hist"]), label="Q(C|C)", **kwargs)
    game.plot(axs[1], log_xs, lambda env: (map)(_coconut.operator.itemgetter((C, D)), env["qs_hist"]), label="Q(D|C)", **kwargs)
    game.plot(axs[1], log_xs, lambda env: (map)(_coconut.operator.itemgetter((D, C)), env["qs_hist"]), label="Q(C|D)", **kwargs)
    game.plot(axs[1], log_xs, lambda env: (map)(_coconut.operator.itemgetter((D, D)), env["qs_hist"]), label="Q(D|D)", **kwargs)
    axs[1].set(xlabel="log(t)")
    axs[1].legend()

    if show:
        plt.show()


def plot_M(game, num_steps=10000, axs=None, **kwargs):
    if num_steps is not None:
        game = game.copy_with_agents(hist_agent("M_hist", "M"))
        game.run(num_steps)

    show = axs is None
    if axs is None:
        fig, axs = plt.subplots(1, 2)

    MCs = (list)((starmap)(lambda i, M: M / (i + 2), (enumerate)((map)(_coconut.operator.itemgetter((C)), game.env["M_hist"]))))
    MDs = (list)((starmap)(lambda i, M: M / (i + 2), (enumerate)((map)(_coconut.operator.itemgetter((D)), game.env["M_hist"]))))

    xs = range(1, len(game.env["M_hist"]) + 1)
    game.plot(axs[0], xs, MCs, label="M(C)/M", **kwargs)
    game.plot(axs[0], xs, MDs, label="M(D)/M", **kwargs)
    axs[0].set(xlabel="t")
    axs[0].legend()

    log_xs = (list)((map)(log, xs))
    game.plot(axs[1], log_xs, MCs, label="M(C)/M", **kwargs)
    game.plot(axs[1], log_xs, MDs, label="M(D)/M", **kwargs)
    axs[1].set(xlabel="log(t)")
    axs[1].legend()

    if show:
        plt.show()


def plot_qs_pcs_M(game, num_steps=10000, **kwargs):
    """Plot qs, pcs, and M together."""
    game = game.copy_with_agents(hist_agent("qs_hist", "qs"), hist_agent("pcs_hist", "pcs"), hist_agent("M_hist", "M"))
    game.run(num_steps)

    fig, axs = plt.subplots(3, 2)

    plot_qs(game, num_steps=None, axs=axs[0], **kwargs)
    plot_pcs(game, num_steps=None, axs=axs[1], **kwargs)
    plot_M(game, num_steps=None, axs=axs[2], **kwargs)

    plt.show()


def run_experiment(game, num_iters=500, num_steps=5000, bucket_size=0.01, pc_calc_steps=500):
    """Measure limiting behavior for the given game."""
    game = game.copy_with_agents(hist_agent("a_hist", "a", maxhist=pc_calc_steps))
    buckets = [0] * int(1 / bucket_size)
    coop_props = []
    print("Running experiment for {_coconut_format_0}...".format(_coconut_format_0=(game.name)))
    for _ in tqdm(range(num_iters)):
        game.run(num_steps, use_tqdm=False)
        prop_coop = sum((a == C for a in game.env["a_hist"])) / pc_calc_steps
        coop_props.append(prop_coop)
        bucket = int(prop_coop // bucket_size)
        if bucket == len(buckets):
            bucket -= 1
        buckets[bucket] += 1
        game.reset()
    for i in range(len(buckets)):
        buckets[i] /= num_iters
    return buckets, sum(coop_props) / len(coop_props)


def run_experiments(*games, **kwargs):
    """Runs multiple experiments and collects the results."""
    return dict(((g.name), (run_experiment(g, **kwargs))) for g in games)


def show_expected_coop_props(results):
    """Print the expected proportion of cooperations for the given games."""
    for name, (_, exp_coop_prop) in sorted(results.items(), key=_coconut_base_compose(_coconut.operator.itemgetter((1)), (_coconut.operator.itemgetter((1)), 0))):
        print("E[1/(m-n) sum[i=n -> m] C[i] | {_coconut_format_0}] =\n\t{_coconut_format_1}".format(_coconut_format_0=(name), _coconut_format_1=(exp_coop_prop)))


def show_percent_in_coop_eq(results):
    """Print the probability of ending up in a cooperative equilibrium."""
    for name, (buckets, _) in sorted(results.items(), key=_coconut_base_compose(_coconut.operator.itemgetter((1)), (_coconut.operator.itemgetter((1)), 0))):
        prob_coop_eq = sum(buckets[len(buckets) // 2:]) / sum(buckets)
        print("P(cooperative equilibrium | {_coconut_format_0}] =\n\t{_coconut_format_1}".format(_coconut_format_0=(name), _coconut_format_1=(prob_coop_eq)))


def plot_experiments(results, linestyles=(":", "-.", "--", "-"), alpha=0.6, linewidth=2.25, **kwargs):
    """Plot cooperation proportions for all the given games."""
    fig, ax = plt.subplots(1)
    for (name, buckets), ls in (_coconut_partial(zip, {1: repeat(linestyles)}, 2))((map)(lambda kv: (kv[0], kv[1][0]), results.items())):
        bucket_xs = np.linspace(0, 1, num=len(buckets))
        ax.plot(bucket_xs, buckets, label=name, ls=ls, alpha=alpha, lw=linewidth)
    ax.set(xlabel="equilibrium cooperation probability", ylabel="probability of equilibrium")
    ax.legend()
    plt.show()


if __name__ == "__main__":
    run_experiment(pol_grad_decoupled_game)
# from coconut import embed; embed()
# plot_pcs(pol_grad_decoupled_game)
# plot_qs_pcs_M(ql_eps_greedy_decay_run_avg_decoupled_game)
# results = run_experiments(
#     pol_grad_game,
#     # pol_grad_decoupled_game,
#     ql_eps_greedy_true_avg_game,
#     ql_eps_greedy_run_avg_game,
#     # ql_boltz_run_avg_game,
#     # ql_boltz_true_avg_game,
#     # ql_eps_greedy_decay_run_avg_decoupled_lr_decay_correction_game,
#     # ql_eps_greedy_decay_run_avg_decoupled_lr_correction_game,
#     # ql_eps_greedy_decay_run_avg_decoupled_game,
#     # ql_eps_greedy_decay_run_avg_game,
#     # ql_eps_greedy_decay_run_avg_lr_decay_correction_game,
#     # ql_eps_greedy_decay_true_avg_game,
#     # ql_eps_greedy_run_avg_lr_decay_correction_game,
#     # ql_eps_greedy_run_avg_lr_decay_game,
#     # ql_eps_greedy_run_avg_lr_correction_game,
#     # ql_eps_greedy_run_avg_decoupled_lr_decay_correction_game,
#     # ql_eps_greedy_run_avg_decoupled_lr_correction_game,
#     # ql_eps_greedy_run_avg_decoupled_game,
#     num_iters=100,
# )
# show_expected_coop_props(results)
# show_percent_in_coop_eq(results)
# plot_experiments(results)
