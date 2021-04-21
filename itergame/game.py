#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x4c3b8f8b

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

from copy import deepcopy

from tqdm import tqdm

from itergame.util import Str
from itergame.util import clean_env
from itergame.agent import Agent
from itergame.agent import init_agent


class Game(_coconut.object):
    """Game class.

    Parameters:
    - _name_ is the name of the game.
    - _agents_ are agents to include in the environment. (name, agent) tuples
        are also allowed.
    - _named_agents_ are names mapped to agents to give those names to in the
        env. _named_agents_ come after _agents_ in an arbitrary order.
    - _independent_update_ controls whether agents are evaluated independently
        or sequentially (defaults to False, i.e. sequentially). When the updates
        are sequential the order of agents passed to Game will be the order in
        which they are evaluated at each step.
    - _default_run_kwargs_ are keyword arguments to use as the defaults in run.
    """
    name = None
    final_step = False
    _sentinel = object()

    @_coconut_mark_as_match
    def __init__(*_coconut_match_to_args, **_coconut_match_to_kwargs):
        _coconut_match_check = False
        _coconut_FunctionMatchError = _coconut_get_function_match_error()
        if (_coconut.sum((_coconut.len(_coconut_match_to_args) > 0, "self" in _coconut_match_to_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 1, "name" in _coconut_match_to_kwargs)) == 1):
            _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("self")
            _coconut_match_temp_1 = _coconut_match_to_args[1] if _coconut.len(_coconut_match_to_args) > 1 else _coconut_match_to_kwargs.pop("name")
            agents = _coconut_match_to_args[2:]
            _coconut_match_temp_2 = _coconut_match_to_kwargs.pop("independent_update") if "independent_update" in _coconut_match_to_kwargs else False
            _coconut_match_temp_3 = _coconut_match_to_kwargs.pop("default_run_kwargs") if "default_run_kwargs" in _coconut_match_to_kwargs else {}
            if _coconut.isinstance(_coconut_match_temp_1, Str):
                self = _coconut_match_temp_0
                name = _coconut_match_temp_1
                independent_update = _coconut_match_temp_2
                default_run_kwargs = _coconut_match_temp_3
                named_agents = _coconut_match_to_kwargs
                _coconut_match_check = True
        if not _coconut_match_check:
            raise _coconut_FunctionMatchError('match def __init__(self, name is Str, *agents, independent_update=False, default_run_kwargs={}, **named_agents):', _coconut_match_to_args)

        self.agents = []
        self.independent_update = independent_update
        self.default_run_kwargs = default_run_kwargs
        self.reset(name, *agents, **named_agents)

    def reset(self, name=None, *agents, **named_agents):
        """Set all default values and start the step counter. If you want to run
        multiple trials with the same game you must explicitly call reset and if
        you are using bbopt agents you must pass a new _name_."""
        self.name = (self.name if name is None else name)
        self.env = {"game": self}
        self.i = 0
        self.set_defaults(self.agents)
        self.add_agents(*agents, **named_agents)
        return self

    def set_defaults(self, agents):
        """Set the defaults for the given agents."""
        for a in agents:
            for k, v in a.get_defaults().items():
                self.env[k] = v

    @_coconut_mark_as_match
    def add_agents(*_coconut_match_to_args, **_coconut_match_to_kwargs):
        """Add the given agents/variables to the game."""
        _coconut_match_check = False
        _coconut_FunctionMatchError = _coconut_get_function_match_error()
        if _coconut.sum((_coconut.len(_coconut_match_to_args) > 0, "self" in _coconut_match_to_kwargs)) == 1:
            _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("self")
            agents = _coconut_match_to_args[1:]
            _coconut_match_temp_1 = _coconut_match_to_kwargs.pop("_set_defaults") if "_set_defaults" in _coconut_match_to_kwargs else True
            self = _coconut_match_temp_0
            _set_defaults = _coconut_match_temp_1
            named_agents = _coconut_match_to_kwargs
            _coconut_match_check = True
        if not _coconut_match_check:
            raise _coconut_FunctionMatchError('match def add_agents(self, *agents, _set_defaults=True, **named_agents):', _coconut_match_to_args)

        new_agents = []
        for a in _coconut.itertools.chain.from_iterable(_coconut_reiterable(_coconut_func() for _coconut_func in (lambda: agents, lambda: named_agents.items()))):
            _coconut_match_to = a
            _coconut_match_check = False
            if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 2):
                name = _coconut_match_to[0]
                actor = _coconut_match_to[1]
                _coconut_match_check = True
            if _coconut_match_check:
                if not callable(actor):
                    a = init_agent(name, actor)
                elif isinstance(actor, Agent):
                    a = actor.clone(name=name)
                else:
                    a = Agent(name, actor)
            assert isinstance(a, Agent), "not isinstance({_coconut_format_0}, Agent)".format(_coconut_format_0=(a))
            new_agents.append(a)
        self.agents += new_agents
        if _set_defaults:
            self.set_defaults(new_agents)
        return self

    def copy(self):
        """Create a deep copy of the game."""
        new_game = Game(self.name, *self.agents, independent_update=self.independent_update, default_run_kwargs=self.default_run_kwargs, _set_defaults=False)
        new_game.i = self.i
        new_game.env = self.env_copy()
        new_game.env["game"] = new_game
        return new_game

    def copy_with_agents(self, *agents, **named_agents):
        """Create a deep copy with new agents."""
        return self.copy().add_agents(*agents, **named_agents)

    def clone(self, *args, **kwargs):
        """Equivalent to .copy().reset(*args, **kwargs)."""
        return self.copy().reset(*args, **kwargs)

    def attach(self, agent, period, name=None):
        """Add an agent to be called at interval _period_."""
        if isinstance(agent, Agent):
            agent = agent.clone(name=name, period=period)
        else:
            agent = Agent(name, agent, period=period)
        self.agents.append(agent)
        return self

    def step(self):
        """Perform one full step of action selection."""
        updating_env = {} if self.independent_update else self.env
        for a in self.agents:
            if self.i % a.period == 0:
                action = a(self.env)
                if a.name is not None:
                    updating_env[a.name] = action
        if self.independent_update:
            self.env.update(updating_env)
        self.i += 1

    def env_copy(self):
        """Get a copy of the environment without the game."""
        new_env = clean_env(self.env)
        for a in self.agents:
            if a.copy_func is not None and a.name in new_env:
                new_env[a.name] = a.copy_func(new_env[a.name])
        return new_env

    def copy_var(self, name, val):
        """Apply all relevant copiers for the given name to val."""
        for a in self.agents:
            if a.name == name and a.copy_func is not None:
                val = a.copy_func(val)
        return val

    @property
    def max_period(self):
        return max((a.period for a in self.agents if a.period < float("inf")))

    def run(self, max_steps=_sentinel, stop_at_equilibrium=_sentinel, use_tqdm=_sentinel, ensure_all_agents_run=_sentinel):
        """Exactly base_run but includes default_run_kwargs."""
        run_kwargs = self.default_run_kwargs.copy()
        if max_steps is not self._sentinel:
            run_kwargs["max_steps"] = max_steps
        if stop_at_equilibrium is not self._sentinel:
            run_kwargs["stop_at_equilibrium"] = stop_at_equilibrium
        if use_tqdm is not self._sentinel:
            run_kwargs["use_tqdm"] = use_tqdm
        if ensure_all_agents_run is not self._sentinel:
            run_kwargs["ensure_all_agents_run"] = ensure_all_agents_run
        return self.base_run(**run_kwargs)

    def base_run(self, max_steps=None, stop_at_equilibrium=False, use_tqdm=True, ensure_all_agents_run=True):
        """Run iterative action selection for _max_steps_ or until equilibrium is reached if _stop_at_equilibrium_."""
        if max_steps is None and not stop_at_equilibrium:
            raise ValueError("run needs either max_steps not None or stop_at_equilibrium True")
        if stop_at_equilibrium:
            prev_env = self.env_copy()
        rng = range(max_steps)
        if use_tqdm:
            rng = (tqdm)(rng)
        for _ in rng if max_steps is not None else count():
            self.step()
            if stop_at_equilibrium and self.i % self.max_period == 0:
                new_env = self.env_copy()
                if new_env == prev_env:
                    break
                prev_env = new_env
        return self.finalize(ensure_all_agents_run=ensure_all_agents_run)

    def finalize(self, ensure_all_agents_run=True):
        """Gather final parameters, running every agent again if _ensure_all_agents_run_."""
        self.final_step = True
        try:
            if ensure_all_agents_run:
                for _ in range(self.max_period):
                    self.step()
            return self.env_copy()
        finally:
            self.final_step = False

    def plot(self, ax, xs, ys, xlabel=None, ylabel=None, label=None, alpha=0.6, **kwargs):
        """Plot _xs_ vs. _ys_ on the given axis with automatic or custom
        label names and _kwargs_ passed to plot. One of _xs_ or _ys_ may
        be None to replace with a sequence and must otherwise be a
        variable name, list, or function of the env."""
        if xs is None and ys is None:
            raise ValueError("both of xs and ys cannot be None")
        if isinstance(xs, Str):
            xs_list = self.env[xs]
        elif callable(xs):
            xs_list = xs(self.env)
        else:
            xs_list = xs
        if isinstance(ys, Str):
            ys_list = self.env[ys]
        elif callable(ys):
            ys_list = ys(self.env)
        else:
            ys_list = ys
        if xs_list is not None:
            xs_list = list(xs_list)
        if ys_list is not None:
            ys_list = list(ys_list)
        xs_list = range(len(ys_list)) if xs_list is None else xs_list
        ys_list = range(len(xs_list)) if ys_list is None else ys_list

        set_kwargs = {}
        xlabel = xs if xlabel is None else xlabel
        if isinstance(xlabel, Str):
            set_kwargs["xlabel"] = xlabel
        ylabel = ys if ylabel is None else ylabel
        if isinstance(ylabel, Str):
            set_kwargs["ylabel"] = ylabel
        if set_kwargs:
            ax.set(**set_kwargs)

        label = (xs if ys is None else ys) if label is None else label
        if isinstance(label, Str):
            kwargs["label"] = label
        ax.plot(xs_list, ys_list, alpha=alpha, **kwargs)
        return ax

_coconut_call_set_names(Game)
