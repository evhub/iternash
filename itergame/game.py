#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x9abc6469

# Compiled with Coconut version 2.0.0-a_dev9 [How Not to Be Seen]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os as _coconut_os
_coconut_file_dir = _coconut_os.path.dirname(_coconut_os.path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os.path.dirname(_coconut_cached_module.__file__) != _coconut_file_dir:
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
from __coconut__ import _coconut_call_set_names, _coconut_handle_cls_kwargs, _coconut_handle_cls_stargs, _coconut, _coconut_MatchError, _coconut_iter_getitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match, _coconut_reiterable, _coconut_self_match_types, _coconut_dict_merge, _coconut_exec, _coconut_comma_op
_coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

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
    def __init__(*_coconut_match_args, **_coconut_match_kwargs):
        _coconut_match_check_0 = False
        _coconut_match_set_name_self = _coconut_sentinel
        _coconut_match_set_name_name = _coconut_sentinel
        _coconut_match_set_name_agents = _coconut_sentinel
        _coconut_match_set_name_independent_update = _coconut_sentinel
        _coconut_match_set_name_default_run_kwargs = _coconut_sentinel
        _coconut_match_set_name_named_agents = _coconut_sentinel
        _coconut_FunctionMatchError = _coconut_get_function_match_error()
        if (_coconut.sum((_coconut.len(_coconut_match_args) > 0, "self" in _coconut_match_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_args) > 1, "name" in _coconut_match_kwargs)) == 1):
            _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("self")
            _coconut_match_temp_1 = _coconut_match_args[1] if _coconut.len(_coconut_match_args) > 1 else _coconut_match_kwargs.pop("name")
            _coconut_match_set_name_agents = _coconut_match_args[2:]
            _coconut_match_temp_2 = _coconut_match_kwargs.pop("independent_update") if "independent_update" in _coconut_match_kwargs else False
            _coconut_match_temp_3 = _coconut_match_kwargs.pop("default_run_kwargs") if "default_run_kwargs" in _coconut_match_kwargs else {}
            if (isinstance)(_coconut_match_temp_1, Str):
                _coconut_match_set_name_self = _coconut_match_temp_0
                _coconut_match_set_name_name = _coconut_match_temp_1
                _coconut_match_set_name_independent_update = _coconut_match_temp_2
                _coconut_match_set_name_default_run_kwargs = _coconut_match_temp_3
                _coconut_match_set_name_named_agents = _coconut_match_kwargs
                _coconut_match_check_0 = True
        if _coconut_match_check_0:
            if _coconut_match_set_name_self is not _coconut_sentinel:
                self = _coconut_match_temp_0
            if _coconut_match_set_name_name is not _coconut_sentinel:
                name = _coconut_match_temp_1
            if _coconut_match_set_name_agents is not _coconut_sentinel:
                agents = _coconut_match_args[2:]
            if _coconut_match_set_name_independent_update is not _coconut_sentinel:
                independent_update = _coconut_match_temp_2
            if _coconut_match_set_name_default_run_kwargs is not _coconut_sentinel:
                default_run_kwargs = _coconut_match_temp_3
            if _coconut_match_set_name_named_agents is not _coconut_sentinel:
                named_agents = _coconut_match_kwargs
        if not _coconut_match_check_0:
            raise _coconut_FunctionMatchError('match def __init__(self, name `isinstance` Str, *agents, independent_update=False, default_run_kwargs={}, **named_agents):', _coconut_match_args)

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
    def add_agents(*_coconut_match_args, **_coconut_match_kwargs):
        """Add the given agents/variables to the game."""
        _coconut_match_check_2 = False
        _coconut_match_set_name_self = _coconut_sentinel
        _coconut_match_set_name_agents = _coconut_sentinel
        _coconut_match_set_name__set_defaults = _coconut_sentinel
        _coconut_match_set_name_named_agents = _coconut_sentinel
        _coconut_FunctionMatchError = _coconut_get_function_match_error()
        if _coconut.sum((_coconut.len(_coconut_match_args) > 0, "self" in _coconut_match_kwargs)) == 1:
            _coconut_match_temp_0 = _coconut_match_args[0] if _coconut.len(_coconut_match_args) > 0 else _coconut_match_kwargs.pop("self")
            _coconut_match_set_name_agents = _coconut_match_args[1:]
            _coconut_match_temp_1 = _coconut_match_kwargs.pop("_set_defaults") if "_set_defaults" in _coconut_match_kwargs else True
            _coconut_match_set_name_self = _coconut_match_temp_0
            _coconut_match_set_name__set_defaults = _coconut_match_temp_1
            _coconut_match_set_name_named_agents = _coconut_match_kwargs
            _coconut_match_check_2 = True
        if _coconut_match_check_2:
            if _coconut_match_set_name_self is not _coconut_sentinel:
                self = _coconut_match_temp_0
            if _coconut_match_set_name_agents is not _coconut_sentinel:
                agents = _coconut_match_args[1:]
            if _coconut_match_set_name__set_defaults is not _coconut_sentinel:
                _set_defaults = _coconut_match_temp_1
            if _coconut_match_set_name_named_agents is not _coconut_sentinel:
                named_agents = _coconut_match_kwargs
        if not _coconut_match_check_2:
            raise _coconut_FunctionMatchError('match def add_agents(self, *agents, _set_defaults=True, **named_agents):', _coconut_match_args)

        new_agents = []
        for a in _coconut.itertools.chain.from_iterable(_coconut_reiterable(_coconut_func() for _coconut_func in (lambda: agents, lambda: named_agents.items()))):
            _coconut_match_to_0 = a
            _coconut_match_check_1 = False
            _coconut_match_set_name_name = _coconut_sentinel
            _coconut_match_set_name_actor = _coconut_sentinel
            if (_coconut.isinstance(_coconut_match_to_0, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to_0) == 2):
                _coconut_match_set_name_name = _coconut_match_to_0[0]
                _coconut_match_set_name_actor = _coconut_match_to_0[1]
                _coconut_match_check_1 = True
            if _coconut_match_check_1:
                if _coconut_match_set_name_name is not _coconut_sentinel:
                    name = _coconut_match_to_0[0]
                if _coconut_match_set_name_actor is not _coconut_sentinel:
                    actor = _coconut_match_to_0[1]
            if _coconut_match_check_1:
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

    def env_copy(self, env=None):
        """Get a copy of the environment without the game."""
        new_env = (clean_env(self.env) if env is None else env)
        for a in self.agents:
            for copy_name, copy_func in a.copiers.items():
                _coconut_match_to_1 = new_env
                _coconut_match_check_3 = False
                _coconut_match_set_name_val = _coconut_sentinel
                if _coconut.isinstance(_coconut_match_to_1, _coconut.abc.Mapping):
                    _coconut_match_temp_0 = _coconut_match_to_1.get(copy_name, _coconut_sentinel)
                    if _coconut_match_temp_0 is not _coconut_sentinel:
                        _coconut_match_set_name_val = _coconut_match_temp_0
                        _coconut_match_check_3 = True
                if _coconut_match_check_3:
                    if _coconut_match_set_name_val is not _coconut_sentinel:
                        val = _coconut_match_temp_0
                if _coconut_match_check_3:
                    new_env[copy_name] = copy_func(val)
        return new_env

    def copy_var(self, name, val):
        """Apply all relevant copiers for the given name to val."""
        return self.env_copy({name: val})[name]

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
