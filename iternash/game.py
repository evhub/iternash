#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x9cd093ad

# Compiled with Coconut version 1.4.0-post_dev40 [Ernest Scribbler]

# Coconut Header: -------------------------------------------------------------

from __future__ import print_function, absolute_import, unicode_literals, division
import sys as _coconut_sys, os.path as _coconut_os_path
_coconut_file_path = _coconut_os_path.dirname(_coconut_os_path.abspath(__file__))
_coconut_cached_module = _coconut_sys.modules.get(str("__coconut__"))
if _coconut_cached_module is not None and _coconut_os_path.dirname(_coconut_cached_module.__file__) != _coconut_file_path:
    del _coconut_sys.modules[str("__coconut__")]
_coconut_sys.path.insert(0, _coconut_file_path)
from __coconut__ import *
from __coconut__ import _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_back_pipe, _coconut_star_pipe, _coconut_back_star_pipe, _coconut_dubstar_pipe, _coconut_back_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert
if _coconut_sys.version_info >= (3,):
    _coconut_sys.path.pop(0)

# Compiled Coconut: -----------------------------------------------------------

from copy import deepcopy

from tqdm import tqdm

from iternash.util import Str
from iternash.agent import Agent


class Game(_coconut.object):
    """Game class. See Game.setup for information on __init__ parameters."""
    final_step = False

    def __init__(self, *args, **kwargs):
        self.env = {"game": self}
        self.agents = {}
        self.handlers = []
        self.immediate_update = True
        self.setup(*args, **kwargs)

    def setup(*_coconut_match_to_args, **_coconut_match_to_kwargs):
        """Initialize the game with agents/variables.

        Parameters:
        - _name_ is the name of the game.
        - _agents_ are agents to include in the environment.
        - _named_agents_ are names mapped to agents to give those names to
            in the environment.
        - _immediate_update_ controls whether new actions are added to the env
            immediately or only at the end of each step (defaults to True). When
            this is on the order of agents passed to Game should be the order in
            which they should be evaluated at each step.
        """
        _coconut_match_check = False
        _coconut_FunctionMatchError = _coconut_get_function_match_error()
        if (_coconut.sum((_coconut.len(_coconut_match_to_args) > 0, "self" in _coconut_match_to_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 1, "name" in _coconut_match_to_kwargs)) == 1):
            _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("self")
            _coconut_match_temp_1 = _coconut_match_to_args[1] if _coconut.len(_coconut_match_to_args) > 1 else _coconut_match_to_kwargs.pop("name")
            agents = _coconut_match_to_args[2:]
            _coconut_match_temp_2 = _coconut_match_to_kwargs.pop("immediate_update") if "immediate_update" in _coconut_match_to_kwargs else None
            if _coconut.isinstance(_coconut_match_temp_1, Str):
                self = _coconut_match_temp_0
                name = _coconut_match_temp_1
                immediate_update = _coconut_match_temp_2
                named_agents = _coconut_match_to_kwargs
                _coconut_match_check = True
        if not _coconut_match_check:
            _coconut_match_val_repr = _coconut.repr(_coconut_match_to_args)
            _coconut_match_err = _coconut_FunctionMatchError("pattern-matching failed for " "'match def setup(self, name is Str, *agents, immediate_update=None, **named_agents):'" " in " + (_coconut_match_val_repr if _coconut.len(_coconut_match_val_repr) <= 500 else _coconut_match_val_repr[:500] + "..."))
            _coconut_match_err.pattern = 'match def setup(self, name is Str, *agents, immediate_update=None, **named_agents):'
            _coconut_match_err.value = _coconut_match_to_args
            raise _coconut_match_err

        self.name = name
        if immediate_update is not None:
            self.immediate_update = immediate_update
        self.i = 0
        for a in _coconut.itertools.chain.from_iterable((_coconut_func() for _coconut_func in (lambda: agents, lambda: named_agents.items()))):
            _coconut_match_to = a
            _coconut_match_check = False
            if (_coconut.isinstance(_coconut_match_to, _coconut.abc.Sequence)) and (_coconut.len(_coconut_match_to) == 2):
                name = _coconut_match_to[0]
                actor = _coconut_match_to[1]
                _coconut_match_check = True
            if _coconut_match_check:
                if not callable(actor):
                    assert isinstance(name, Str), "not isinstance({_coconut_format_0}, Str)".format(_coconut_format_0=(name))
                    self.env[name] = actor
                    continue
                elif isinstance(actor, Agent):
                    a = actor.clone(name=name)
                else:
                    a = Agent(name, actor)
            assert isinstance(a, Agent), "not isinstance({_coconut_format_0}, Agent)".format(_coconut_format_0=(a))
            if a.name is not None:
                if a.has_default():
                    self.env[a.name] = a.default
                self.agents[a.name] = a
        self.step()

    def attach(self, handler, period=100):
        """Attach a handler to be called at interval _period_. _handler_ should be
        some function which takes in an env."""
        self.handlers.append((handler, period))

    def call_handlers(self):
        """Call all attached handlers."""
        for handler, period in self.handlers:
            if self.i % period == 0:
                handler(self.env)

    def step(self, final=False):
        """Perform one full step of action selection."""
        if final:
            self.final_step = True
            try:
                return self.step()
            finally:
                self.final_step = False
        else:
            updating_env = self.env if self.immediate_update else {}
            for a in self.agents.values():
                action = a(self.env)
                if a.name is not None:
                    updating_env[a.name] = action
            if not self.immediate_update:
                self.env.update(updating_env)
            self.i += 1
            self.call_handlers()
            return self.env

    def run(self, max_steps=500):
        """Iterate until equilibrium or _max_steps_ is reached."""
        for _ in tqdm(range(max_steps)):
            prev_env = deepcopy(self.env)
            self.step()
            if self.env == prev_env:
                break
        return self.step(final=True)

    def clone(self, name=None, *args, **kwargs):
        """Create a copy of the environment (optionally) with new parameters."""
        new_game = deepcopy(self)
        new_game.setup((self.name if name is None else name), *args, **kwargs)
        return new_game
