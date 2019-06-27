#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xa7934180

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

import math

from bbopt import BlackBoxOptimizer
from bbopt.constants import default_alg


no_default = object()
_no_default_passed = object()

class Agent(_coconut.object):
    """Agent class.

    Parameters:
    - _name_ is the key to assign this agent's action in the environment.
    - _actor_ is a function from the environment to the agent's action.
    - _default_ is the agent's initial action.
    - _debug_ controls whether actions should be printed.
    """

    def __init__(*_coconut_match_to_args, **_coconut_match_to_kwargs):
        _coconut_match_check = False
        _coconut_FunctionMatchError = _coconut_get_function_match_error()
        if (_coconut.len(_coconut_match_to_args) <= 5) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 0, "self" in _coconut_match_to_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 1, "name" in _coconut_match_to_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 2, "actor" in _coconut_match_to_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 3, "default" in _coconut_match_to_kwargs)) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 4, "debug" in _coconut_match_to_kwargs)) <= 1):
            _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("self")
            _coconut_match_temp_1 = _coconut_match_to_args[1] if _coconut.len(_coconut_match_to_args) > 1 else _coconut_match_to_kwargs.pop("name")
            _coconut_match_temp_2 = _coconut_match_to_args[2] if _coconut.len(_coconut_match_to_args) > 2 else _coconut_match_to_kwargs.pop("actor")
            _coconut_match_temp_3 = _coconut_match_to_args[3] if _coconut.len(_coconut_match_to_args) > 3 else _coconut_match_to_kwargs.pop("default") if "default" in _coconut_match_to_kwargs else no_default
            _coconut_match_temp_4 = _coconut_match_to_args[4] if _coconut.len(_coconut_match_to_args) > 4 else _coconut_match_to_kwargs.pop("debug") if "debug" in _coconut_match_to_kwargs else False
            if (_coconut.isinstance(_coconut_match_temp_1, str)) and (not _coconut_match_to_kwargs):
                self = _coconut_match_temp_0
                name = _coconut_match_temp_1
                actor = _coconut_match_temp_2
                default = _coconut_match_temp_3
                debug = _coconut_match_temp_4
                _coconut_match_check = True
        if not _coconut_match_check:
            _coconut_match_val_repr = _coconut.repr(_coconut_match_to_args)
            _coconut_match_err = _coconut_FunctionMatchError("pattern-matching failed for " "'def __init__(self, name is str, actor, default=no_default, debug=False):'" " in " + (_coconut_match_val_repr if _coconut.len(_coconut_match_val_repr) <= 500 else _coconut_match_val_repr[:500] + "..."))
            _coconut_match_err.pattern = 'def __init__(self, name is str, actor, default=no_default, debug=False):'
            _coconut_match_err.value = _coconut_match_to_args
            raise _coconut_match_err

        self.name = name
        self.actor = actor
        self.default = default
        self.debug = debug

    def __call__(self, env):
        """Call the agent's actor function."""
        action = self.actor(env)
        if self.debug:
            print("{_coconut_format_0} = {_coconut_format_1}".format(_coconut_format_0=(self.name), _coconut_format_1=(action)))
        return action

    def has_default(self):
        """Whether the agent has a default."""
        return self.default is not no_default

    def clone(self, name=None, actor=None, default=_no_default_passed, debug=None):
        """Create a copy of the agent (optionally) with new parameters."""
        if default is _no_default_passed:
            default = self.default
        return Agent((new_name if name is None else name), (self.actor if actor is None else actor), default, (self.debug if debug is None else debug))


def agent(name_or_agent_func=None, **kwargs):
    """Decorator for easily constructing agents.

    If a string is passed to the decorator it will use that as the name,
    otherwise the name is inferred from the name of the function.

    Examples:

        @agent  # or @agent()
        def x(env):
            return ...

        @agent("x")
        def x_agent(env):
            return ...
    """
    if name_or_agent_func is None:
        return agent
    elif isinstance(name_or_agent_func, str):
        return _coconut.functools.partial(Agent, name, **kwargs)
    else:
        return Agent(name_or_agent_func.__name__, name_or_agent_func, **kwargs)


default_expr_aliases = {"\n": "", "^": "**"}

def expr_agent(name, expr, globs=vars(math), aliases=default_expr_aliases, **kwargs):
    """Construct an agent that computes its action by evaluating an expression.

    Parameters:
    - _name_ is the name the agent's action will be assigned in the environment.
    - _expr_ is an expression to be evaluated in the environment to determine the
        agent's action.
    - _globs_ are the globals to be used for evaluating the agent's action (the
        default is vars(math)).
    - _aliases_ are simple replacements to be made to the expr before evaluating it
        (the default is "\n" -> "" and "^" -> "**").
    - _kwargs_ are passed to `Agent`.
    """
    for k, v in aliases.items():
        expr = expr.replace(k, v)
    return Agent(name, _coconut.functools.partial(eval, expr, globs), **kwargs)


def bbopt_agent(name, tunable_actor, util_func, file, alg=default_alg, **kwargs):
    """Construct an agent that selects its action using a black box optimizer.

    Parameters:
    - _name_ is the name the agent's action will be assigned in the environment.
    - _tunable_actor_ is a function from (bb, env) to an action (see the BBopt docs
        for how to use the bb object to define tunable parameters).
    - _util_func_ is the a function from the env resulting from the agent's action
        to the utility it should get for that action.
    - _file_ should be set to __file__.
    - _alg_ determines the black box optimization algorithm to use (the default
        is tree_structured_parzen_estimator).
    - _kwargs_ are passed to `Agent`.
    """
    bb = BlackBoxOptimizer(file=file, tag=name)
    first_action = [True]
    def bbopt_actor(env):
        if first_action[0]:
            first_action[0] = False
        else:
            bb.maximize(util_func(env))
        bb.run(alg=alg if not env["final_step"] else None)
        return tunable_actor(bb, env)
    return Agent(name, bbopt_actor, **kwargs)
