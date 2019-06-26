#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0xcefeb736

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

no_default = object()

class Agent(_coconut.object):
    def __init__(*_coconut_match_to_args, **_coconut_match_to_kwargs):
        _coconut_match_check = False
        _coconut_FunctionMatchError = _coconut_get_function_match_error()
        if (_coconut.len(_coconut_match_to_args) <= 4) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 0, "self" in _coconut_match_to_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 1, "name" in _coconut_match_to_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 2, "actor" in _coconut_match_to_kwargs)) == 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 3, "default" in _coconut_match_to_kwargs)) <= 1):
            _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("self")
            _coconut_match_temp_1 = _coconut_match_to_args[1] if _coconut.len(_coconut_match_to_args) > 1 else _coconut_match_to_kwargs.pop("name")
            _coconut_match_temp_2 = _coconut_match_to_args[2] if _coconut.len(_coconut_match_to_args) > 2 else _coconut_match_to_kwargs.pop("actor")
            _coconut_match_temp_3 = _coconut_match_to_args[3] if _coconut.len(_coconut_match_to_args) > 3 else _coconut_match_to_kwargs.pop("default") if "default" in _coconut_match_to_kwargs else no_default
            if (_coconut.isinstance(_coconut_match_temp_1, str)) and (not _coconut_match_to_kwargs):
                self = _coconut_match_temp_0
                name = _coconut_match_temp_1
                actor = _coconut_match_temp_2
                default = _coconut_match_temp_3
                _coconut_match_check = True
        if not _coconut_match_check:
            _coconut_match_val_repr = _coconut.repr(_coconut_match_to_args)
            _coconut_match_err = _coconut_FunctionMatchError("pattern-matching failed for " "'def __init__(self, name is str, actor, default=no_default):'" " in " + (_coconut_match_val_repr if _coconut.len(_coconut_match_val_repr) <= 500 else _coconut_match_val_repr[:500] + "..."))
            _coconut_match_err.pattern = 'def __init__(self, name is str, actor, default=no_default):'
            _coconut_match_err.value = _coconut_match_to_args
            raise _coconut_match_err

        self.name = name
        if isinstance(actor, Agent):
            self.actor = actor.actor
            self.default = default if default is not no_default else actor.default
        else:
            self.actor = actor
            self.default = default

    def __call__(self, env):
        if callable(self.actor):
            return self.actor(env)
        else:
            return self.actor

    def __str__(self):
        return self.name

    def has_default(self):
        return self.default is not no_default


def agent(*_coconut_match_to_args, **_coconut_match_to_kwargs):
    _coconut_match_check = False
    _coconut_FunctionMatchError = _coconut_get_function_match_error()
    if (_coconut.len(_coconut_match_to_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 0, "name" in _coconut_match_to_kwargs)) == 1):
        _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("name")
        if (_coconut.isinstance(_coconut_match_temp_0, str)) and (not _coconut_match_to_kwargs):
            name = _coconut_match_temp_0
            _coconut_match_check = True
    if not _coconut_match_check:
        _coconut_match_val_repr = _coconut.repr(_coconut_match_to_args)
        _coconut_match_err = _coconut_FunctionMatchError("pattern-matching failed for " "'def agent(name is str) ='" " in " + (_coconut_match_val_repr if _coconut.len(_coconut_match_val_repr) <= 500 else _coconut_match_val_repr[:500] + "..."))
        _coconut_match_err.pattern = 'def agent(name is str) ='
        _coconut_match_err.value = _coconut_match_to_args
        raise _coconut_match_err

    return _coconut.functools.partial(Agent, name)

@_coconut_addpattern(agent)
def agent(*_coconut_match_to_args, **_coconut_match_to_kwargs):
    """Decorator for easily constructing Agents."""
    _coconut_match_check = False
    _coconut_FunctionMatchError = _coconut_get_function_match_error()
    if (_coconut.len(_coconut_match_to_args) <= 1) and (_coconut.sum((_coconut.len(_coconut_match_to_args) > 0, "agent_func" in _coconut_match_to_kwargs)) == 1):
        _coconut_match_temp_0 = _coconut_match_to_args[0] if _coconut.len(_coconut_match_to_args) > 0 else _coconut_match_to_kwargs.pop("agent_func")
        if not _coconut_match_to_kwargs:
            agent_func = _coconut_match_temp_0
            _coconut_match_check = True
    if not _coconut_match_check:
        _coconut_match_val_repr = _coconut.repr(_coconut_match_to_args)
        _coconut_match_err = _coconut_FunctionMatchError("pattern-matching failed for " "'addpattern def agent(agent_func) ='" " in " + (_coconut_match_val_repr if _coconut.len(_coconut_match_val_repr) <= 500 else _coconut_match_val_repr[:500] + "..."))
        _coconut_match_err.pattern = 'addpattern def agent(agent_func) ='
        _coconut_match_err.value = _coconut_match_to_args
        raise _coconut_match_err

    return Agent(agent_func_or_name.__name__, agent_func_or_name)


default_expr_aliases = {"\n": "", "^": "**"}

def expr_agent(name, expr, default=no_default, globs=None, aliases=default_expr_aliases):
    """Construct an agent that evaluates the given expression."""
    for k, v in aliases.items():
        expr = expr.replace(k, v)
    return Agent(name, _coconut.functools.partial(eval, expr, globs), default)
