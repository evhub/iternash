#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x6fa65068

# Compiled with Coconut version 1.4.1 [Ernest Scribbler]

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

from pprint import pprint

from bbopt import BlackBoxOptimizer
from bbopt.constants import default_alg

from iternash.util import Str
from iternash.util import printret


no_default = object()
_no_default_passed = object()

class Agent(_coconut.object):
    """Agent class.

    Parameters:
    - _name_ is the key to assign this agent's action in the environment, or None
        for no name.
    - _actor_ is a function from the environment to the agent's action.
    - _default_ is the agent's initial action.
    - _period_ is the period at which to call the agent (default is 1).
    - _debug_ controls whether the agent should print what it's doing.
    """

    def __init__(self, name, actor, default=no_default, period=1, debug=False):
        self.name = name
        self.actor = actor
        self.default = default
        self.period = period
        self.debug = debug

    def __call__(self, env):
        """Call the agent's actor function."""
        try:
            result = self.actor(env)
            if self.debug:
                print("{_coconut_format_0}({_coconut_format_1}) = {_coconut_format_2}".format(_coconut_format_0=(self), _coconut_format_1=(env), _coconut_format_2=(result)))
            return result
        except:
            print("Error calculating action for {_coconut_format_0}({_coconut_format_1}):".format(_coconut_format_0=(self), _coconut_format_1=(env)))
            raise

    def __repr__(self):
        return "Agent({_coconut_format_0})".format(_coconut_format_0=(self.name))

    def has_default(self):
        """Whether the agent has a default."""
        return self.default is not no_default

    def clone(self, name=None, actor=None, default=_no_default_passed, period=None):
        """Create a copy of the agent (optionally) with new parameters."""
        if default is _no_default_passed:
            default = self.default
        return Agent((self.name if name is None else name), (self.actor if actor is None else actor), default, (self.period if period is None else period))


def agent(name_or_agent_func=None, **kwargs):
    """Decorator for easily constructing agents.

    If a string is passed to the decorator it will use that as the name,
    otherwise the name is inferred from the name of the function.

    Examples:

        @agent()  # or just @agent
        def x(env) =
            ...

        @agent("x")
        def x_agent(env) =
            ...

        @agent(name="x", default=...)
        def x_agent(env) =
            ...
    """
    if name_or_agent_func is None:
        return _coconut.functools.partial(agent, **kwargs)
    elif isinstance(name_or_agent_func, Str) or name_or_agent_func is None:
        return _coconut.functools.partial(Agent, name, **kwargs)
    elif "name" in kwargs:
        return Agent(kwargs.pop("name"), name_or_agent_func, **kwargs)
    else:
        return Agent(name_or_agent_func.__name__, name_or_agent_func, **kwargs)


default_expr_aliases = {"\n": "", "^": "**"}

def expr_agent(name, expr, vars={}, aliases=default_expr_aliases, eval=eval, **kwargs):
    """Construct an agent that computes its action by evaluating an expression.

    Parameters:
    - _name_ is the name the agent's action will be assigned in the environment.
    - _expr_ is an expression to be evaluated in the environment to determine the
        agent's action.
    - _vars_ are the globals to be used for evaluating the agent's action.
    - _aliases_ are simple replacements to be made to the expr before evaluating it
        (the default is {"\\n": "", "^": "**"}).
    - _eval_ is the eval function to use (defaults to Python eval, but can be set to
        coconut.convenience.coconut_eval instead to use Coconut eval).
    - _kwargs_ are passed to `Agent`.
    """
    for k, v in aliases.items():
        expr = expr.replace(k, v)
    return Agent(name, _coconut.functools.partial(eval, expr, vars), **kwargs)


def human_agent(name, vars={}, aliases=default_expr_aliases, **kwargs):
    """Construct an agent that prompts a human for an expression as in expr_agent."""
    def human_actor(env):
        pprint(env.get_clean_env())
        return eval(input("{_coconut_format_0} = ".format(_coconut_format_0=(name))), vars, env)
    return Agent(name, human_actor, **kwargs)


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
    def bbopt_actor(env):
        _coconut_match_to = env
        _coconut_match_check = False
        if _coconut.isinstance(_coconut_match_to, _coconut.abc.Mapping):
            _coconut_match_temp_0 = _coconut_match_to.get((name + "_bb"), _coconut_sentinel)
            if _coconut_match_temp_0 is not _coconut_sentinel:
                bb = _coconut_match_temp_0
                _coconut_match_check = True
        if _coconut_match_check:
            bb.maximize(util_func(env))
        else:
            bb = BlackBoxOptimizer(file=file, tag=env["game"].name + "_" + name)
            env[name + "_bb"] = bb
        bb.run(alg=alg if not env["game"].final_step else None)
        return tunable_actor(bb, env)
    return Agent(name, bbopt_actor, **kwargs)


def debug_agent(debug_str, name=None, **kwargs):
    """Construct an agent that prints a formatted debug string.

    Example:
        debug_agent("x = {x}")
            is roughly equivalent to
        Agent(None, env -> print("x = {x}".format(**env)))
    """
    return Agent(name, lambda env: (printret)(debug_str.format(**env)), **kwargs)


def debug_all_agent(**kwargs):
    """Construct an agent that prints the entire env."""
    return debug_agent("{game.env}", **kwargs)


def initializer_agent(name, constant):
    """Construct an agent that just initializes name to the given constant."""
    return Agent(name, lambda env: constant, default=constant, period=float("inf"))
