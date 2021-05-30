#!/usr/bin/env python
# -*- coding: utf-8 -*-
# __coconut_hash__ = 0x25cea359

# Compiled with Coconut version 1.5.0-post_dev57 [Fish License]

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
                type(_coconut_v).__module__ = _coconut_full_module_name
    _coconut_sys.modules[_coconut_full_module_name] = _coconut__coconut__
from __coconut__ import *
from __coconut__ import _coconut_call_set_names, _coconut, _coconut_MatchError, _coconut_igetitem, _coconut_base_compose, _coconut_forward_compose, _coconut_back_compose, _coconut_forward_star_compose, _coconut_back_star_compose, _coconut_forward_dubstar_compose, _coconut_back_dubstar_compose, _coconut_pipe, _coconut_star_pipe, _coconut_dubstar_pipe, _coconut_back_pipe, _coconut_back_star_pipe, _coconut_back_dubstar_pipe, _coconut_none_pipe, _coconut_none_star_pipe, _coconut_none_dubstar_pipe, _coconut_bool_and, _coconut_bool_or, _coconut_none_coalesce, _coconut_minus, _coconut_map, _coconut_partial, _coconut_get_function_match_error, _coconut_base_pattern_func, _coconut_addpattern, _coconut_sentinel, _coconut_assert, _coconut_mark_as_match, _coconut_reiterable
_coconut_sys.path.pop(0)
# Compiled Coconut: -----------------------------------------------------------

from pprint import pprint
from collections import deque
from copy import deepcopy

from bbopt import BlackBoxOptimizer
from bbopt.constants import meta_opt_alg_var

from itergame.util import Str
from itergame.util import printret
from itergame.util import printerr
from itergame.util import clean_env


class Agent(_coconut.object):
    """Agent class.

    Parameters:
    - _name_ is the key to assign this agent's action in the environment, or None
        for no name.
    - _actor_ is a function from the environment to the agent's action.
    - _default_ is the agent's initial action.
    - _period_ is the period at which to call the agent (default is 1).
    - _extra_defaults_ are extra variables that need to be given defaults.
    - _copy_func_ determines the function used to copy the agent's action (default is deepcopy).
    - _extra_copy_funcs_ are extra copiers for other env vars put there by this agent.
    - _debug_ controls whether the agent should print what it's doing.
    """
    NO_DEFAULT = object()
    _sentinel = object()

    def __init__(self, name, actor, default=NO_DEFAULT, period=1, extra_defaults={}, copy_func=deepcopy, extra_copy_funcs=None, debug=False):
        self.name = name
        self.actor = actor
        self.default = default
        self.period = period
        self.extra_defaults = extra_defaults
        self.debug = debug
        self.copiers = {}
        if copy_func is not None:
            self.copiers[name] = copy_func
        if extra_copy_funcs is not None:
            self.copiers.update(extra_copy_funcs)

    def clone(self, name=None, actor=None, default=_sentinel, period=None, extra_defaults=None, copy_func=_sentinel, extra_copy_funcs=None, debug=None):
        """Create a copy of the agent (optionally) with new parameters."""
        if default is self._sentinel:
            default = deepcopy(self.default)
        if copy_func is self._sentinel:
            copy_func = deepcopy(self.copy_func)
        return Agent((self.name if name is None else name), (deepcopy(self.actor) if actor is None else actor), default, (self.period if period is None else period), (lambda _coconut_x: deepcopy(self.extra_defaults) if _coconut_x is None else _coconut_x)(extra_defaults), copy_func, (lambda _coconut_x: self.extra_copy_funcs if _coconut_x is None else _coconut_x)(extra_copy_funcs), (self.debug if debug is None else debug))

    @property
    def copy_func(self):
        return self.copiers[self.name]

    @property
    def extra_copy_funcs(self):
        extra_copiers = self.copiers.copy()
        del extra_copiers[self.name]
        return extra_copiers

    def __call__(self, env, *args, **kwargs):
        """Call the agent's actor function."""
        try:
            result = self.actor(env, *args, **kwargs)
            if self.debug:
                print("{_coconut_format_0}({_coconut_format_1}, *{_coconut_format_2}, **{_coconut_format_3}) = {_coconut_format_4}".format(_coconut_format_0=(self), _coconut_format_1=(env), _coconut_format_2=(args), _coconut_format_3=(kwargs), _coconut_format_4=(result)))
            return result
        except:
            printerr("Error calculating action for {_coconut_format_0}({_coconut_format_1}, *{_coconut_format_2}, **{_coconut_format_3}):".format(_coconut_format_0=(self), _coconut_format_1=(env), _coconut_format_2=(args), _coconut_format_3=(kwargs)))
            raise

    def __repr__(self):
        return "Agent({_coconut_format_0})".format(_coconut_format_0=(self.name))

    def get_defaults(self):
        """Get a dictionary of all default values to assign."""
        defaults = {}
        if self.default is not self.NO_DEFAULT:
            defaults[self.name] = deepcopy(self.default)
        for name, val in self.extra_defaults.items():
            defaults[name] = deepcopy(val)
        return defaults


_coconut_call_set_names(Agent)
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
    elif isinstance(name_or_agent_func, Str):
        return _coconut.functools.partial(Agent, name_or_agent_func, **kwargs)
    elif "name" in kwargs:
        return Agent(kwargs.pop("name"), name_or_agent_func, **kwargs)
    else:
        return Agent(name_or_agent_func.__name__, name_or_agent_func, **kwargs)


DEFAULT_EXPR_ALIASES = {"\n": ""}

def expr_agent(name, expr, globs=None, aliases=None, eval=eval, **kwargs):
    """Construct an agent that computes its action by evaluating an expression.

    Parameters:
    - _name_ is the name the agent's action will be assigned in the environment.
    - _expr_ is an expression to be evaluated in the environment to determine the
        agent's action.
    - _globs_ are the globals to be used for evaluating the agent's action.
    - _aliases_ are simple replacements to be made to the expr before evaluating it
        (the default is {"\\n": ""}).
    - _eval_ is the eval function to use (defaults to Python eval, but can be set to
        coconut.convenience.coconut_eval instead to use Coconut eval).
    - _kwargs_ are passed to `Agent`.
    """
    if globs is None:
        globs = {}
    if aliases is None:
        aliases = DEFAULT_EXPR_ALIASES
    for k, v in aliases.items():
        expr = expr.replace(k, v)
    return Agent(name, _coconut.functools.partial(eval, expr, globs), **kwargs)


def human_agent(name, pprint=True, globs=None, aliases=None, eval=eval, **kwargs):
    """Construct an agent that prompts a human for an expression as in expr_agent.

    Parameters are as per expr_agent plus _pprint_ which determines whether to
    pretty print the environment for the human."""
    def human_actor(env):
        if pprint:
            pprint(clean_env(env))
        expr = input("{_coconut_format_0} = ".format(_coconut_format_0=(name)))
        return expr_agent(expr, globs, aliases, eval)(env)
    return Agent(name, human_actor, **kwargs)


DEFAULT_BBOPT_ALG = "tree_structured_parzen_estimator"

def bbopt_agent(name, tunable_actor, util_func, file, alg=None, extra_copy_funcs=None, print_chosen_alg=False, **kwargs):
    """Construct an agent that selects its action using a black box optimizer.

    Parameters:
    - _name_ is the name the agent's action will be assigned in the environment.
    - _tunable_actor_ is a function from (bb, env) to an action (see the BBopt docs
        for how to use the bb object to define tunable parameters).
    - _util_func_ is the a function from the env resulting from the agent's action
        to the utility it should get for that action (or just a variable name).
    - _file_ should be set to __file__.
    - _alg_ determines the black box optimization algorithm to use (the default
        is tree_structured_parzen_estimator).
    - _kwargs_ are passed to `Agent`.
    """
    alg = DEFAULT_BBOPT_ALG if alg is None else alg
    extra_copy_funcs = {} if extra_copy_funcs is None else extra_copy_funcs
    bb_name = name + "_bb"
    extra_copy_funcs[bb_name] = lambda bb: None
    def bbopt_actor(env):
        _coconut_match_to_0 = env
        _coconut_match_check_0 = False
        if _coconut.isinstance(_coconut_match_to_0, _coconut.abc.Mapping):
            _coconut_match_temp_0 = _coconut_match_to_0.get(bb_name, _coconut_sentinel)
            if _coconut_match_temp_0 is not _coconut_sentinel:
                bb = _coconut_match_temp_0
                _coconut_match_check_0 = True
        if _coconut_match_check_0 and not (bb is not None):
            _coconut_match_check_0 = False
        if _coconut_match_check_0:
            if isinstance(util_func, Str):
                util = env[util_func]
            else:
                util = util_func(env)
            bb.maximize(util)
        else:
            bb = BlackBoxOptimizer(file=file, tag=env["game"].name + "_" + name)
            env[bb_name] = bb
        bb.run(alg=alg if not env["game"].final_step else None)
        if print_chosen_alg:
            chosen_alg = bb.get_current_run()["values"].get(meta_opt_alg_var)
            if chosen_alg is not None:
                print("\nusing BBopt alg =", chosen_alg)
        return tunable_actor(bb, env)
    return Agent(name, bbopt_actor, extra_copy_funcs=extra_copy_funcs, **kwargs)


def debug_agent(debug_str, name=None, **kwargs):
    """Construct an agent that prints a formatted debug string.

    Example:
        debug_agent("x = {x}")
            is roughly equivalent to
        Agent(None, env -> print("x = {x}".format(**env)))
    """
    return Agent(name, lambda env: (printret)(debug_str.format(**env)), copy_func=None, **kwargs)


def debug_all_agent(pretty=True, **kwargs):
    """Construct an agent that prints the entire env, prettily if _pretty_."""
    print_func = pprint if pretty else print
    return Agent(None, lambda env: print_func(clean_env(env)), copy_func=None, **kwargs)


def init_agent(name, constant):
    """Construct an agent that just initializes name to the given constant."""
    return Agent(name, lambda env: constant, default=constant, period=float("inf"))


def hist_agent(name, record_var, maxhist=None, record_var_copy_func=Agent._sentinel, initializer=(), **kwargs):
    """Construct an agent that records a history.

    Parameters:
    - _name_ is the name of this agent.
    - _record_var_ is the name of the agent to record or a function of env to get the value to record.
    - _maxhist_ is the maximum history to store.
    - _initializer_ is an iterable to fill the initial history with.
    - _kwargs_ are passed to Agent.
    """
    def hist_actor(env):
        copier = record_var_copy_func
        if isinstance(record_var, Str):
            got_val = env[record_var]
            if copier is Agent._sentinel:
                copier = _coconut.functools.partial(env["game"].copy_var, record_var)
        else:
            got_val = record_var(env)
            if copier is Agent._sentinel:
                copier = deepcopy
        if copier is not None:
            got_val = copier(got_val)
        env[name].append(got_val)
        return env[name]
    init_hist = [] if maxhist is None else deque(maxlen=maxhist)
    for x in initializer:
        init_hist.append(x)
    return Agent(name, hist_actor, default=init_hist, **kwargs)


def iterator_agent(name, iterable, extra_defaults=None, extra_copy_funcs=None, **kwargs):
    """Construct an agent that successively produces values from the given
    iterable. Extra arguments are passed to Agent."""
    extra_defaults = {} if extra_defaults is None else extra_defaults
    extra_copy_funcs = {} if extra_copy_funcs is None else extra_copy_funcs
    it_name = name + "_it"
    extra_defaults[it_name] = iterable
    extra_copy_funcs[it_name] = deepcopy
    def iterator_actor(env):
        env[it_name] = (iter)(env[it_name])
        return next(env[it_name])
    return Agent(name, iterator_actor, extra_defaults=extra_defaults, extra_copy_funcs=extra_copy_funcs, **kwargs)
