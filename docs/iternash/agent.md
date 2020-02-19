Module iternash.agent
=====================

Functions
---------

    
`agent(name_or_agent_func=None, **kwargs)`
:   Decorator for easily constructing agents.
    
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

    
`bbopt_agent(name, tunable_actor, util_func, file, alg='tree_structured_parzen_estimator', **kwargs)`
:   Construct an agent that selects its action using a black box optimizer.
    
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

    
`debug_agent(debug_str, name=None, **kwargs)`
:   Construct an agent that prints a formatted debug string.
    
    Example:
        debug_agent("x = {x}")
            is roughly equivalent to
        Agent(None, env -> print("x = {x}".format(**env)))

    
`debug_all_agent(**kwargs)`
:   Construct an agent that prints the entire env.

    
`expr_agent(name, expr, vars={}, aliases={'\n': '', '^': '**'}, eval=<built-in function eval>, **kwargs)`
:   Construct an agent that computes its action by evaluating an expression.
    
    Parameters:
    - _name_ is the name the agent's action will be assigned in the environment.
    - _expr_ is an expression to be evaluated in the environment to determine the
        agent's action.
    - _vars_ are the globals to be used for evaluating the agent's action.
    - _aliases_ are simple replacements to be made to the expr before evaluating it
        (the default is {"\n": "", "^": "**"}).
    - _eval_ is the eval function to use (defaults to Python eval, but can be set to
        coconut.convenience.coconut_eval instead to use Coconut eval).
    - _kwargs_ are passed to `Agent`.

    
`human_agent(name, vars={}, aliases={'\n': '', '^': '**'}, **kwargs)`
:   Construct an agent that prompts a human for an expression as in expr_agent.

    
`initializer_agent(name, constant)`
:   Construct an agent that just initializes name to the given constant.

Classes
-------

`Agent(name, actor, default=<object object at 0x0000023305C66C10>, period=1, debug=False)`
:   Agent class.
    
    Parameters:
    - _name_ is the key to assign this agent's action in the environment, or None
        for no name.
    - _actor_ is a function from the environment to the agent's action.
    - _default_ is the agent's initial action.
    - _period_ is the period at which to call the agent (default is 1).
    - _debug_ controls whether the agent should print what it's doing.

    ### Methods

    `clone(self, name=None, actor=None, default=<object object at 0x0000023305C66C20>, period=None)`
    :   Create a copy of the agent (optionally) with new parameters.

    `has_default(self)`
    :   Whether the agent has a default.