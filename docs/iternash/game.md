Module iternash.game
====================

Classes
-------

`Game(*_coconut_match_to_args, **_coconut_match_to_kwargs)`
:   Game class.
    
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

    ### Class variables

    `final_step`
    :

    ### Instance variables

    `max_period`
    :

    ### Methods

    `add_agents(self, *agents, **named_agents)`
    :   Add the given agents/variables to the game.

    `attach(self, agent, period, name=None)`
    :   Add an agent to be called at interval _period_.

    `base_run(self, max_steps=None, stop_at_equilibrium=False, ensure_all_agents_run=True)`
    :   Run iterative action selection for _max_steps_ or
        until equilibrium is reached if _stop_at_equilibrium_.

    `env_copy(self)`
    :   Get a copy of the environment without the game.

    `finalize(self, ensure_all_agents_run=True)`
    :   Gather final parameters, running every agent again if _ensure_all_agents_run_.

    `reset(self, name, *agents, **named_agents)`
    :   Set all default values and start the step counter. If you want to call run
        multiple times on the same game you must explicitly call reset and if you are
        using bbopt agents you must pass a new _name_.

    `run(self, max_steps=None, **kwargs)`
    :   Exactly base_run but includes default_run_kwargs.

    `step(self)`
    :   Perform one full step of action selection.