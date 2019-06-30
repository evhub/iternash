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
        env. _named_agents come after agents in an arbitrary order.
    - _independent_update_ controls whether agents are evaluated independently
        or sequentially (defaults to False, i.e. sequentially). When the updates
        are sequential the order of agents passed to Game will be the order in
        which they are evaluated at each step.

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

    `finalize(self)`
    :   Gather final parameters.

    `run(self, n=None)`
    :   Run _n_ steps of iterative action selection.

    `step(self)`
    :   Perform one full step of action selection.