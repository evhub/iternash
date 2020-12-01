Module iternash.examples.self_prisoner_dilemma
==============================================

Functions
---------

    
`coop_with_prob(p)`
:   

    
`get_boltz_pc(env, s, temp=None)`
:   

    
`get_eps_greedy_pc(env, s, eps=None)`
:   

    
`get_prev_a(env)`
:   

    
`plot_M(game, num_steps=10000, axs=None, **kwargs)`
:   

    
`plot_experiments(results, linestyles=(':', '-.', '--', '-'), alpha=0.6, linewidth=2.25, **kwargs)`
:   Plot cooperation proportions for all the given games.

    
`plot_pcs(game, num_steps=10000, axs=None, **kwargs)`
:   Plot pcs over time in the given game.

    
`plot_qs(game, num_steps=10000, axs=None, **kwargs)`
:   Plot qs over time in the given game.

    
`plot_qs_pcs_M(game, num_steps=10000, **kwargs)`
:   Plot qs, pcs, and M together.

    
`run_experiment(game, num_iters=500, num_steps=5000, bucket_size=0.01, pc_calc_steps=500)`
:   Measure limiting behavior for the given game.

    
`run_experiments(*games, **kwargs)`
:   Runs multiple experiments and collects the results.

    
`show_expected_coop_props(results)`
:   Print the expected proportion of cooperations for the given games.

    
`show_percent_in_coop_eq(results)`
:   Print the probability of ending up in a cooperative equilibrium.