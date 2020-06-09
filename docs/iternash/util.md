Module iternash.util
====================

Functions
---------

    
`clean_env(env)`
:   Make a copy of env without game.

    
`clip(x, m=2.220446049250313e-16, M=0.9999999999999998)`
:   Clip x into [m, M] (defaults to [eps, 1-eps]).

    
`printerr(*args)`
:   Print to standard error.

    
`printret(obj)`
:   Print then return _obj_.

    
`real(x)`
:   Get only the real part of x.

    
`safe_log(x)`
:   Safe log allows calling log on floats that could be zero.