#!/usr/bin/env python3
"""Compatibility entry point: plot the current causal day0 levels and events.

The former band-versus-retrospective-cluster comparison has been retired.
Accepts the same --code/--kind/--since/--out/--events flags as plot_zones.py.
"""
from plot_zones import main
if __name__ == '__main__':
    main()
