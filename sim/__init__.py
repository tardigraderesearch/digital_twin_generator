"""Virtual bench: an end-to-end simulated hardware test for robo-prober + DTG.

A randomized bot1-like arm (hidden ground truth), the real robo-prober
firmware core behind a pseudo terminal, a rendered camera feeding DTG's real
ArUco path, a scripted operator, and scoring against the ground truth.
Run ``python -m sim --help``.
"""
