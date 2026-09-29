"""The live demo server.

Video is processed in the browser; this process receives landmark numbers only. See
:mod:`seam.serve.app` for why, and for what the endpoint does and does not claim.
"""

from seam.serve.app import analyse, build_app, measure, parse_window

__all__ = ["analyse", "build_app", "measure", "parse_window"]
