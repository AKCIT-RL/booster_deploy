"""Backward-compatible robot configuration imports.

Use ``robots.t1``, ``robots.t2`` or ``robots.k1`` for new code.
"""

from .k1 import K1_CFG
from .t1 import T1_23DOF_CFG
from .t2 import T2_31DOF_CFG

__all__ = ["K1_CFG", "T1_23DOF_CFG", "T2_31DOF_CFG"]
