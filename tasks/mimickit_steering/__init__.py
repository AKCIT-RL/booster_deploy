"""MimicKit steering task for booster_deploy.

The MimicKit checkout has to be importable before observation.py loads, and
that is earlier than it looks: scripts/deploy.py walks and imports EVERY task
submodule to build its registry, so putting the path setup in the Policy
constructor is too late - `--list` would already have crashed. Hence it happens
here, in the package.

Set MIMICKIT_PATH when the checkout is not a sibling of booster_deploy.

Requires, on top of booster_deploy's own dependencies: a MimicKit checkout and
`gymnasium` (envs.char_env imports it for its action space, though the
observation function itself is plain torch). That is the price of importing the
observation definition instead of copying it into this repo, and it is worth
paying - a copied 200-dim layout goes stale silently, and tests/unit/
test_deploy_observation.py could not then compare against anything.

Neither is needed to register the task or to run `deploy.py --list`:
observation.py imports MimicKit on first use, not at module scope. They are
needed to construct the policy, and their absence reports itself there.
"""

import os
import sys

_DEFAULT_MIMICKIT = os.path.normpath(os.path.join(
    os.path.dirname(__file__), os.pardir, os.pardir, os.pardir, "MimicKit", "mimickit"))

MIMICKIT_PATH = os.path.abspath(os.environ.get("MIMICKIT_PATH", _DEFAULT_MIMICKIT))
if MIMICKIT_PATH not in sys.path:
    sys.path.insert(0, MIMICKIT_PATH)

try:
    from .mimickit_steering import T1MimicKitSteeringControllerCfg
except ImportError as e:
    print("[mimickit_steering] not registered: {}. This task needs a MimicKit "
          "checkout (looked in {}; override with MIMICKIT_PATH) and gymnasium."
          .format(e, MIMICKIT_PATH))
else:
    from booster_deploy.utils.registry import register_task
    register_task("t1_mimickit_steering", T1MimicKitSteeringControllerCfg())
