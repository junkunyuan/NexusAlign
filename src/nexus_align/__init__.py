# Import submodules to trigger registry registration.
# Must run before any registry.get() calls.
import nexus_align.datasets  # noqa: F401
import nexus_align.models  # noqa: F401
import nexus_align.reward_models  # noqa: F401
import nexus_align.algorithms  # noqa: F401
import nexus_align.pipelines  # noqa: F401

from nexus_align.launcher import launch  # noqa: F401
