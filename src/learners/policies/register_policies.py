"""All custom policies must be registered here."""
import inspect

from stable_baselines3.common.policies import BasePolicy, register_policy

from src.learners.policies.custom_policies import *


def register_policies():
    """All custom policies from `custom_policies.py` are traversed and
    registered in StableBaselines3.
    """

    for c_name, c_type in globals().items():
        if (
            inspect.isclass(c_type)
            and issubclass(c_type, BasePolicy)
            and c_type != BasePolicy
        ):
            register_policy(c_name, c_type)
