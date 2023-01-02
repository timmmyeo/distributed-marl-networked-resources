from gymnasium.envs.registration import register

register(
    id="custom_environments/MultiDataCenterEnvironment",
    entry_point="custom_environments.envs:MultiDataCenterEnvironment",
)