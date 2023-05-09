from __future__ import annotations

from gymnasium.envs.registration import register

def register_ant_envs():
    register(
        id="AntDir-v0",
        entry_point="antdirenv.ant:AntDirEnv",
        max_episode_steps=1000,
        reward_threshold=6000.0,
    )
    register(
        id="AntDir-v1",
        entry_point="antdirenv.ant:AntDirEnv",
        max_episode_steps=1000,
        reward_threshold=6000.0,
        kwargs={"floor_reward": True},
    )
