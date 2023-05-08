from typing import Optional
import numpy as np
from gymnasium import spaces
from gymnasium.envs.mujoco.ant_v4 import AntEnv
from minigrid.core.mission import MissionSpace

DIRS = ['north', 'east', 'south', 'west']
DIR_VECS = np.array([
    [0, 1],
    [1, 0],
    [0, -1],
    [-1, 0],
], dtype=np.float_)

class AntDirEnv(AntEnv):
    def __init__(
        self,
        xml_file="ant.xml",
        ctrl_cost_weight=0.5,
        use_contact_forces=False,
        contact_cost_weight=5e-4,
        healthy_reward=1.0,
        terminate_when_unhealthy=True,
        healthy_z_range=(0.2, 1.0),
        contact_force_range=(-1.0, 1.0),
        reset_noise_scale=0.1,
        exclude_current_positions_from_observation=True,
        **kwargs,
    ):
        super().__init__(
            xml_file=xml_file,
            ctrl_cost_weight=ctrl_cost_weight,
            use_contact_forces=use_contact_forces,
            contact_cost_weight=contact_cost_weight,
            healthy_reward=healthy_reward,
            terminate_when_unhealthy=terminate_when_unhealthy,
            healthy_z_range=healthy_z_range,
            contact_force_range=contact_force_range,
            reset_noise_scale=reset_noise_scale,
            exclude_current_positions_from_observation=exclude_current_positions_from_observation,
            **kwargs
        )

        mission_space = MissionSpace(
            mission_func=self._gen_mission,
            ordered_placeholders=[DIRS]
        )

        ant_obs_space = self.observation_space
        self.observation_space = spaces.Dict({
            'state': ant_obs_space,
            'mission': mission_space,
            "mission_id": spaces.Discrete(4),
        })

    @staticmethod
    def _gen_mission(target: str):
        return f'go {target}'

    def reset(
        self,
        *args,
        seed: Optional[int] = None,
        options: Optional[dict] = None,
    ):
        ob, infos = super().reset(*args, seed=seed, options=options)
        task_id = None if options is None else options.get('task_id', None)
        if task_id is None:
            self.goal_id = np.random.choice(len(DIRS))
        else:
            self.goal_id = task_id % len(DIRS)
        self.goal = self._gen_mission(DIRS[self.goal_id])
        self.tags = [DIRS[self.goal_id]]
        self.goal_vec = DIR_VECS[self.goal_id]
        infos['tags'] = self.tags
        return {
            'state': ob,
            'mission': self.goal,
            'mission_id': self.goal_id,
        }, infos

    def _reward(self, xy_velocity):
        dir_reward = xy_velocity @ self.goal_vec
        healthy_reward = self.healthy_reward
        infos = {
            'reward_forward': dir_reward,
            "reward_survive": healthy_reward,
        }
        return dir_reward + healthy_reward, infos

    def step(self, action):
        xy_position_before = self.get_body_com("torso")[:2].copy()
        self.do_simulation(action, self.frame_skip)
        xy_position_after = self.get_body_com("torso")[:2].copy()

        xy_velocity = (xy_position_after - xy_position_before) / self.dt
        x_velocity, y_velocity = xy_velocity

        rewards, info = self._reward(xy_velocity)

        costs = ctrl_cost = self.control_cost(action)

        terminated = self.terminated
        observation = self._get_obs()
        info.update({
            "reward_ctrl": -ctrl_cost,
            "x_position": xy_position_after[0],
            "y_position": xy_position_after[1],
            "distance_from_origin": np.linalg.norm(xy_position_after, ord=2),
            "x_velocity": x_velocity,
            "y_velocity": y_velocity,
            "tags": self.tags
        })
        if self._use_contact_forces:
            contact_cost = self.contact_cost
            costs += contact_cost
            info["reward_ctrl"] = -contact_cost

        reward = rewards - costs

        if self.render_mode == "human":
            self.render()

        ob = {
            'state': observation,
            'mission': self.goal,
            'mission_id': self.goal_id,
        }

        return ob, reward, terminated, False, info
