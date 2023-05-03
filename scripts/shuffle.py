#!/usr/bin/env python3

import numpy as np
import gymnasium as gym

def shuffle(env):
    env.reset()
    T = 0
    while True:
        T += 1
        action = env.action_space.sample()
        observation, reward, done, truncated, infos = env.step(action)
        terminal = done or truncated
        if terminal:
            break
    print(f'T = {T}')

if __name__ == "__main__":
    env = gym.make('AntDir', render_mode='human')

    shuffle(env)
