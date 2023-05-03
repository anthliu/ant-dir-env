from setuptools import setup, find_packages

setup(
    name="antdirenv",
    version="0.0.1",
    description='Ant env with direction tasks',
    author='Anthony Liu',
    license='MIT',
    packages=['antdirenv'],
    install_requires=[
        'gymnasium',
        'minigrid',
    ],
    python_requires='>=3.7',
    entry_points={
        "gymnasium.envs": ["__root__ = antdirenv.__init__:register_ant_envs"]
    },
)
