import gym
import numpy as np
from gym import spaces

from xiangqi_zero.env.common import *

# Soldier, Cannon, Chariot(Rook), Horse, Elephant, Advisor, General
id2piece = {
    -16: 'R1',              -5: 'S1', 5: 's1',                 16: 'r1',
    -15: 'H1',  -7: 'C1',                           7: 'c1',   15: 'h1',
    -14: 'E1',              -4: 'S2', 4: 's2',                 14: 'e1',
    -13: 'A1',                                                 13: 'a1',
    -12: 'G',               -3: 'S3', 3: 's3',                 12: 'g',
    -11: 'A2',                                                 11: 'a2',
    -10: 'E2',              -2: 'S4', 2: 's4',                 10: 'e2',
    -9:  'H2',  -6: 'C2',                           6: 'c2',   9:  'h2',
    -8:  'R2',              -1: 'S5', 1: 's5',                 8:  'r2',
    0:   '.'
}


class ChineseChessEnv(gym.Env):

    def __init__(self, df):
        super(ChineseChessEnv, self).__init__()
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Box(-16, 16, shape=(10, 9), dtype=np.int)  # board 8x8

    def step(self, action):
        # Execute one time step within the environment
        ...

    def reset(self):
        # Reset the state of the environment to an initial state
        board = list()
        board.append(['R1', '.', '.', 'S1', '.', '.', 's1', '.', '.', 'r1'])
        board.append(['H1', '.', 'C1', '.', '.', '.', '.', 'c1', '.', 'h1'])
        board.append(['E1', '.', '.', 'S2', '.', '.', 's2', '.', '.', 'e1'])
        board.append(['A1', '.', '.', '.', '.', '.', '.', '.', '.', 'a1'])
        board.append(['G',  '.', '.', 'S3', '.', '.', 's3', '.', '.', 'g'])
        board.append(['A2', '.', '.', '.', '.', '.', '.', '.', '.', 'a2'])
        board.append(['E2', '.', '.', 'S4', '.', '.', 's4', '.', '.', 'e2'])
        board.append(['H2', '.', 'C2', '.', '.', '.', '.', 'c2', '.', 'h2'])
        board.append(['R2', '.', '.', 'S5', '.', '.', 's5', '.', '.', 'r2'])

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...