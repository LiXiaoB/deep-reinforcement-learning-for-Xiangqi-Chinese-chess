from typing import Tuple, Dict, List

import gym
import torch
import numpy as np
from gym import spaces
from pprint import pprint

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
piece2id = {v: k for k, v in id2piece.items()}

DIRECTION_SET = {
    'N',
    'NW',
    'NE',
    'W',
    'E',
    'SE',
    'SW',
    'S'
}

MOVE_SET = {
    'chariot_move',
    'horse_move',
    'elephant_move',
    'advisor_move',
    'general_move',
    'canon_move',
    'soldier_move'
}


class Action:

    def __init__(self, position: Tuple[int, int], type: str, dir: str):
        ...


class ChineseChessEnv(gym.Env):

    def __init__(self):
        super(ChineseChessEnv, self).__init__()
        self.action_space = spaces.Discrete(ACTION_SPACE)
        self.observation_space = spaces.Box(-16, 16, shape=(10, 9), dtype=np.int)  # board 8x8
        self.reset()
        self.state = None
        self.done = False
        # current board
        self.board = []
        self.history = []

    def step(self, action_tensor: torch.tensor):
        # Execute one time step within the environment
        from_pos, to_pos = self.decode_action_from_model(action_tensor)
        if self.validate_move(from_pos, to_pos):
            self._apply_move(from_pos, to_pos)
        else:
            raise AssertionError(f'Invalid move {from_pos} to {to_pos}')

    @staticmethod
    def _out_of_board(pos)->bool:
        vertical = pos[0]
        horizontal = pos[1]
        if vertical < 0 or vertical > 9:
            return True
        if horizontal < 0 or horizontal > 8:
            return True
        return False

    def validate_move(self, from_pos, to_pos)->bool:
        # piece out of the board
        if self._out_of_board(from_pos) or self._out_of_board(to_pos):
            return False
        piece = self.board[from_pos[0]][from_pos[1]]
        if piece in ('R1', 'R2', 'r1', 'r2'):
            pass
        return True

    @staticmethod
    def decode_action_from_model(action_tensor):
        from_pos = (action_tensor[0], action_tensor[1])
        move_id = action_tensor[2]
        to_pos = None

        # chariot move
        if move_id <= 35:
            direction, distance = divmod(move_id, 9)
            distance += 1
            if direction == 0:  # N
                to_pos = (from_pos[0] - distance, from_pos[1])
            elif direction == 1:  # E
                to_pos = (from_pos[0], from_pos[1] + distance)
            elif direction == 2:  # S
                to_pos = (from_pos[0] + distance, from_pos[1])
            elif direction == 3:  # W
                to_pos = (from_pos[0], from_pos[1] - distance)
            else:
                raise ValueError(f'{move_id} cannot be identified as a valid CHARIOT move!')

        # horse move
        elif 35 < move_id <= 43:
            if move_id == 36:  # NE
                to_pos = (from_pos[0] - 2, from_pos[1] + 1)
            elif move_id == 37:  # EN
                to_pos = (from_pos[0] - 1, from_pos[1] + 2)
            elif move_id == 38:  # ES
                to_pos = (from_pos[0] + 1, from_pos[1] + 2)
            elif move_id == 39:  # SE
                to_pos = (from_pos[0] + 2, from_pos[1] + 1)
            elif move_id == 40:  # SW
                to_pos = (from_pos[0] + 2, from_pos[1] - 1)
            elif move_id == 41:  # WS
                to_pos = (from_pos[0] + 1, from_pos[1] - 2)
            elif move_id == 42:  # WN
                to_pos = (from_pos[0] - 1, from_pos[1] - 2)
            elif move_id == 43:  # NW
                to_pos = (from_pos[0] - 2, from_pos[1] - 1)
            else:
                raise ValueError(f'{move_id} cannot be identified as a valid HORSE move!')

        # elephant move
        elif 43 < move_id <= 47:
            if move_id == 44:  # NE
                to_pos = (from_pos[0] - 2, from_pos[1] + 2)
            elif move_id == 45:  # SE
                to_pos = (from_pos[0] + 2, from_pos[1] + 2)
            elif move_id == 46:  # SW
                to_pos = (from_pos[0] + 2, from_pos[1] - 2)
            elif move_id == 47:  # NW
                to_pos = (from_pos[0] - 2, from_pos[1] - 2)
            else:
                raise ValueError(f'{move_id} cannot be identified as a valid ELEPHANT move!')

        # advisor move
        elif 47 < move_id <= 51:
            if move_id == 48:  # NE
                to_pos = (from_pos[0] - 1, from_pos[1] + 1)
            elif move_id == 49:  # SE
                to_pos = (from_pos[0] + 1, from_pos[1] + 1)
            elif move_id == 50:  # SW
                to_pos = (from_pos[0] + 1, from_pos[1] - 1)
            elif move_id == 51:  # NW
                to_pos = (from_pos[0] - 1, from_pos[1] - 1)
            else:
                raise ValueError(f'{move_id} cannot be identified as a valid ADVISOR move!')

        # general move
        elif 51 < move_id <= 55:
            if move_id == 48:  # N
                to_pos = (from_pos[0] - 1, from_pos[1])
            elif move_id == 49:  # E
                to_pos = (from_pos[0], from_pos[1] + 1)
            elif move_id == 50:  # S
                to_pos = (from_pos[0] + 1, from_pos[1])
            elif move_id == 51:  # W
                to_pos = (from_pos[0], from_pos[1] - 1)
            else:
                raise ValueError(f'{move_id} cannot be identified as a valid GENERAL move!')

        # canon fire move
        elif 55 < move_id <= 59:
            # fire!
            if move_id == 56:  # N
                direction = 'N'
            elif move_id == 57:  # E
                direction = 'E'
            elif move_id == 58:  # S
                direction = 'S'
            elif move_id == 59:  # W
                direction = 'W'
            to_pos = self.get_canon_target(from_pos, direction)

        # canon normal move
        elif 59 < move_id <= 95:
            direction, distance = divmod(move_id - 60, 9)
            if direction == 0:  # N
                to_pos = (from_pos[0] - distance, from_pos[1])
            elif direction == 1:  # E
                to_pos = (from_pos[0], from_pos[1] + distance)
            elif direction == 2:  # S
                to_pos = (from_pos[0] + distance, from_pos[1])
            elif direction == 3:  # W
                to_pos = (from_pos[0], from_pos[1] - distance)
            else:
                raise ValueError(f'{move_id} cannot be identified as a valid CANON move!')

        # soldier move
        elif 95 < move_id <= 98:
            if move_id == 96:  # N
                to_pos = (from_pos[0] - 1, from_pos[1])
            elif move_id == 97:  # E
                to_pos = (from_pos[0], from_pos[1] + 1)
            elif move_id == 98:  # W
                to_pos = (from_pos[0], from_pos[1] - 1)
            else:
                raise ValueError(f'{move_id} cannot be identified as a valid SOLDIER move!')
        return from_pos, to_pos

    def get_canon_target(self, canon_pos, direction: str):
        assert direction in {'N', 'E', 'S', 'W'}
        curr_vertical = canon_pos[0]
        curr_horizontal = canon_pos[1]
        piece_in_the_middle = 0

        if direction == 'N':
            while curr_vertical >= 0:
                if self.board[curr_vertical][curr_horizontal] != '.':
                    piece_in_the_middle += 1
                if piece_in_the_middle == 2:
                    return curr_vertical, curr_horizontal
                curr_vertical -= 1

        elif direction == 'E':
            while curr_horizontal <= 8:
                if self.board[curr_vertical][curr_horizontal] != '.':
                    piece_in_the_middle += 1
                if piece_in_the_middle == 2:
                    return curr_vertical, curr_horizontal
                curr_horizontal += 1

        elif direction == 'S':
            while curr_vertical <= 8:
                if self.board[curr_vertical][curr_horizontal] != '.':
                    piece_in_the_middle += 1
                if piece_in_the_middle == 2:
                    return curr_vertical, curr_horizontal
                curr_vertical += 1

        elif direction == 'W':
            while curr_horizontal <= 0:
                if self.board[curr_vertical][curr_horizontal] != '.':
                    piece_in_the_middle += 1
                if piece_in_the_middle == 2:
                    return curr_vertical, curr_horizontal
                curr_horizontal -= 1

        else:
            raise ValueError(f'Invalid direction {direction}')

    def reset(self):
        # Reset the state of the environment to an initial state
        state = dict()
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

        self.board = board
        state['board'] = board
        state['obs'] = [[piece2id[p] for p in row] for row in board]
        return state

    def render(self, mode='human', close=False):
        # Render the environment to the screen
        ...

    def rotate_board(self, state):
        # Convert to opponent's view
        ...

    def get_valid_mask_tensor(self):
        ...

    def _apply_move(self, from_pos: Tuple[int, int], to_pos: Tuple[int, int]):
        piece = self.board[from_pos[0]][from_pos[1]]
        self.board[from_pos[0]][from_pos[1]] = '.'
        self.board[to_pos[0]][to_pos[1]] = piece
        if piece == 'G' or 'g':
            self.done = True

    def _get_valid_action(self, state):
        ...

    def chariot_move(self, action_tensor):
        ...

    def horse_move(self, action_tensor):
        ...

    def elephant_move(self):
        ...

    def advisor_move(self):
        ...

    def general_move(self):
        ...

    def canon_move(self):
        ...

    def soldier_move(self):
        ...


if __name__ == '__main__':
    env = ChineseChessEnv()
    state = env.reset()
    pprint(env.board)
    env.step((0, 0, 17))
    pprint(env.board)
