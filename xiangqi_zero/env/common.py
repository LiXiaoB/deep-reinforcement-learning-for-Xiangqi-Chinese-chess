BOARD_WIDTH = 9
BOARD_LENGTH = 10
TYPE_OF_PIECES = 7  # Soldier, Cannon, Chariot, Horse, Elephant, Advisor, General
NUM_MODEL_INPUT = TYPE_OF_PIECES * 2 * 8 + 2  # color + total move_count

# Chariot: 9 * 4
# Horse: 8
# Elephant: 4
# Advisor: 4
# General: 4
# Cannon: (9 * 4 + 4)
# Soldier: 3
# 9 * 4 + 8 + 4 + 4 + 4 + (4 + 9 * 4) + 3 = 99
ACTION_SPACE = BOARD_LENGTH * BOARD_WIDTH * 99  # 8910
