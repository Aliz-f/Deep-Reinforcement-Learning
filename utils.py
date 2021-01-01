import numpy as np


def generate_map(maps):
    maze = np.zeros((len(maps), len(maps)))
    diamonds = []
    bases = []

    for i in range(len(maps)):
        for j in range(len(maps)):
            if maps[i][j] == '.':
                maze[i][j] = 1.
            elif maps[i][j] == '*':
                maze[i][j] = 0.0
            elif maps[i][j].isdigit():
                maze[i][j] = 1.
                diamonds.append((i, j, int(maps[i][j])))
            elif maps[i][j] == "a":
                maze[i][j] = 1.
                bases.append((i, j))

    return maze, diamonds, bases
