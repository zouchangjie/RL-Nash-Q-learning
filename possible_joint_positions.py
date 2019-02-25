from itertools import permutations

# world height
WORLD_HEIGHT =3

# world width
WORLD_WIDTH = 3

gridIndexList = []
for i in range(0, WORLD_HEIGHT):
    for j in range(0, WORLD_WIDTH):
        gridIndexList.append(WORLD_WIDTH * i + j)