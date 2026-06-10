# config
WIDTH, HEIGHT = 800, 600
PLAYER_WIDTH, PLAYER_HEIGHT = 24, 32
TILE_WIDTH, TILE_HEIGHT = 30, 30

#starting position
X, Y = 50, 550

WALK_SPEED = 5
VELOCITY = -10
GRAVITY = 0.5

POPULATION = 1000
GENES = 600 # aka total frames
BASE_MUT = 0.02

MUTATION = BASE_MUT
CHECK_TIME = GENES//3
PREV_FITNESS = 0
TOP_X = POPULATION // 75 # best 15% selected

level = [
    "                          ",  
    "                       Y  ",  
    "                       X  ",  
    "                     X    ",  
    "                          ",  
    "                 X        ",  
    "                          ",  
    "                          ",  
    "              XX          ",  
    "         X                ",  
    "      XX                  ",  
    "  XY                      ",  
    "  XX                      ",  
    "                          ",  
    "      XX                  ",  
    "         X                ",  
    "             Y            ",  
    "           X X            ",  
    "      X    X              ",  
    "  XX   XX  X              "   
]
