from game_objects import Obsticle, Goal, BG
import pygame
import config


tiles = []
goals = []
# initialisation
pygame.init()
window = pygame.display.set_mode(
    (config.WIDTH, config.HEIGHT)
)  # set game window pixels
clock = pygame.time.Clock()  # to set framerate
pygame.display.set_caption("Genetically Superior Mario")  # set window title


# Map creation
def create_map():

    for row_idx, row in enumerate(config.level):
        for col_idx, char in enumerate(row):
            if char == "X":
                tiles.append(
                    Obsticle(col_idx * config.TILE_WIDTH, row_idx * config.TILE_HEIGHT)
                )
            if char == "Y":
                goals.append(
                    Goal(col_idx * config.TILE_WIDTH, row_idx * config.TILE_HEIGHT)
                )

    # Making checkpoints start from the bottom instead of top
    goals.reverse()


# Collision logic
def check_tile_collision(mario):
    for t in tiles:
        if mario.colliderect(t):
            return t
    for g in goals:
        if mario.colliderect(g):
            return g
    return None


def check_tile_collision_x(mario):
    col = check_tile_collision(mario)
    if col is not None:
        if isinstance(col, Obsticle):
            if mario.speed < 0:
                mario.x = col.x + col.width
            elif mario.speed > 0:
                mario.x = col.x - mario.width
            mario.speed = 0
        else:
            col.reached()
            if (col.x, col.y) not in mario.checks:
                mario.checks.add((col.x, col.y))
                mario.time += config.CHECK_TIME


def check_tile_collision_y(mario):
    col = check_tile_collision(mario)
    if col is not None:
        if isinstance(col, Obsticle):
            if mario.velocity < 0:
                mario.y = col.y + col.height
            elif mario.velocity > 0:
                mario.y = col.y - mario.height
                mario.jumping = False
            mario.velocity = 0
        else:
            col.reached()
            if (col.x, col.y) not in mario.checks:
                mario.checks.add((col.x, col.y))
                mario.time += config.CHECK_TIME


def draw(headless, agents):
    window.blit(BG, (0, 0))

    for t in tiles:
        window.blit(t.image, t)
    for g in goals:
        window.blit(g.image, g)

    if not headless:
        for mario in agents:
            if mario.alive:
                mario.image_jump()
                window.blit(mario.image, mario)


def move(mario):
    mario.x += mario.speed
    if mario.x < 0:
        mario.x = 0
    elif mario.x + mario.width > config.WIDTH:
        mario.x = config.WIDTH - mario.width
    check_tile_collision_x(mario)

    mario.velocity += config.GRAVITY
    mario.y += mario.velocity
    check_tile_collision_y(mario)
