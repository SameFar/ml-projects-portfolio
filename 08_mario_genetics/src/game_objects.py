import pygame
from pathlib import Path
import numpy as np

from . import config

# Import assets
assets_dir = Path(__file__).resolve().parent.parent / "assets"
logo = pygame.image.load(assets_dir / "logo.png")
pygame.display.set_icon(logo)
BG = pygame.image.load(assets_dir / "background.png")
BG = pygame.transform.scale(BG, (config.WIDTH, config.HEIGHT))
tile = pygame.transform.scale(
    pygame.image.load(assets_dir / "tile.png"), (config.TILE_WIDTH, config.TILE_HEIGHT)
)
bowser = pygame.transform.scale(
    pygame.image.load(assets_dir / "bowser.png"),
    (config.TILE_WIDTH, config.TILE_HEIGHT),
)
notbowser = pygame.transform.scale(
    pygame.image.load(assets_dir / "goal.png"), (config.TILE_WIDTH, config.TILE_HEIGHT)
)
img = pygame.transform.scale(
    pygame.image.load(assets_dir / "mario_standing.png"),
    (config.PLAYER_WIDTH, config.PLAYER_HEIGHT),
)
img2 = pygame.transform.scale(
    pygame.image.load(assets_dir / "mario_jumping.png"),
    (config.PLAYER_WIDTH, config.PLAYER_HEIGHT),
)


class Obsticle(pygame.Rect):
    def __init__(self, x, y):
        pygame.Rect.__init__(self, x, y, config.TILE_WIDTH, config.TILE_HEIGHT)
        self.image = tile


class Goal(pygame.Rect):
    def __init__(self, x, y):
        pygame.Rect.__init__(self, x, y, config.TILE_WIDTH, config.TILE_HEIGHT)
        self.image = bowser
        self.check = False

    def reached(self):
        if not self.check:
            self.image = notbowser
            self.check = True


class Player(pygame.Rect):
    def __init__(self, dna=None):
        pygame.Rect.__init__(
            self, config.X, config.Y, config.PLAYER_WIDTH, config.PLAYER_HEIGHT
        )
        self.image = img
        self.image2 = img2
        self.velocity = 0
        self.speed = config.WALK_SPEED
        self.jumping = False
        self.time = config.CHECK_TIME

        if dna is None:
            self.dna = np.random.randint(0, 2, size=(config.GENES, 3))
        else:
            self.dna = dna

        self.bonus = 0
        self.alive = 1
        self.distance = 1
        self.checks = set()
        self.previous_x = config.X
        self.stagnant_count = 0

    def image_jump(self):
        if self.jumping:
            self.image = img2
        else:
            self.image = img

    def fitness(self):
        penalty = 5 if self.alive == 0 else 0
        score = (
            (2000 / (self.distance)) + (len(self.checks) * 300) - (penalty) + self.bonus
        )
        return max(0.01, score)
