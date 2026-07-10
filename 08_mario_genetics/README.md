# Genetically Superior Mario

A population of little Marios learns to beat a custom platformer level, not through reinforcement learning, but through a genetic algorithm written from scratch in NumPy and Pygame. Every Mario's "brain" is just a fixed sequence of button presses; the ones that get furthest survive, breed, and pass their moves on.

## How it works

The level is a small ASCII grid in `src/config.py` (`X` for solid tiles, `Y` for checkpoints), turned into rectangles at startup.

Each Mario's DNA is a `(600, 3)` array of random bits — one row per frame, for up to 600 frames, where the three bits mean jump / left / right. There's no perception involved: a Mario just plays back its own DNA blindly and either gets somewhere or doesn't.

Every generation, all 1000 Marios in the population run at once. A Mario dies early if it falls off the bottom of the screen, goes 90 frames without making progress, or runs out of time (it starts with a small time budget that gets extended every time it reaches a new checkpoint). Once everyone is dead or the 600 frames run out, each Mario gets a fitness score:

```
score = 2000 / distance_to_next_goal
      + checkpoints_reached * 300
      - 5 if dead else 0
      + 2000 if all checkpoints reached
```

The top ~1.3% of the population (`1000 // 75` = 13 Marios) become the elites. To make the next generation, two random elites are picked as parents, each parent's 600-frame DNA is cut into 5 chronological chunks at random split points, and the child is stitched together by taking each chunk from either mom or dad. Bits are then flipped with some mutation probability.

The base mutation rate is 2%. If the best fitness in the population hasn't improved for 5 generations in a row, the mutation rate jumps to 5% to help escape a local optimum, and drops back down once progress resumes. If fitness stays stagnant for 15 generations, or the run reaches 150 generations, the simulation stops.

## Getting started

Controls, while the simulation is running:

- `F` — toggle headless mode (stop rendering the agents, runs faster)
- `S` — save the current population's DNA to `saved_model/elite_mario_dna.npy`
- `L` — load a previously saved DNA pool back in as the current population

```bash
uv sync
uv run src/main.py
```
