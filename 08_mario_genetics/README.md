<div align="center">

# 🍄 Genetically Superior Mario

**A population of Marios evolves through a level — no RL, just a genetic algorithm.**

![Machine Learning](https://img.shields.io/badge/🔵_MACHINE_LEARNING-1f6feb?style=for-the-badge)

![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)
![Pygame](https://img.shields.io/badge/Pygame-006400?style=flat-square)

</div>

---

A population of little Marios learns to beat a custom platformer level through a from-scratch genetic algorithm in NumPy and Pygame. Each Mario's "brain" is a fixed sequence of button presses; the ones that get furthest survive, breed, and pass their moves on.

## 🧠 How it works

The level is a small ASCII grid in `src/config.py` (`X` = solid tile, `Y` = checkpoint), turned into rectangles at startup.

Each Mario's DNA is a `(600, 3)` array of random bits — one row per frame (up to 600), the three bits meaning jump / left / right. There's no perception: a Mario just plays back its own DNA blindly.

Every generation, all 1000 Marios run at once. A Mario dies if it falls off the bottom, goes 90 frames without progress, or runs out of time (a small budget extended at each new checkpoint). Once everyone is dead or 600 frames elapse, each gets a fitness score:

```
score = 2000 / distance_to_next_goal
      + checkpoints_reached * 300
      - 5 if dead else 0
      + 2000 if all checkpoints reached
```

The top ~1.3% (`1000 // 75` = 13 Marios) become elites. For the next generation, two random elites are picked as parents, each parent's 600-frame DNA is cut into 5 chronological chunks at random points, and the child takes each chunk from either mom or dad. Bits then flip with some mutation probability.

Base mutation is **2%**. If best fitness stalls for 5 generations, it jumps to **5%** to escape a local optimum, then drops back once progress resumes. The run stops after 15 stagnant generations or 150 total.

## 🎮 Getting started

Controls while running:

| Key | Action |
| --- | --- |
| `F` | toggle headless mode (stop rendering, runs faster) |
| `S` | save the current population's DNA to `saved_model/elite_mario_dna.npy` |
| `L` | load a previously saved DNA pool as the current population |

```bash
uv sync
uv run main.py
```
