---

# Genetically Superior Mario

A reinforcement-adjacent machine learning simulation that trains an ensemble of autonomous agents to navigate a custom platformer level using a Genetic Algorithm (GA).

## Overview

This project uses evolutionary computing to optimize player action sequences over successive generations. Agents learn to traverse obstacles and reach target checkpoints through iterative selection and mutation.

The pipeline includes:

* **Custom Environment:** Lightweight tile-based physics and collision engine built in Pygame.
* **Vectorized Genetic Logic:** Genome generation and mutation masks powered by NumPy.
* **Multi-Point Crossover:** Chronological behavioral splicing across parent chromosomes.
* **Stagnation Control:** Dynamic mutation scaling based on population fitness ceilings.
* **Genotype Serialization:** State-decoupled saving and loading via NumPy binaries.

---

## Genetic Pipeline

Key evolutionary steps:

* **Selection:** Top 1.3% of the population (`TOP_X`) selected via elite ranking.
* **Crossover:** Parents are sampled to produce offspring using a 5-segment chronological split.
* **Mutation:** Dynamic bit-flip mutation applied via random distribution masks.

```python
# Adaptive Mutation Scaling
if stagnant_gen >= 5:
    MUTATION = 0.05  # Scale up 250% to break out of local optima

```

This strategy injects behavioral diversity when max fitness stalls, automatically resetting once progress resumes.

---

## Architecture

### Genotype & Physics Separation

The player's AI data (the genome) is strictly isolated from the rendering and physics state (`pygame.Rect`).

* **State Space:** 3-bit binary action array across a fixed frame ceiling (`GENES = 600`).
* **Fitness Metric:** Vector distance calculation combined with checkpoint collection weightings.

$$Score = \frac{2000}{\text{Distance To Goal}} + (\text{Checkpoints} \times 300) - \text{Death Penalty}$$

---

## State Serialization

A lightweight saving mechanism preserves trained models without pickling heavy Pygame objects.

Users can save the current elite DNA pool during execution and reload it to immediately watch optimized runs.

---

## Usage

### Run Simulation / Train Agents

```bash
python main.py

```

### Controls
* `F`: Headless mode, save CPU usage by not rendering in the agents
* `S`: Serialize and save the current elite DNA pool to `elite_mario_dna.npy`.
* `L`: Load saved DNA binary directly into the current active population.

---

## Current Status

* Matrix-based crossover and mutation pipeline implemented
* Adaptive mutation rates functional
* State serialization working via NumPy
* Headless training mode pending integration

---

## Tech Stack

* Python
* NumPy
* Pygame
* Pathlib

---
