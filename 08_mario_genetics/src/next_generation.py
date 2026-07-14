import random
import numpy as np

from . import config
from .game_objects import Player


def natural_selection(population):
    # Sort the population by fitness score in descending order and grab the top X values
    sorted_population = sorted(
        population, key=lambda agent: agent.fitness(), reverse=True
    )
    return sorted_population[: config.TOP_X]


def mariojrs(elites):
    offspring_agents = []

    for _ in range(config.POPULATION):
        mom, dad = random.sample(elites, 2)

        dna1 = mom.dna
        dna2 = dad.dna

        # Choose 4, sorted split frames from the total frames (genes)
        splits = sorted(random.sample(range(1, config.GENES), 4))
        indices = [0] + splits + [config.GENES]

        # Slice both parents DNA into 5 segments
        seg1 = [dna1[indices[i] : indices[i + 1]] for i in range(5)]
        seg2 = [dna2[indices[i] : indices[i + 1]] for i in range(5)]

        # Make a random blueprint pattern of 0s and 1s for the 5 segments
        pattern = [random.choice([0, 1]) for _ in range(5)]

        child_dna_segments = []  # [0,1,1]
        for segment_idx, parent_choice in enumerate(pattern):
            if parent_choice == 0:
                child_dna_segments.append(seg1[segment_idx])
            else:
                child_dna_segments.append(seg2[segment_idx])

        # Merge the 5 stitched segments back together into a single matrix
        child_dna = np.concatenate(child_dna_segments, axis=0)

        # Mutation
        mutation_mask = np.random.rand(*child_dna.shape) < config.MUTATION
        child_dna[mutation_mask] = 1 - child_dna[mutation_mask]

        offspring_agents.append(Player(dna=child_dna))

    return offspring_agents
