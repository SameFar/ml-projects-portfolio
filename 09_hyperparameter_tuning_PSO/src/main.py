import numpy as np
import pandas as pd
from pathlib import Path
from Particle import Particle

# Configuration
MAX_ITER = 50
POPULATION = 50

MAX_W = 0.9
MIN_W = 0.4
COV_TYPES= ['full','tied','diag','spherical']


def main():
    df = pd.read_csv(Path(__file__).resolve().parent / 'data.csv')
    swarm = [Particle(df) for _ in range(POPULATION)]
    
    # Gets best models position and score
    best = min(swarm, key=lambda p: p.sBest)
    GsBest = best.sBest
    GpBest = best.pBest

    print(f"Initial Swarm Best BIC: {GsBest:.2f}\n")
    
    # Constants
    c1=1.8
    c2=1.8

    for i in range(MAX_ITER):
        # Weight decays over time
        w = MAX_W - ((MAX_W-MIN_W)/MAX_ITER) * i

        for particle in swarm:
            r1 = np.random.rand(3)
            r2 = np.random.rand(3)

            v1 = (w * particle.V) + (c1 * r1 * (particle.pBest - particle.X )) + (c2 * r2 * (GpBest - particle.X))
            particle.update_Values(v1)

            if particle.sBest < GsBest:
                GsBest = particle.sBest
                GpBest = particle.pBest

        print(f"{i+1}/{MAX_ITER}, Best BIC score: {GsBest:.2f}")

    print(f'''Best parameters for Gaussian Mixture
            n_components = {round(GpBest[0])},
            covariance_type = {COV_TYPES[round(GpBest[1])]},
            tol = {GpBest[2]:.5f}''')

if __name__ == '__main__':
    main()