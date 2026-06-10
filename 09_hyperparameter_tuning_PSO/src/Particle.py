from sklearn.mixture import GaussianMixture
import numpy as np

# Configuration
L_BOUND = np.array([2, 0, 0.0001])
U_BOUND = np.array([20, 3, 0.1])
RANGE = U_BOUND - L_BOUND
V_MAX = 0.20 * (RANGE)
COV_TYPES= ['full','tied','diag','spherical']

class Particle:
    def __init__(self, df):

        # Initial position and velocity of the particle
        self.df = df
        self.X = np.random.uniform(L_BOUND, U_BOUND)
        self.V = np.random.uniform(-V_MAX, V_MAX)

        self.sBest = self.fitness()
        self.pBest = np.copy(self.X)
        
    def fitness(self) -> float:
        '''Trains GMM with parameters and returns BIC score'''
        # Clamp parameters within bounds
        n_comp = int(np.clip(round(self.X[0]), L_BOUND[0], U_BOUND[0]))
        cov_idx = int(np.clip(round(self.X[1]), L_BOUND[1], U_BOUND[1]))
        tol = round(float(np.clip(self.X[2], L_BOUND[2], U_BOUND[2])), 5)

        try:
            gmm = GaussianMixture(
                n_components = n_comp,
                covariance_type = COV_TYPES[cov_idx],
                tol = tol,
                max_iter=300,
                random_state=42
                )
            gmm.fit(self.df)

            return gmm.bic(self.df)
        except:
            # High penatly if theres an error
            return float('inf')

    def update_Values(self, v1) -> None:
        '''Updates Velocity, adds it to position, updates best score and position'''
        self.V = np.clip(v1, -V_MAX, V_MAX)
        self.X += self.V

        self.X = np.clip(self.X, L_BOUND, U_BOUND)

        current_score = self.fitness()

        if current_score < self.sBest:
            self.sBest = current_score
            self.pBest = np.copy(self.X)
