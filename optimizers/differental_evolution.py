from optimizers.optimizer import Optimizer
from scipy.optimize import differential_evolution

class DEA(Optimizer):
    def __init__(self, agent,
                 boundaries,
                 strategy='best1bin',
                 popsize=15,
                 mutation=0.5,
                 recombination=0.7,
                 tol=0.01, seed=
                 2020):
        super().__init__(agent)
        self.boundaries = boundaries
        self.strategy = strategy
        self.popsize = popsize
        self.mutation = mutation
        self.recombination = recombination
        self.tol = tol
        self.seed = seed

    def optimize(self, episodes):
        solver = differential_evolution(self.objective, self.boundaries, args=episodes, strategy=self.strategy,
                                        popsize=self.popsize, mutation=self.mutation, recombination=self.recombination,
                                        tol=self.tol, seed=self.seed)
        # Calculate best hyperparameters and resulting rmse
        best_hyperparams = solver.x
        best_rmse = solver.fun
        # Print final results
        return best_hyperparams, best_rmse