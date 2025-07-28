import time
import numpy as np
from tqdm import tqdm
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import norm

from algorithms import BaseOptimizer



class RecommendationSolver:
    def __init__(self, eps: float = 0.1, max_iter: int = 10_000):
        """
        Solver for recommendation systems using optimization algorithms.
        input:
            eps: float - stopping criterion for the duality gap
            max_iter: int - maximum number of iterations to run the optimization algorithm
        """
        self.eps = eps
        self.max_iter = max_iter
        self.train_error_list = []
        self.duality_gap_list = []
        self.cpu_timings = []

    @staticmethod
    def rmse(X_obs: coo_matrix, R_obs: coo_matrix):
        """
        Compute the root mean square error (RMSE) between the observed entries of the matrix and the predicted entries.
        input:
            X_obs: coo_matrix - observed entries of the matrix (predicted)
            R_obs: coo_matrix - observed entries of the matrix (ground truth)
        output:
            rmse: float - the RMSE value
        """
        return norm(X_obs - R_obs) / np.sqrt(R_obs.data.shape[0])
    

    def fit(self, optimizer: BaseOptimizer):
        """
        Fit the optimization algorithm to the observed entries of the matrix.
        input:
            optimizer: BaseOptimizer - the optimization algorithm to use
        """
        start_time = time.time()
        pbar = tqdm(range(self.max_iter))
        for k in pbar:
            
            optimizer.step()

            train_error = self.rmse(optimizer.X_obs, optimizer.R_obs)
            duality_gap = optimizer.duality_gap()

            self.train_error_list.append(train_error)
            self.duality_gap_list.append(duality_gap)
            self.cpu_timings.append(time.time() - start_time)

            optimizer_logs = optimizer.logs.copy()
            optimizer_logs.update({f"rmse": train_error})
            pbar.set_postfix(optimizer_logs)
            
            if duality_gap < self.eps: # checking stopping criterion
                pbar.set_description(f"Converged at iteration {k + 1} with duality gap {duality_gap:.4f}")
                break

        if k == self.max_iter - 1:
            pbar.set_description(f"Reached max iterations {self.max_iter} with duality gap {duality_gap:.4f}")
        
        pbar.close()

    # erase the fit method to reset the state
    def reset(self):
        """
        Reset the solver state."""
        self.X = None
        self.train_error_list = []
        self.duality_gap_list = []
        self.cpu_timings = []
