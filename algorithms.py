import numpy as np
from typing import Literal
from abc import ABC, abstractmethod

from scipy.sparse.linalg import svds
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm


class BaseOptimizer(ABC):
    def __init__(self, X: np.ndarray, R_obs: csr_matrix, delta: float):
        """
        Base class for optimization algorithms.
        input:
            X: np.ndarray - initial point in the optimization space
            R_obs: csr_matrix - observed entries of the matrix (ground truth)
            delta: float - hyper parameter that controls the bound of the solution space, i.e., ||X||_* (nuclear norm) <= delta
        """
        self.R_obs = R_obs
        self.X = X.copy()

        row, col = R_obs.nonzero()
        self.X_obs = csr_matrix((self.X[row, col], (row, col)), shape=R_obs.shape)

        self.delta = delta
        self.k = 0
        self.lr = None

        self.duality_gap_value = None

        self.logs = {}

    def grad(self) -> csr_matrix:
        """
        Compute the gradient of the objective function at the current point X_k.
        """
        return 2 * (self.X_obs - self.R_obs)

    def lmo(self, Grad: csr_matrix) -> np.ndarray:
        """
        Linear Minimization Oracle (LMO) for the Frank-Wolfe algorithm.

        input:
            Grad: csr_matrix - the gradient at the current point X_k
        output:
            S_k: np.ndarray - the solution to the LMO, which is the point that minimizes the linear approximation of the objective function at X_k
        """

        u, s, vt = svds(Grad, k=1, which="LM")  #CHANGE which="LM"

        S_k = - self.delta * (u @ vt)

        return S_k

    def line_search(self, D_k: np.ndarray, max_alpha: float = 1) -> float:
        """
        Perform a line search to find the optimal step size alpha that minimizes the objective function along the direction D_k.
        
        input:
            D_k: np.ndarray - the descent direction at the current point X_k
            max_alpha: float - maximum step size to consider
        
        output:
            alpha: float - the optimal step size that minimizes the objective function along the direction D_k
            alpha = d_k^T * (X - R) / ||d_k||^2, clipped to [0, max_alpha]
        """
        rows, cols = self.R_obs.nonzero()  # Get the indices of the observed entries
        r_data = self.R_obs.data

        resid = self.X[rows, cols] - r_data # Compute residual = X_{ij} - R_{ij} at observed entries

        dvals = D_k[rows, cols] # Compute direction‐values on those same entries

        num   = float(np.dot(dvals, resid)) # Numerator = sum_i [ dvals_i * resid_i ]
        denom = float(np.dot(dvals, dvals)) # Denominator = sum_i [ dvals_i^2 

        alpha = - num / denom

        alpha = np.clip(alpha, 0, max_alpha)
        return alpha

    @abstractmethod
    def step(self):
        pass

    def duality_gap(self) -> float:
        if self.duality_gap_value is None:
            raise ValueError("Duality gap has not been computed yet. Call step() first.")

        return self.duality_gap_value        


class FW(BaseOptimizer):
    def __init__(self, X: np.ndarray, R_obs: csr_matrix, delta: float, lr_strategy: Literal["line-search", "fixed"] = "fixed"):
        """
        Frank-Wolfe algorithm for matrix completion.
        input:
            X: np.ndarray - initial point in the optimization space
            R_obs: csr_matrix - observed entries of the matrix (ground truth)
            delta: float - hyper parameter that controls the bound of the solution space, i.e., ||X||_* (nuclear norm) <= delta
            lr_strategy: str - strategy for learning rate, either "line-search" or "fixed"
        """
        super().__init__(X, R_obs, delta)

        if lr_strategy not in ["line-search", "fixed"]:
            raise ValueError("lr_strategy must be either 'line-search' or 'fixed'")
        
        self.lr_strategy = lr_strategy
        self.delta = delta

    def fw_step(self, Grad: csr_matrix) -> np.ndarray:
        S_k = self.lmo(Grad) # linear minimization oracle
        D_k = S_k - self.X # descent direction

        rows, cols = Grad.nonzero()
        gap = - float((Grad.data * D_k[rows, cols]).sum()) / Grad.data.shape[0]

        return S_k, D_k, gap

    def step(self):
        """
        Perform one step of the Frank-Wolfe algorithm.
        output:
            X_{k+1}: np.ndarray - updated point in the optimization space
            X_{k+1} = X_k + lr * D_k, where D_k = S_k - X_k, S_k is the solution to the linear minimization oracle (LMO)
        """
        Grad = self.grad()

        S_k, D_k, gap = self.fw_step(Grad)

        if self.lr_strategy == "fixed":
            self.lr = 2 / (2 + self.k) # fixed stepsize
        else:
            self.lr = self.line_search(D_k, max_alpha=1) # line search stepsize

        self.X += self.lr * D_k # update X

        rows, cols = self.X_obs.nonzero()
        self.X_obs = csr_matrix((self.X[rows, cols], (rows, cols)), shape=self.X_obs.shape)

        self.duality_gap_value = gap

        self.k += 1

        self.logs = {
            "duality_gap": self.duality_gap_value,
            "step_size": self.lr,
            
        }

#--------------
class FWAwayStepInFace(FW):
    """
    In-Face Away-Step Frank-Wolfe algorithm for matrix completion.

    Extends the standard Frank-Wolfe (FW) method by introducing in-face away steps,
    which aim to reduce the rank by moving inside or to a lower-dimensional face 
    of the nuclear norm ball. At each iteration, it compares the classical FW step
    with an in-face away direction and selects the one providing the largest 
    duality gap reduction.

    Attributes:
        X: np.ndarray
            Current matrix estimate.
        R_obs: csr_matrix
            Sparse matrix of observed entries (ground truth).
        delta: float
            Bound on the nuclear norm ||X||_* ≤ delta.
        rank: int
            Target rank for thin SVD to define the current face.
        duality_gap_tolerance: float
            Tolerance for determining if we stop moving to new faces.
    """

    def __init__(self, X: np.ndarray, R_obs: csr_matrix, delta: float, rank: int = 5):

        """
        Initialize the FWAwayStepInFace optimizer.

        Extends the standard Frank-Wolfe (FW) algorithm to incorporate
        in-face away steps by setting up the necessary parameters.

        Sets:
            self.duality_gap_tolerance: float
                Stopping tolerance for deciding when to switch from
                FW steps to in-face steps.
            self.rank: int
                Stores the rank for low-dimensional face operations.
        """

        super().__init__(X, R_obs, delta, lr_strategy="line-search")
        self.duality_gap_tolerance = 10 ** (0)
        self.rank = rank   # target rank for Face

    def compute_inface_away_direction(self, Grad: csr_matrix):

        """
        Compute the in-face away direction based on the current iterate X.

        Performs the following steps:
        1. Computes a thin SVD of the current iterate X to identify 
            the local low-rank face of the nuclear norm ball.
        2. Projects the gradient onto this small r x r face subspace
            to obtain G^k.
        3. Solves an approximate minimization inside this face by 
            using the dominant eigenvector of the symmetrized G^k 
            to construct a low-rank matrix M_hat.
        4. Builds Z_hat as the lifted matrix back in the full space.
        5. Computes the away direction D_Away = X - Z_hat and estimates 
            the approximate duality gap for this direction.

        input:
            Grad: csr_matrix
                Gradient matrix computed on observed entries.

        output:
            Z_hat: np.ndarray
                Approximate optimal low-rank matrix inside the face.
            D_Away: np.ndarray
                Away descent direction pointing from X to Z_hat.
            gap: float
                Approximate duality gap along this in-face away direction.
        """

        # 1. Thin SVD of current iterate X
        U, S, Vt = svds(self.X, k=self.rank)
        U = U[:, ::-1]
        S = S[::-1]
        Vt = Vt[::-1, :]

        # 2. Compute G^k which is a small r x r matrix projected on the small face of our convex constraint
        Gk = U.T @ Grad @ Vt.T  # shape (r, r)

        # 3. Solve Quadratic Program: min trace(M^T Gk), subject to ||M||_* <= delta
        # Optimal solution: take M = - delta * sign(Gk)
        # But to be stable: project on nuclear norm ball
        # In our case, simplest: take only top singular vector (approximate step)

        # Eigendecomposition of sym(Gk)
        sym_Gk = 0.5 * (Gk + Gk.T)
        eigvals, eigvecs = np.linalg.eigh(sym_Gk)
        u_k = eigvecs[:, -1]  # largest eigvec

        M_hat = self.delta * np.outer(u_k, u_k)

        # Build new Z_hat
        Z_hat = U @ M_hat @ Vt

        D_Away = self.X - Z_hat

        rows, cols = Grad.nonzero()
        gap = -float((Grad.data * D_Away[rows, cols]).sum()) / Grad.data.shape[0]

        return Z_hat, D_Away, gap

    def inface_max_step_size(self, D_k: np.ndarray) -> float:
        
        """
        Estimate the maximum feasible step size along the in-face direction D_k.

        Projects the proposed direction D_k into the low-rank face defined by the
        thin SVD of the current iterate X. By computing the eigenvalues of the
        symmetrized projected matrix, it ensures that stepping along this direction
        does not violate the local face or push the iterate outside the feasible
        set.

        Returns a clipped step size between 0 and 1. Falls back to 1.0 in case
        of numerical issues or exceptions.

        input:
            D_k: np.ndarray
                Proposed descent direction (difference matrix).

        output:
            alpha_max: float
                Maximum safe step size along D_k, clipped to [0, 1].
        """

        try:
            d = D_k
            U, S, Vt = svds(self.X, k = self.rank)
            D_proj = U.T @ d @ Vt.T
            D_sym = 0.5 * (D_proj + D_proj.T)
            eigvals = np.linalg.eigenvalsh(D_sym)
            alpha_max = - 1.0 / min(eigvals)
            return float(np.clip(alpha_max, 0, 1.0))
        except Exception:
            return 1.0
        
    def step(self):

        """
        Perform one iteration of the in-face Frank-Wolfe algorithm. While comparing the FW step with the in-face Away Step.

        Updates:
            self.X: np.ndarray
                Current matrix iterate after the update.
            self.X_obs: csr_matrix
                Updated sparse matrix view on observed entries.
            self.k: int
                Iteration counter incremented by one.
            self.lr: float
                Last step size used.
            self.duality_gap_value: float
                Current estimate of the duality gap.
            self.logs: dict
                Dictionary recording duality gap, step size, and step type.
        """

        Grad = self.grad()

        # FW step
        _, D_FW, gap_FW = self.fw_step(Grad)

        # In-face Away step
        _, D_Away, gap_Away = self.compute_inface_away_direction(Grad)


        if gap_FW >= gap_Away:
            direction = D_FW
            gamma_max = 1.0
            step_type = "FW"
            self.duality_gap_value = gap_FW
        else:
            direction = D_Away
            gamma_max = self.inface_max_step_size(direction)  # no active set tracking here
            step_type = "InFace"
            self.duality_gap_value = gap_Away

        gamma = self.line_search(direction, max_alpha=gamma_max)
        self.lr = gamma
        self.X += gamma * direction

        rows, cols = self.X_obs.nonzero()
        self.X_obs = csr_matrix((self.X[rows, cols], (rows, cols)), shape=self.X_obs.shape)

        self.k += 1

        self.logs = {
            "duality_gap": self.duality_gap_value,
            "step_size": self.lr,
            "step_type": step_type,
        }

#--------------

class FWPairwiseInFace(FWAwayStepInFace):

    """
    Pairwise In-Face Frank-Wolfe algorithm for matrix completion.

    Extends the in-face away-step variant by introducing a pairwise
    direction that explicitly moves between two atoms: one from the
    standard Frank-Wolfe direction and one from the in-face away 
    direction.

    Inherits from:
        FWAwayStepInFace

    Notes:
        - S_fw: np.ndarray
            The new atom found by the Frank-Wolfe LMO (leading singular vectors).
        - S_aw: np.ndarray
            The atom inside the current low-rank face used to build the in-face away step.
    """

    def __init__(self, X: np.ndarray, R_obs: csr_matrix, delta: float, rank: int = 5):
        super().__init__(X, R_obs, delta, rank)

    def step(self):
        Grad = self.grad()


        S_fw, D_FW, gap_FW = self.fw_step(Grad)

        # 2. In-face Away direction
        S_aw, _, _ = self.compute_inface_away_direction(Grad)

        # 3. Pairwise direction: S_fw - Z_hat
        direction = S_fw - S_aw

        # Duality gap approx (just for tracking)
        rows, cols = Grad.nonzero()
        gap_PW = -float((Grad.data * direction[rows, cols]).sum()) / Grad.data.shape[0]

        gamma_max = self.inface_max_step_size(direction)
        step_type = "PairwiseInFace"

        self.duality_gap_value = gap_PW
        
        self.lr = self.line_search(direction, max_alpha=gamma_max)
        self.X += self.lr * direction

        rows, cols = self.X_obs.nonzero()
        self.X_obs = csr_matrix((self.X[rows, cols], (rows, cols)), shape=self.X_obs.shape)

        self.k += 1

        self.logs = {
            "duality_gap": self.duality_gap_value,
            "step_size": self.lr,
            "step_type": step_type,
        }
