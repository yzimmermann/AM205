#Lucas Steinberger
#2024-06-10
#Class For Power Method implementation

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
from scipy.sparse import csr_matrix, issparse
from scipy.sparse.linalg import eigs

class PowerMethod:
    def __init__(self, matrix, alpha, tol=1e-6, max_iter=1000, MODE = 'auto'):
        """
        Initialize the PowerMethod instance.
        general designed to have one matrix per instance.
        assumes a sparse or dense numpy array input with NO DANGLING NODES (i.e. already handled).
        :param matrix: The Pagerank Matrix
        """

        self.dangle_check(matrix)
        #self.matrix, self.issparse = self.choose_matrix_format(matrix, mode=MODE)
        self.matrix, self.issparse = self.simple_choose_matrix_format(matrix)
        self.n = self.matrix.shape[0]
        self.alpha = alpha #for nome, just a tag for identification later on. assuming matrix contains alpha already.
        self.tol = tol
        self.max_iter = max_iter
        self.state = np.ones(self.n) / self.n # Start with uniform distribution
        self.residuals = []
        self.v = np.ones(self.n)/ self.n # Personalization vector (uniform)
        self.pi = None # this is the final eigenvector

    @staticmethod
    def dangle_check(matrix):
        """
        checks to make sure matrix is appropriate: all columns sum to 1
        """
        col_sums = np.asarray(matrix.sum(axis=0)).flatten()
        if not np.allclose(col_sums, 1):
            raise ValueError("Matrix contains dangling nodes (columns that do not sum to 1). Please handle dangling nodes before using PowerMethod.")
    @staticmethod
    def mode_check(matrix):
        """
        checks to see if matrix is dense or sparse
        """
        if sparse.issparse(matrix):
            return 'sparse'
        elif isinstance(matrix, np.ndarray):
            return 'dense'
        else:
            raise TypeError("Matrix must be either a numpy array or a scipy sparse matrix.")

    @staticmethod
    def choose_matrix_format(matrix, mode='auto', size_thresh=1000, density_thresh=0.1):
        """
        Convert `matrix` to either dense (numpy.ndarray) or sparse CSR depending on `mode`.
        Returns (A_converted, is_sparse_bool).
        mode: 'auto'|'sparse'|'dense'
        Simple heuristic for 'auto': use sparse if n > size_thresh and density <= density_thresh.
        """
        # detect input and basic size info
        if issparse(matrix):
            n = matrix.shape[0]
            nnz = matrix.nnz
        else:
            A = np.asarray(matrix)
            n = A.shape[0]
            nnz = np.count_nonzero(A)

        density = nnz / (n * n)
        if mode == 'auto':
            use_sparse = (n > size_thresh) and (density <= density_thresh)
        elif mode == 'sparse':
            use_sparse = True
        elif mode == 'dense':
            use_sparse = False
        else:
            raise ValueError("mode must be 'auto', 'sparse' or 'dense'")

        if use_sparse:
            if not issparse(matrix):
                return csr_matrix(matrix, dtype=float), True
            return matrix.tocsr().astype(float), True
        else:
            if issparse(matrix):
                return matrix.toarray().astype(float), False
            return np.asarray(matrix, dtype=float), False

    @staticmethod
    def simple_choose_matrix_format(matrix):
        if issparse(matrix):
            if matrix.dtype != np.float64:
                matrix = matrix.astype(np.float64, copy=False)
            return matrix, True

        A = np.asarray(matrix, dtype=np.float64)
        return A, False

    def __getitem__(self, key):
        #to make accessing variables easier
        return getattr(self, key)


    def run(self, eigs = False):
        #run the standard suite
        self.iterate()
        self.fit_residuals()
        if eigs:
            self.full_eigenvalues()
    def step(self):
        """
        Perform one iteration of the power method, normalizing the result and computing the residual.
        """
        y = self.matrix @ self.state
        state_new = self.alpha * y + (1.0 - self.alpha) * self.v
        residual = np.linalg.norm(state_new - self.state, 1) # L1 norm for residual
        self.state = state_new
        self.residuals.append(residual)

    def iterate(self):
        """
        Run the power method until convergence or maximum iterations reached.
        Data from run is stored as an instance variable. running again will wipe this instance variable.
        """
        if self.pi is not None:
            print("Warning: Overwriting existing pi vector.")
        self.state = np.ones(self.matrix.shape[0]) / self.matrix.shape[0]
        self.residuals = []
        for _ in range(self.max_iter):
            self.step()
            if self.residuals[-1] < self.tol:
                break
        self.pi = self.state.copy()

    def full_eigenvalues(self):
        """
        Compute all eigenvalues and eigenvectors of the matrix.
        """
        M = self.matrix
        if issparse(M):
            M = M.toarray()
        eigenvalues, eigenvectors = np.linalg.eig(M)
        self.eigenvalues = eigenvalues
        self.eigenvectors = eigenvectors
        return eigenvalues

    def estimate_second_eigenvalue(self, k=3) -> float:
        """
        Yoel's method (potentially redundant).
        Estimate the magnitude of the second-largest eigenvalue.

        Returns:
            float: |lambda_2| (real scalar). np.nan if estimation fails.
        """
        M = self.matrix
        n = M.shape[0]
        if n <= 2:
            return np.nan

        # k must be < n
        k = min(k, n - 1)

        v = self.v.reshape(-1, 1)

        oneT = np.ones((1, n))
        G = self.alpha * M + (1.0 - self.alpha) * (v @ oneT)
        G = csr_matrix(G)

        try:
            vals, _ = eigs(G.T, k=k, which="LM")
        except Exception as e:
            print(f"[WARNING] eigs failed: {e}")
            return np.nan

        abs_vals = np.sort(np.abs(vals))[::-1]  # sort
        if abs_vals.size < 2:
            return np.nan

        lam2 = abs_vals[1]
        return float(np.real(lam2))

    def fit_residuals(self):
        """
        do fits for the residuals
        """
        coeffs = np.polyfit(np.arange(len(self.residuals)), np.log(self.residuals), 1)
        #store the slope and intercept
        self.fit = coeffs



class MultiPlotter():
    def __init__(self, power_methods,):
        """
        Object to handle plotting of data from one or multiple PowerMethod instances
        Takes in a group of PowerMethod instances, and can parse them to make subgroups depending on the flags they may have.
      """
        #deal with single instance input
        if type(power_methods) is PowerMethod:
            power_methods = [power_methods]
        self.power_methods = power_methods
        self.plots = {} #dictionary of plots.



    def plot_residuals(self, filters, label_attributes, ax=None, fits = True, show = True):
        """
        Plot residuals for PowerMethod instances that match the given filters.
        This method is a little confusing, the hope is to make it easy to control what you want to plot from just a few parameters
        in the notebook/script, but that means some complexity on the backend here.
        :param filters: Dictionary of attributes to filter PowerMethod instances.
        :param label_attributes: List of attributes to include in the plot labels.
        :param ax: Matplotlib axis to plot on. If None, a new figure and axis are created.
        """
        if ax is None:
            fig, ax = plt.subplots()

        for pm in self.power_methods:
            match = all(pm[key] == value for key, value in filters.items()) if filters else True
            if match:
                label = ', '.join(f"{attr}={getattr(pm, attr)}" for attr in label_attributes)
                ax.semilogy(pm.residuals, label=label)
                if fits and hasattr(pm, 'fit'):
                    slope, intercept = pm.fit
                    x_vals = np.arange(len(pm.residuals))
                    fit_line = np.exp(intercept) * np.exp(slope * x_vals)
                    ax.semilogy(x_vals, fit_line, label = f"slope = {slope:.3f}", linestyle='--')

        ax.set_xlabel('Iteration')
        ax.set_ylabel('Residual (L1 norm)')
        ax.set_title('Power Method Residuals')
        ax.legend()

        if show:
            plt.show()