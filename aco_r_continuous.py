import numpy as np


class ACOR:
    """
    Simplified continuous Ant Colony Optimization (based on ACO_R)
    for minimizing a continuous objective function f(x).

    x is a vector in R^dim, with bounds [lb, ub] in each dimension.
    """

    def __init__(
        self,
        func,
        dim,
        n_ants=8,
        archive_size=8,
        q=0.3,
        xi=0.65,
        max_iter=300,
        lb=-5.12,
        ub=5.12,
        tol=1e-3,
    ):
        self.func = func          # func(real_vector) -> scalar
        self.dim = dim
        self.n_ants = n_ants      # number of new candidates per iter
        self.archive_size = archive_size
        self.q = q
        self.xi = xi
        self.max_iter = max_iter
        self.lb = lb
        self.ub = ub
        self.tol = tol

        # history of best values (for possible plotting)
        self.best_history = []

        # history of best positions (for plotting if needed)
        self.best_history_X = []

    # --- internal helpers -------------------------------------------------

    def _evaluate(self, X_norm):
        """
        Evaluate a batch of candidates given in normalized [0,1]^dim space.
        We map to [lb, ub] and call func on each.
        """
        X_real = self.lb + X_norm * (self.ub - self.lb)
        return np.array([self.func(x) for x in X_real])

    # --- main run ---------------------------------------------------------

    def run(self):
        """
        Run the optimization.

        Returns:
            best_x_real (np.ndarray): best solution in real space
            best_y (float): best objective value
        """
        nSize = self.archive_size
        nVar = self.dim

        # --- initialization: archive with uniform random points in [0,1]^dim
        S_norm = np.random.uniform(0.0, 1.0, size=(nSize, nVar))
        f_vals = self._evaluate(S_norm).reshape(-1, 1)

        # S: archive of [x_norm, f]
        S = np.hstack([S_norm, f_vals])
        # sort by fitness (ascending = minimization)
        S = S[np.argsort(S[:, -1])]

        best_x_norm = S[0, :nVar].copy()
        best_y = S[0, -1]
        self.best_history.append(best_y)

        # weights (as in ACO_R)
        qk = self.q * nSize
        w = np.zeros(nSize)
        for i in range(nSize):
            w[i] = (
                1.0
                / (qk * 2.0 * np.pi)
                * np.exp(-i ** 2 / (2.0 * (self.q ** 2) * (nSize ** 2)))
            )
        w = w / np.sum(w)

        stop_counter = 0

        for it in range(1, self.max_iter + 1):
            # in original ACO_R selection is argmax(w) (always best index ~ 0)
            selection = np.random.choice(range(nSize), p=w)  # index in archive

            # compute sigma for each dimension (ACO_R style)
            sigma = np.zeros((nVar,))
            for j in range(nVar):
                sigma[j] = self.xi * (np.std(S[:, j]) + 1e-6)

            # generate new ants around selected solution
            Stemp_norm = np.zeros((self.n_ants, nVar))
            for k in range(self.n_ants):
                for j in range(nVar):
                    Stemp_norm[k, j] = (
                        sigma[j] * np.random.random_sample() + S[selection, j]
                    )
                    # clip to [0,1]
                    Stemp_norm[k, j] = np.clip(Stemp_norm[k, j], 0.0, 1.0)

            f_new = self._evaluate(Stemp_norm).reshape(-1, 1)
            Ssample = np.hstack([Stemp_norm, f_new])

            # merge and keep best archive_size solutions
            S_all = np.vstack([S, Ssample])
            S_all = S_all[np.argsort(S_all[:, -1])]
            S = S_all[:nSize, :]
            
            # --- periodic random restart every 50 iterations ---
            if it % 50 == 0:
                S_norm_random = np.random.uniform(0.0, 1.0, size=(int(self.n_ants * 0.2), nVar))
                Srandom_vals = self._evaluate(S_norm_random).reshape(-1, 1)
                Srandom = np.hstack([S_norm_random, Srandom_vals])

                S_all = np.vstack([S, Srandom])
                S_all = S_all[np.argsort(S_all[:, -1])]
                S = S_all[:nSize, :]

            # update best
            current_best_y = S[0, -1]
            self.best_history.append(current_best_y)

            if current_best_y + self.tol < best_y:
                best_y = current_best_y
                best_x_norm = S[0, :nVar].copy()
                stop_counter = 0
                self.best_history_X.append(best_x_norm.copy())
            else:
                stop_counter += 1

            # Removed this line as per instructions:
            # self.best_history_X.append(best_x_norm.copy())

        # map best solution back to real space
        best_x_real = self.lb + best_x_norm * (self.ub - self.lb)
        return best_x_real, float(best_y)