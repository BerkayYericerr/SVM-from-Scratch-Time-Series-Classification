import numpy as np
from scipy.optimize import minimize



def linear_kernel(x, y):
    return np.dot(x, y)


def rbf_kernel(x, y, sigma=1.0):
    diff = x - y
    return np.exp(-np.dot(diff, diff) / (2 * sigma * sigma))


def poly_kernel(x, y, c=1.0, d=2):
    return (np.dot(x, y) + c) ** d


def trend_kernel(x, y, sigma_delta=1.0):
    dx = np.diff(x)
    dy = np.diff(y)
    diff = dx - dy
    return np.exp(-np.dot(diff, diff) / (2 * sigma_delta * sigma_delta))


def custom_kernel(x, y, alpha=1/3, beta=1/3, gamma=1/3, sigma=1.0, sigma_delta=1.0):
    s = alpha + beta + gamma
    if s <= 0:
        raise ValueError("alpha+beta+gamma must be > 0")
    alpha, beta, gamma = alpha / s, beta / s, gamma / s

    return (
        alpha * linear_kernel(x, y) +
        beta * rbf_kernel(x, y, sigma=sigma) +
        gamma * trend_kernel(x, y, sigma_delta=sigma_delta)
    )


class SVM:
    def __init__(
        self,
        C=1.0,
        kernel='linear',
        sigma=1.0,          # for RBF and custom
        degree=2,           # for polynomial
        coef0=1.0,          # c in (x·y + c)^d
        k_alpha=1/3, k_beta=1/3, k_gamma=1/3,  # custom kernel weights
        sigma_delta=1.0     # for trend kernel part
    ):
        self.C = float(C)
        self.kernel_type = kernel

        # kernel params
        self.sigma = float(sigma)
        self.degree = int(degree)
        self.coef0 = float(coef0)
        self.k_alpha = float(k_alpha)
        self.k_beta = float(k_beta)
        self.k_gamma = float(k_gamma)
        self.sigma_delta = float(sigma_delta)

        # learned params
        self.alphas = None            
        self.support_vectors = None
        self.sv_labels = None
        self.sv_alphas = None
        self.b = 0.0

        
        self.X_train = None
        self.y_train = None

    def kernel(self, x, y):
        if self.kernel_type == 'linear':
            return linear_kernel(x, y)

        elif self.kernel_type == 'rbf':
            return rbf_kernel(x, y, sigma=self.sigma)

        elif self.kernel_type in ['poly', 'polynomial']:
            return poly_kernel(x, y, c=self.coef0, d=self.degree)

        elif self.kernel_type in ['custom', 'trend']:
            return custom_kernel(
                x, y,
                alpha=self.k_alpha, beta=self.k_beta, gamma=self.k_gamma,
                sigma=self.sigma, sigma_delta=self.sigma_delta
            )

        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")

    def fit(self, X, y):
        X = np.asarray(X, dtype=float)
        y = np.asarray(y)

        n = len(X)
        if n == 0:
            raise ValueError("Empty training set")

        # convert 0/1 to -1/+1
        y_conv = np.where(y == 0, -1.0, 1.0)

        self.X_train = X
        self.y_train = y_conv

       
        # FAST kernel matrix 
        X = np.asarray(X, dtype=float)

        if self.kernel_type == "linear":
            K = X @ X.T

        elif self.kernel_type in ["poly", "polynomial"]:
            K = (X @ X.T + self.coef0) ** self.degree

        elif self.kernel_type == "rbf":
            X2 = np.sum(X * X, axis=1, keepdims=True)
            D2 = X2 + X2.T - 2 * (X @ X.T)
            K = np.exp(-D2 / (2 * self.sigma * self.sigma))

        elif self.kernel_type in ["custom", "trend"]:
            K_lin = X @ X.T

            X2 = np.sum(X * X, axis=1, keepdims=True)
            D2 = X2 + X2.T - 2 * (X @ X.T)
            K_rbf = np.exp(-D2 / (2 * self.sigma * self.sigma))

            DX = np.diff(X, axis=1)
            DX2 = np.sum(DX * DX, axis=1, keepdims=True)
            D2_trend = DX2 + DX2.T - 2 * (DX @ DX.T)
            K_trend = np.exp(-D2_trend / (2 * self.sigma_delta * self.sigma_delta))

            # normalize weights
            s = self.k_alpha + self.k_beta + self.k_gamma
            a = self.k_alpha / s
            b = self.k_beta / s
            g = self.k_gamma / s

            K = a * K_lin + b * K_rbf + g * K_trend

        else:
            raise ValueError(f"Unknown kernel_type: {self.kernel_type}")


        
        Q = np.outer(y_conv, y_conv) * K
        Q = Q + 1e-8 * np.eye(n)

        def func(a):
            return 0.5 * (a @ (Q @ a)) - np.sum(a)

        def grad(a):
            return (Q @ a) - np.ones_like(a)


       
        cons = {'type': 'eq', 'fun': lambda a: np.dot(a, y_conv)}
        bounds = [(0.0, self.C)] * n

        res = minimize(
            func,
            x0=np.zeros(n),
            jac=grad,
            bounds=bounds,
            constraints=cons,
            method="SLSQP",
            options={"maxiter": 200, "ftol": 1e-6, "disp": False}
        )


        if not res.success:
            raise RuntimeError(f"SVM optimization failed: {res.message}")

        self.alphas = res.x

        # support vectors
        sv_idx = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = X[sv_idx]
        self.sv_labels = y_conv[sv_idx]
        self.sv_alphas = self.alphas[sv_idx]

        m = len(sv_idx)
        if m == 0:
            self.b = 0.0
        else:
            K_sv = K[np.ix_(sv_idx, sv_idx)]              
            v = self.sv_alphas * self.sv_labels            
            decision_on_sv = v @ K_sv                      
            self.b = np.mean(self.sv_labels - decision_on_sv)
        return self
    def decision_function(self, X):
        X = np.asarray(X, dtype=float)
        scores = np.zeros(len(X), dtype=float)

        for idx, x in enumerate(X):
            s = 0.0
            for i in range(len(self.support_vectors)):
                s += self.sv_alphas[i] * self.sv_labels[i] * self.kernel(self.support_vectors[i], x)
            scores[idx] = s + self.b

        return scores

    def predict(self, X):
        scores = self.decision_function(X)
        return (scores >= 0).astype(int)

    def score(self, X, y):
        y = np.asarray(y)
        preds = self.predict(X)
        return np.mean(preds == y)
