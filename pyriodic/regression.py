import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression


class CircularLinearRegressor:
    def __init__(self):
        self.slope_sin = None
        self.slope_cos = None
        self.intercept_sin = None
        self.intercept_cos = None
        self.rho = None
        self.fitted = False

    def fit(self, x, angles):
        """Fit the model to linear predictor `x` and circular response `angles`."""
        x = np.asarray(x)
        angles = np.asarray(angles)

        sin_y = np.sin(angles)
        cos_y = np.cos(angles)

        self.slope_sin, self.intercept_sin, *_ = stats.linregress(x, sin_y)
        self.slope_cos, self.intercept_cos, *_ = stats.linregress(x, cos_y)

        self.rho = self._circular_linear_corr(angles, x)
        self.fitted = True
        return self

    def predict(self, x):
        """Predict circular response angles from linear predictor x."""
        if not self.fitted:
            raise RuntimeError("Model must be fit before predicting.")

        x = np.asarray(x)
        sin_pred = self.slope_sin * x + self.intercept_sin
        cos_pred = self.slope_cos * x + self.intercept_cos
        predicted_angles = np.arctan2(sin_pred, cos_pred) % (2 * np.pi)
        return predicted_angles

    def permutation_test(self, x, angles, n_perm=1000, random_state=None):
        """Perform permutation test for significance of correlation."""
        if not self.fitted:
            raise RuntimeError("Model must be fit before running permutation test.")

        if random_state is not None:
            np.random.seed(random_state)

        x = np.asarray(x)
        angles = np.asarray(angles)
        rhos = np.zeros(n_perm)

        for i in range(n_perm):
            x_perm = np.random.permutation(x)
            rhos[i] = self._circular_linear_corr(angles, x_perm)

        p_value = np.mean(rhos >= self.rho)
        return p_value

    @staticmethod
    def _circular_linear_corr(angles, x):
        sin_a = np.sin(angles)
        cos_a = np.cos(angles)
        rxs = stats.pearsonr(x, sin_a)[0]
        rxc = stats.pearsonr(x, cos_a)[0]
        rcs = stats.pearsonr(sin_a, cos_a)[0]
        rho = np.sqrt((rxc**2 + rxs**2 - 2 * rxc * rxs * rcs) / (1 - rcs**2))
        return rho


class LinearCircularRegressor:
    """
    Linear-circular regression model.

    Models a linear response variable (e.g., RT) as a function of a circular predictor
    (e.g., phase) using sine and cosine components.

    Attributes:
        model (LinearRegression): underlying scikit-learn regression model
        intercept_ (float): model intercept
        beta_cos (float): coefficient for cos(phase)
        beta_sin (float): coefficient for sin(phase)
        r_squared (float): model R^2 (explained variance)
        fitted (bool): whether model has been fit
    """

    def __init__(self):
        self.model = None
        self.intercept_ = None
        self.beta_cos = None
        self.beta_sin = None
        self.r_squared = None
        self.fitted = False

    def fit(self, angles, y):
        """
        Fit the model to circular predictor and linear response.

        Parameters:
            angles (array-like): Circular predictor in radians.
            y (array-like): Linear response variable.
        """
        angles = np.asarray(angles)
        y = np.asarray(y)

        X = np.column_stack((np.cos(angles), np.sin(angles)))
        self.model = LinearRegression().fit(X, y)

        self.intercept_ = self.model.intercept_
        self.beta_cos, self.beta_sin = self.model.coef_
        self.r_squared = self.model.score(X, y)
        self.fitted = True
        return self

    def predict(self, angles):
        """
        Predict linear response from circular predictor.

        Parameters:
            angles (array-like): Circular predictor in radians.

        Returns:
            np.ndarray: Predicted linear values.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fit before predicting.")

        angles = np.asarray(angles)
        X = np.column_stack((np.cos(angles), np.sin(angles)))
        return self.model.predict(X)

    def permutation_test(self, angles, y, n_perm=1000, random_state=None):
        """
        Perform a permutation test on the model's R^2.

        Parameters:
            angles (array-like): Circular predictor in radians.
            y (array-like): Linear response.
            n_perm (int): Number of permutations.
            random_state (int or None): Seed for reproducibility.

        Returns:
            float: Permutation p-value.
        """
        if not self.fitted:
            raise RuntimeError("Model must be fit before running permutation test.")

        if random_state is not None:
            np.random.seed(random_state)

        r2_null = np.zeros(n_perm)
        angles = np.asarray(angles)
        y = np.asarray(y)

        for i in range(n_perm):
            y_perm = np.random.permutation(y)
            X_perm = np.column_stack((np.cos(angles), np.sin(angles)))
            model_perm = LinearRegression().fit(X_perm, y_perm)
            r2_null[i] = model_perm.score(X_perm, y_perm)

        p_value = np.mean(r2_null >= self.r_squared)
        return p_value
