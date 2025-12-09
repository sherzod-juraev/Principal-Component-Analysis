from numpy import ndarray, cov
from numpy.linalg import eigh


class PCA:
    """Principal Component Analysis (PCA) implementation using NumPy.

    PCA is a dimensionality reduction technique that transforms the original
    data into a new coordinate system where the greatest variance lies on the
    first axes (principal components), the second greatest variance on the
    second axes, and so on. It is commonly used to reduce feature dimensionality
    while preserving as much variance as possible.

    Parameters
    ----------
    n_components : int
        Number of principal components to keep. Determines the dimensionality
        of the transformed data.
    """

    def __init__(self, n_components: int, /):

        self.n_components = n_components

    def fit(self, X: ndarray, /) -> ndarray:
        """Fit training data"""

        Xc, n, d = X - X.mean(axis=0), X.shape[0], X.shape[1]
        Cov = cov(Xc, rowvar=False)
        eignvals, eigenvecs = eigh(Cov)
        idx = eignvals.argsort()[::-1]
        eignvals, eigenvecs = eignvals[idx], eigenvecs[:, idx]
        W = eigenvecs[:, :self.n_components]
        Z = Xc.dot(W)
        return Z