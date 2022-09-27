from numba import njit, prange, float32, int64, uint64
import numpy as np
from sklearn.random_projection import (
    BaseRandomProjection,
    johnson_lindenstrauss_min_dim,
)
from sklearn.exceptions import DataDimensionalityWarning
from sklearn.utils.validation import check_is_fitted
from scipy.optimize import minimize
from scipy.sparse import csr_matrix
from scipy.stats import beta
from typing import Dict, Iterable, List, Tuple, Type, Union
import warnings


@njit(uint64(uint64))
def splitmix64(index):
    """
    Fast, simple function for taking an integer and returning a random number. Function
    used in Java random number generator.

    Parameters
    ----------
    index : uint64

    Returns
    -------
    uint64
    """
    z = index + uint64(0x9E3779B97F4A7C15)
    z = (z ^ (z >> uint64(30))) * uint64(0xBF58476D1CE4E5B9)
    z = (z ^ (z >> uint64(27))) * uint64(0x94D049BB133111EB)

    return z ^ (z >> uint64(31))


@njit(float32[:, :](float32[:], int64[:], int64[:], int64), parallel=True)
def random_projection(
    data: np.ndarray, indices: np.ndarray, indptr: np.ndarray, d: int
):
    """
    Randomly project a sparse matrix (standard CSR representation) to a dense vectors
    by effectively multiplying by a random matrix whose elements are
    :math:`\pm 1 / \sqrt{d}`. The projection matrix is never stored in memory. Instead
    the elements are generated as needed using splitmix64().
    **Note** Embedding dimension, `d`, is rounded to a multiple of 64 since we generate
    random bits in batches of 64 in order to utilize bit manipulation to speed things
    up.

    The column indices for row i are stored in indices[indptr[i]:indptr[i+1]] and the
    corresponding values are stored in data[indptr[i]:indptr[i+1]].

    Parameters
    ----------
    data : np.ndarray, shape=(nnz,), dtype=float32
        The values of the nonzero elements in the sparse csr_matrix where nnz = number
        of nonzero elements
    indices : np.ndarray, shape=(nnz,), dtype=int64
        The column indices of the nonzero elements in the sparse csr_matrix
    indptr : np.ndarray, shape=(n_rows+1,), dtype=int64
        Pointers into `data` and `indices` to indicate where the rows start and stop.
        If you have just a single record, then indtpr=[0, len(data)]
    d : int
        Embedding dimension of dense vectors.
    
    Returns
    -------
    X : np.ndarray, shape=(n_rows, d), dtype=float32
        Dense 2-d array containing the randomly projected dense vectors for each row of
        the input sparse matrix.
    """
    assert data.shape == indices.shape, "Shape of data and indices do not match"
    assert data.ndim == 1, "data and indices must be 1-d"
    assert indptr.ndim == 1, "indptr must be 1-d"

    d = max(64, d // 64 * 64)
    # This preserves the magnitude of the vector
    scale = float32(1.0 / np.sqrt(d))
    n_rows = indptr.shape[0] - 1
    X = np.zeros((n_rows, d), float32)

    # Iterate through the rows in parallel
    for row_id in prange(n_rows):
        # Get the slices that correspond to just row_id
        index = indices[indptr[row_id] : indptr[row_id + 1]]
        value = data[indptr[row_id] : indptr[row_id + 1]]
        # Accumulate the matrix-vector project one column at a time
        for i in range(index.size):
            # Incrementally generate a column of the random matrix, 64 bits at a time
            # Multiply it by the corresponding element of the sparse vector
            # Accumulate the result in the dense projection vector
            n_chunks = d // 64
            for chunk in range(n_chunks):
                bits = splitmix64(index[i] * n_chunks + chunk)
                for bitpos in range(64):
                    sign = ((bits >> bitpos) & 1) * 2 - 1
                    X[row_id, chunk * 64 + bitpos] += sign * scale * value[i]

    return X


def distributional_johnson_lindenstrauss_optimal_delta(
    sparse_dim: int, n_components: int, eps: float
) -> float:
    """
    Algorithm to find the optimal failure rate, delta, for a given ``eps`` (error rate),
    ``sparse_dim``, and ``n_components`` (embedding dimension) for the Distributional
    Johnson-Lindenstrauss Lemma which is a different formulation of the problem.
    Taken from

    M. Skorski. *Johnson-Lindenstrauss Transforms with Best Confidence*,
    Proceedings of Machine Learning Research **134**, 1 (2021)

    which can be found at http://proceedings.mlr.press/v134/skorski21a/skorski21a.pdf

    If :math:`\mathbf{A}` is the random projection matrix, then :math:`\delta` is the
    probability of exceeding error limits given by
    
    .. math::

        \delta = \mathbb{P} \lbrack \\vert \Vert \mathbf{A}x \Vert^2_2 -
        \Vert x \Vert^2_2 \\vert > \epsilon \Vert x \Vert^2_2 {\\rbrack}
    
    
    Parameters
    ----------
    sparse_dim : int
        Original dimension of the data
    n_components : int
        Embedding dimension
    eps : float
        Error rate
    
    Returns
    -------
    delta : float
        The best probability of failure that you can expect
    """
    a = n_components / 2.0
    b = (sparse_dim - n_components) / 2.0
    z0 = a / (a + b)
    dist = beta(a, b)
    fun = lambda z: -dist.cdf((1.0 + eps) * z) + dist.cdf((1.0 - eps) * z)
    betainc_jac = lambda z: dist.pdf(z)
    jac = lambda z: -(1.0 + eps) * betainc_jac((1.0 + eps) * z) + (
        1.0 - eps
    ) * betainc_jac((1.0 - eps) * z)
    out = minimize(fun, x0=z0, jac=jac, method="Newton-CG")
    scale, delta = out.x, 1.0 + out.fun

    return delta


def random_sparse_vectors(
    n_samples: int,
    *,
    sparse_dim: int = 2 ** 31 - 1,
    min_n_features: int = 20,
    max_n_features: int = 100,
    normalize: bool = False,
    rng: np.random.Generator = None,
) -> csr_matrix:
    """
    Randomly generate sparse vectors for testing

    Parameters
    ----------
    n_samples : int
        Number of sparse vectors to create
    sparse_dim : int, default = 2\*\*31 - 1
        Dimensionality of the sparse vectors.
    min_n_features : int, default = 10
        Minimum number of features any given sparse vector may have
    max_n_features : int, default = 100
        Maximum number of features any given sparse vector may have
    normalize : bool, default = False
        If true, normalizes the sparse vectors to all have unit length.
    rng : np.random.Generator, default = None
        A numpy random generator. If None, then one is created

    Returns
    -------
    csr_matrix
        Shape is (n_samples, sparse_dim)
    """
    if rng is None:
        rng = np.random.default_rng()

    if sparse_dim >= 2 ** 31:
        raise ValueError(
            f"{sparse_dim=:,} must be below 2**31 due to csr_matrix limits"
        )
    n_features = rng.integers(min_n_features, max_n_features, n_samples)
    data = rng.uniform(1.0, 20.0, np.sum(n_features)).astype(np.float32)
    data = data * rng.choice([-1, 1], np.sum(n_features)).astype(np.float32)
    indices = rng.integers(0, sparse_dim, np.sum(n_features)).astype(np.int32)
    indptr = np.concatenate((np.zeros(1), np.cumsum(n_features))).astype(np.int32)

    # Maybe normalize, but make sure we don't have repeat indices in a given vector
    for i in range(n_samples):
        start_idx = indptr[i]
        end_idx = indptr[i + 1]
        vec_indices = indices[start_idx:end_idx]
        while len(vec_indices) != len(set(vec_indices)):
            indices[start_idx:end_idx] = rng.integers(0, sparse_dim, n_features[i])
            vec_indices = indices[start_idx:end_idx]
        if normalize:
            data[start_idx:end_idx] = data[start_idx:end_idx] / np.linalg.norm(
                data[start_idx:end_idx]
            )

    return csr_matrix((data, indices, indptr), shape=(n_samples, sparse_dim))


class JustInTimeRandomProjection(BaseRandomProjection):
    """
    Reduce the dimensionality of sparse vectors using a dense random projection matrix
    in a memory-efficient way.

    This is an alternative to scikit-learn.random_projection.SparseRandomProjection.
    This implementation provides higher quality embedding and is constant in the amount
    of RAM needed, irrespective of the sparse starting dimension. It achieves this by
    not storing the projection matrix in memory, but instead generates only the needed
    columns just-in-time for any given input vector. This makes the same as a
    SparseRandomProjection when the density = 1.0.

    The elements of the projection matrix, :math:`\pm 1 / \sqrt{n\_components}`, are
    generated as needed using a hashing function to provide a random choice of the
    sign. This uses numba functions to enable faster computation and is parallelized
    across the available cpus.

    This allows you to utilize sparse starting dimension of 2\*\*31 - 1 without any
    issues. For really values of ``n_components`` transforming will also be faster
    than SparseRandomProjection.

    Parameters
    ----------
    n_components : int or 'auto', default = 'auto'
        Dimensionality of the target projection space. If not a multiple of 64, then
        it will be rounded down to the nearest multiple of 64.
        
        If 'auto', then it will pick a dimensionality that satisfies the
        Johnson-Lindenstrauss Lemma based upon the ``eps`` parameter. The
        dimensionality will then be rounded up to the nearest multiple of 64
        to ensure you satisfy the conditions in the lemma.
        
        **NOTE** This can yield a very conservative estimate of the required
        dimensionality.
    eps : float, default = 0.1
        Parameter to control the quality of the embedding according to the
        Johnson-Lindenstrauss Lemma. This is used if ``n_components`` is 'auto'.
        This represents the error rate between distances in the sparse dimension
        and the resulting lower embedding dimension. Smaller values of ``eps``
        give larger values of ``n_components``.
    
    Attributes
    ----------
    n_components_ : int
        Dimensionality of the embedding dimension. It will be a multiple of 64.
    
    Examples
    --------

    ::

        import numpy as np
        from pocket_dimension.random_projection import (
            JustInTimeRandomProjection,
            random_sparse_vectors,
        )
        
        n_components = 256
        sparse_dim = 1_073_741_824
        n_samples = 100_000
        X = random_sparse_vectors(n_samples, sparse_dim=sparse_dim, normalize=True)

        transformer = JustInTimeRandomProjection(n_components)
        # No need to fit if provide a value for n_components at initialization
        X_new = transformer.transform(X)
        X_new.shape
        # (25, 64)

    """

    def __init__(self, n_components="auto", *, eps=0.1):
        super().__init__(
            n_components=n_components,
            eps=eps,
            compute_inverse_components=False,
            random_state=None,
        )
        if isinstance(self.n_components, int):
            self.n_components = max(64, (self.n_components // 64) * 64)
            self.n_components_ = self.n_components
            if self.n_components != n_components:
                warnings.warn(
                    "The number of components has been round down to the nearest "
                    + f"multiple of 64. It's been set to {self.n_components}"
                )
        elif self.n_components != "auto":
            raise ValueError(
                f"{n_components=:}. It must either be a positive integer or 'auto'"
            )

    def fit(self, X, y=None):
        """
        This is essential a no-op function since there is no need to generate a random
        projection matrix. Instead, this is validating input data and if
        ``n_components`` is 'auto', then it sets it according to the
        Johnson-Lindenstrauss lemma limits based upon X.shape[1] and ``eps``.

        Parameters
        ----------
        X : csr_matrix
            Only the shape is used to find the optimal value of ``n_components`` to
            satisfy Johnson-Lindenstrauss lemma theoretical guarantees.
        y : Ignored

        Returns
        -------
        self : object
            JustInTimeRandomProjection class instance
        """
        X = self._validate_data(X, accept_sparse="csr", dtype=[np.float32, np.float64])
        n_samples, n_features = X.shape

        if self.n_components == "auto":
            self.n_components_ = johnson_lindenstrauss_min_dim(
                n_samples=n_samples, eps=self.eps
            )
            self.n_components_ = max(64, (1 + self.n_components_ // 64) * 64)

            if self.n_components_ > n_features:
                raise ValueError(
                    f"eps={self.eps:f} and {n_samples=:} lead to a target dimension "
                    + f"{self.n_components_} which is larger than the original space "
                    + f"with {n_features=:}"
                )
        elif self.n_components > n_features:
            warnings.warn(
                "The number of components is higher than the number of features: "
                + f"n_features < n_components ({n_features} < {self.n_components}. "
                + "The dimensionality of the problem will not be reduced.",
                DataDimensionalityWarning,
            )

        return self

    def transform(self, X: csr_matrix) -> np.ndarray:
        """
        Project the data using the just-in-time generated dense random projection
        matrix.

        Parameters
        ----------
        X : csr_matrix
            The input data to project into a smaller dimensional space. Shape is
            (n_samples, sparse_dim)
        
        Returns
        -------
        np.ndarray
            Projected data of shape (n_samples, n_components)
        """
        check_is_fitted(self)
        X = self._validate_data(
            X, accept_sparse="csr", reset=False, dtype=[np.float32, np.float64]
        )
        if not isinstance(X, csr_matrix):
            raise TypeError(f"X must be a sparse csr_matrix. You gave {type(X)}")

        input_dtype = X.data.dtype

        return random_projection(
            X.data.astype(np.float32),
            X.indices.astype(np.int64),
            X.indptr.astype(np.int64),
            self.n_components_,
        ).astype(input_dtype)

    def _make_random_matrix(self, n_components, n_features):
        """
        This is an abstract method in the base class, so it must be overwritten, but
        it is not need, so returning an empty numpy array.

        Parameters
        ----------
        n_components : int
            The embedding dimension
        n_features : int
            The sparse starting dimension
        
        Returns
        -------
        np.ndarray
            Empty array to conform to BaseRandomProjection, the parent class
        """
        return np.array([])
