"""
Pocket Dimension provides a memory-efficient, dense, random projection of sparse vectors
and then applies this to Term Frequency (TF) and Term Frequency, Inverse Document
Frequency (TFIDF) data.

Copyright (C) 2022 Matthew Hendrey & Brendan Murphy

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

from numba import njit, prange, float32, int64, uint64
import numpy as np
import logging
from pathlib import Path
from pybloomfilter import BloomFilter
from scipy.optimize import minimize
from scipy.stats import beta
from sketchnu.countmin import load as load_cms, CountMin
from sklearn.feature_extraction import FeatureHasher
from typing import Dict, Iterable, List, Tuple, Union

logger = logging.getLogger("pocket_dimension")


def best_delta(orig_dim: int, embed_dim: int, eps: float) -> float:
    """
    Algorithm to find the lowest failure rate, delta, for a given eps (error rate),
    orig_dim (starting dimension), and embed_dim (embedding dimension) for
    the Distributional Johnson-Lindenstrauss Lemma. Taken from

    M. Skorski. *Johnson-Lindenstrauss Transforms with Best Confidence*,
    Proceedings of Machine Learning Research **134**, 1 (2021)

    which can be found at http://proceedings.mlr.press/v134/skorski21a/skorski21a.pdf

    Parameters
    ----------
    orig_dim : int
        Original dimension of the data
    embed_dim : int
        Embedding dimension
    eps : float
        Error rate
    
    Returns
    -------
    delta : float
        The best probability of failure that you can expect
    """
    a = embed_dim / 2.0
    b = (orig_dim - embed_dim) / 2.0
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
    Randomly project a sparse matrix (standard CSR representation) to a dense vector by
    effectively multiplying by a random matrix whose elements are -1 / sqrt(`d`) or
    1 / sqrt(`d`). The projection matrix is never stored in memory. Instead the elements
    are generated as needed using splitmix64(). **Note** Embedding dimension, `d`, is
    rounded to a multiple of 64 since we generate random bits in batches of 64.

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


@njit(float32(float32, float32, float32, float32))
def transform_counts_tfidf(c, one_over_temp, doc_freq, n_records):
    """
    Numba function to calculate the tfidf value for a given feature

    Parameters
    ----------
    c : float32
        Raw counts of the term frequency
    one_over_temp : float32
        Raise the raw counts, `c`, to 1.0 / temperature. I higher temperature flattens
        the raw counts of the term frequency relative to each other.
    doc_freq : float32
        Number of records that contain this particular feature.
    n_records : float32
        Total number of records in the data set.
    
    Returns
    -------
    float32
        The scaled value for the TF-IDF for this particular feature
    """
    c = c ** one_over_temp
    idf = np.log10(max(float32(1.0), n_records / (float32(1.0) + doc_freq)))
    return c * idf


class TFVectorizer:
    """
    Randomly project records by first applying a high-dimensional feature hasher
    to create a sparse vector representation of the term-frequency and then applying
    the random projection to a dense embedding dimension.

    Parameters
    ----------
    d : int
        Dimension of the dense vector output. Gets converted to multiple of 64 if
        not already
    cms_file : str | Path, optional
        Filename of a saved count-min sketch to use for filtering out features
        whose document frequency falls outside of [minDF, maxDF]. Default is None
    hash_dim : int, optional
        Number of dimensions that features get hashed to for the sparse vector
        representation. Should have `hash_dim >> d`. Default is the largest
        possible 2\*\*31 - 1 (~2 billion) for sklearn's FeatureHasher
    minDF : int, optional
        Minimum document frequency (number of records with this feature) a feature
        must have to be included in the embedding. Default is 1
    maxDF : int, optional
        Maximum document frequency (number of records with this feature) a feature
        can have to be included in the embedding. Default is 2\*\*32 - 1.
    temperature : float, optional
        Option to reshape the term frequency vector by raising each count to
        (1/temperature). Using a value above 1.0 flattens the values relative to
        each other. Using a value below 1.0 sharpens the contrast of values
        relative to each other.
    filter : pybloomfilter3, option
        Bloom filter to use for filtering features before creating sparse vectors.
        Default is None
    filter_out : bool, optional
        If True, then remove features that are in the filter before making the
        sparse vector. If False, then only use features that are in the filter in
        the sparse vector. Not used if `filter` is None. Default is True.

    Attributes
    ----------
    d : int
        Embedding dimension of the dense vector. Must be multiple of 64
    hash_dim : int
        Dimension of the sparse vector. Must be <= 2\*\*31-1
    hasher : sklearn.feature_extraction.FeatureHasher
        Hashes input features (bytes) to integer representing corresponding dimension
    temperature : float
        Exponential scale raw counts by 1 / temperature when making TF vector
    cms : sketchnu.CountMinLinear | sketchnu.CountMinLog16 | sketchnu.CountMinLog8
        A count-min sketch that stores document frequency info
    minDF : int
        Minimum document frequency a feature must have to be incuded in the vector
    maxDF : int
        Maximum document frequency a feature can have to be included in the vector
    filter : pybloomfilter.BloomFilter
        BloomFilter containing features to be filtered
    filter_out : bool
        If True, then any feature found in the `filter` will be excluded from the
        vector. If False, then only features found in the `filter` are included in
        vector.
    """

    def __init__(
        self,
        d: int,
        *,
        cms_file: Union[str, Path] = None,
        hash_dim: int = 2 ** 31 - 1,
        minDF: int = 1,
        maxDF: int = 2 ** 32 - 1,
        temperature: float = 1.0,
        filter: str = None,
        filter_out: bool = True,
    ):
        """
        Initialize the a Term-Frequency vectorizer. All optional parameters must be
        passed as keyword arguments. That is, everything but the embedding dimension
        `d`.

        Parameters
        ----------
        d : int
            Dimension of the dense vector output. Gets converted to multiple of 64 if
            not already
        cms_file : str | Path, optional
            Filename of a saved count-min sketch to use for filtering out features
            whose document frequency falls outside of [minDF, maxDF]. Default is None
        hash_dim : int, optional
            Number of dimensions that features get hashed to for the sparse vector
            representation. Should have `hash_dim >> d`. Default is the largest
            possible 2\*\*31 - 1 (~2 billion) for sklearn's FeatureHasher
        minDF : int, optional
            Minimum document frequency (number of records with this feature) a feature
            must have to be included in the embedding. Default is 1
        maxDF : int, optional
            Maximum document frequency (number of records with this feature) a feature
            can have to be included in the embedding. Default is 2\*\*32 - 1.
        temperature : float, optional
            Option to reshape the term frequency vector by raising each count to
            (1/temperature). Using a value above 1.0 flattens the values relative to
            each other. Using a value below 1.0 sharpens the contrast of values
            relative to each other.
        filter : pybloomfilter3, option
            Bloom filter to use for filtering features before creating sparse vectors.
            Default is None
        filter_out : bool, optional
            If True, then remove features that are in the filter before making the
            sparse vector. If False, then only use features that are in the filter in
            the sparse vector. Not used if `filter` is None. Default is True.
        """
        if d % 64 != 0:
            d = 64 * max(1, int(d // 64))
            logger.warning(f"d is not a multiple of 64. Changing it to {d}")
        assert d < hash_dim, f"{d=:} must be less than {hash_dim=:}"
        assert temperature > 0.0, f"{temperature=:} must be positive value"
        assert minDF <= maxDF, f"{minDF} must be <= {maxDF}"

        self.d = d
        self.hash_dim = hash_dim
        self.minDF = minDF
        self.maxDF = maxDF
        self.temperature = temperature
        self.filter_out = filter_out
        self.one_over_temp = 1.0 / temperature
        self.hasher = FeatureHasher(
            hash_dim, input_type="pair", alternate_sign=True, dtype="float32"
        )

        if filter is None:
            self.filter = []
        elif isinstance(filter, str):
            self.filter = BloomFilter.open(filter, "r")
        else:
            raise ValueError(f"filter must be str | None. You gave {type(filter)}")

        if cms_file is None:
            self.cms = CountMin(width=1, depth=1, cms_type="linear")
            self.cms.add(b"anything")
        elif isinstance(cms_file, (str, Path)):
            self.cms = load_cms(cms_file)

    def __call__(self, records: Iterable[Dict]) -> Tuple[np.ndarray, np.ndarray]:
        """
        Transform the records into sparse vectors with the FeatureHasher and then apply
        the random projection.

        Parameters
        ----------
        records: Iterable[Dict]
            Iterable of records where each record has
            {"id": str, "features": List[bytes], "counts": List[int]}
        
        Returns
        -------
        X : np.ndarray, shape = [n_records, d]
            Resulting dense vector representation of the records. Each vector is
            normalized to unit length.
        ids : np.ndarray, shape = [n_records,]
            Array listing the record's id in order to track which row in X corresponds
            with which id.
        """
        if not isinstance(records, Iterable):
            raise TypeError(
                f"records, {type(records)}, must be an instance of Iterable"
            )
        ids = []
        F = self.yield_record(records, ids)
        H = self.hasher.transform(F)
        X = random_projection(
            H.data, H.indices.astype(np.int64), H.indptr.astype(np.int64), self.d
        )
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        return X, np.array(ids)

    def _transform_count(self, c: float, doc_freq: float) -> float:
        """
        Transform an individual count by raising it to 1.0 / temperature.

        Parameters
        ----------
        c : float
            Raw count of a given feature in a record
        doc_freq : float
            The document frequency of this feature. This is not used for TFVectorizer
        """
        return c ** self.one_over_temp

    def yield_features_counts(
        self, features: List[bytes], counts: List[int]
    ) -> Tuple[bytes, float]:
        """
        Yield the individual feature and count where the count is transformed calling
        `_transform_count()`

        Parameters
        ----------
        features : List[bytes]
            List of the individual features. These will be hashed by the FeatureHasher
            to turn into a sparse vector representation
        counts : List[int]
            Number of times that feature is present in a given record.
        
        Yields
        ------
        Tuple[bytes, float]
        """
        for f, c in zip(features, counts):
            doc_freq = self.cms[f]
            if ((f in self.filter) != self.filter_out) and (
                self.minDF <= doc_freq and doc_freq <= self.maxDF
            ):
                yield (f, self._transform_count(c, doc_freq))

    def yield_record(self, records: Iterable[Dict], ids: List):
        """
        Yields a generator of an individual record's features/counts.

        Parameters
        ----------
        records : Iterable[Dict]
            An iterable of records where each records is a dict of
            {"id": str, "features": List[bytes], "counts": List[int]}
        ids : List
            Pass in an empty list which will be returned with corresponding ids from
            the `records` passed in
        
        Yields
        ------
        Generator
            Of the appropriate individual (features,counts) for a given record
        """
        for rec in records:
            ids.append(rec["id"])
            yield self.yield_features_counts(rec["features"], rec["counts"])


class TFIDFVectorizer(TFVectorizer):
    """
    Randomly project records by first applying a high-dimensional feature hasher
    to create a sparse vector representation of the tf-idf and then applying the
    random projection to a dense embedding dimension.

    tf-idf = c**(1/temp) * log10(n_records / (doc_freq + 1))
    """

    def __init__(
        self,
        d: int,
        cms_file: Union[str, Path],
        *,
        hash_dim: int = 2 ** 31 - 1,
        minDF: int = 1,
        maxDF: int = 2 ** 32 - 1,
        temperature: float = 1.0,
        filter: str = None,
        filter_out: bool = True,
    ):
        """
        Initialize a Term-Frequency, Inverse Document-Frequency vectorizer. All
        optional parameters must be passed as keyword arguments. That is, everything
        but `d` and `cms_file`.

        Parameters
        ----------
        d : int
            Dimension of the dense vector output. Gets converted to multiple of 64 if
            not already
        cms_file : str | Path
            Filename of a saved count-min sketch to use for filtering out features
            whose document frequency falls outside of [minDF, maxDF]. Default is None
        hash_dim : int, optional
            Number of dimensions that features get hashed to for the sparse vector
            representation. Should have `hash_dim >> d`. Default is the largest
            possible 2\*\*31 - 1 (~2 billion) for sklearn's FeatureHasher
        minDF : int, optional
            Minimum document frequency (number of records with this feature) a feature
            must have to be included in the embedding. Default is 1
        maxDF : int, optional
            Maximum document frequency (number of records with this feature) a feature
            can have to be included in the embedding. Default is 2\*\*32 - 1.
        temperature : float, optional
            Option to reshape the term frequency vector by raising each count to
            (1/temperature). Using a value above 1.0 flattens the values relative to
            each other. Using a value below 1.0 sharpens the contrast of values
            relative to each other.
        filter : pybloomfilter3, option
            Bloom filter to use for filtering features before creating sparse vectors.
            Default is None
        filter_out : bool, optional
            If True, then remove features that are in the filter before making the
            sparse vector. If False, then only use features that are in the filter in
            the sparse vector. Not used if `filter` is None. Default is True.
        """
        super(TFIDFVectorizer, self).__init__(
            d,
            cms_file=cms_file,
            hash_dim=hash_dim,
            minDF=minDF,
            maxDF=maxDF,
            temperature=temperature,
            filter=filter,
            filter_out=filter_out,
        )

    def _transform_count(self, c: float, doc_freq: float) -> float:
        """
        Transform an individual count raising it to 1.0 / temperature and then
        multiplying by the inverse document frequency.

        Parameters
        ----------
        c : float
            Raw count of a given feature in a record
        doc_freq : float
            The document frequency of this feature. This is not used for TFVectorizer
        """
        return transform_counts_tfidf(
            c, self.one_over_temp, doc_freq, self.cms.n_records()
        )
