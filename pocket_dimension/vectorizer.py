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

from numba import njit, float32
import numpy as np
import logging
from pathlib import Path
from pybloomfilter import BloomFilter
from sketchnu.countmin import load as load_cms, CountMin
from sklearn.feature_extraction import FeatureHasher
from typing import Dict, Iterable, List, Tuple, Union

from .random_projection import JustInTimeRandomProjection

logger = logging.getLogger("pocket_dimension")


@njit(float32(float32, float32))
def numba_idf(doc_freq, n_records):
    """
    Numba function to calculate the idf value for a ``doc_freq``

    Parameters
    ----------
    doc_freq : float32
        Number of records that contain this particular feature.
    n_records : float32
        Total number of records in the data set.
    
    Returns
    -------
    float32
    """
    idf = np.log10(max(float32(1.0), n_records / (float32(1.0) + doc_freq)))
    return idf


class TFVectorizer:
    """
    Calculate a Term-Frequency vector representation by first using sklearn's
    FeatureHasher to create a high-dimensional, sparse vector representation.
    Then apply the JustInTimeRandomProjection to return a lower-dimensional
    dense vector representation. The resulting vectors will be normalized to unit
    length.

    Parameters
    ----------
    d : int
        Dimension of the dense vector output. Rounded down to nearest multiple of 64 if
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
    jit_rp : JustInTimeRandomProjection
        Efficiently projects high-dimensional sparse vectors down to lower dimensional
        dense vectors
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
        self.jit_rp = JustInTimeRandomProjection(n_components=self.d)

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
        the random projection. The vectors are then normalized to unit length.

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
        X = self.jit_rp.transform(H)
        X = X / np.linalg.norm(X, axis=1, keepdims=True)

        return X, np.array(ids)

    def _idf(self, doc_freq: float) -> float:
        """
        Return the inverse document frequency given ``doc_freq``

        **NOTE** For TFVectorizer, this simply returns 1.0. Included here for
        compatability with TFIDFVectorizer

        Parameters
        ----------
        doc_freq : float
            A document frequency
        
        Returns
        -------
        idf : float
            The inverse document frequency
        """
        return 1.0

    def yield_features_counts(
        self, features: List[bytes], counts: List[int]
    ) -> Tuple[bytes, float]:
        """
        Yield the individual feature and count where the count is scaled by raising it
        to :math:`1/temperature` and then multiplied by the idf value. For TFVectorizer
        the idf = 1.0.

        If ``minDF` and ``maxDF`` or a ``filter`` is provided, then the individual
        features will be filtered appropriately. Only those that pass the filters will
        contribute to the final vector.

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
        if self.temperature != 1.0:
            counts = np.array(counts) ** self.one_over_temp

        for f, c in zip(features, counts):
            doc_freq = self.cms[f]
            if ((f in self.filter) != self.filter_out) and (
                self.minDF <= doc_freq and doc_freq <= self.maxDF
            ):
                yield (f, c * self._idf(doc_freq))

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

    .. math::

        tfidf = c^{1/temp} \log{10}(n\_records / (doc\_freq + 1))
    
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

    def _idf(self, doc_freq: float) -> float:
        """
        Return the inverse document frequency given ``doc_freq``. To speed things up
        this calls a numba function.

        :math:`idf = \log_{10}(\max(1.0, N / (doc\_freq + 1.0)))`

        where :math:`N` is the total number of documents (or records in this case). The
        max is used because we are using a count-min sketch to estimate the document
        frequencies which may result in the denominator be bigger than N.

        Parameters
        ----------
        doc_freq : float
            A document frequency
        
        Returns
        -------
        idf : float
            The inverse document frequency
        """
        return numba_idf(doc_freq, self.cms.n_records())
