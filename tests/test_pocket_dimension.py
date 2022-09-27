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
import numpy as np
from pybloomfilter import BloomFilter
from pytest import approx
from sketchnu.countmin import CountMin
from scipy.sparse import csr_matrix
from sklearn.random_projection import johnson_lindenstrauss_min_dim
from typing import Tuple


from pocket_dimension.random_projection import (
    random_sparse_vectors,
    random_projection,
    distributional_johnson_lindenstrauss_optimal_delta,
)
from pocket_dimension.vectorizer import (
    numba_idf,
    TFVectorizer,
    TFIDFVectorizer,
)


def sparse_distance_squared(S1: csr_matrix, S2: csr_matrix) -> float:
    """
    Calculate the distance between two sparse vectors
    """
    total = 0.0
    for idx, value in zip(S1.indices, S1.data):
        if idx in S2.indices:
            total += (value - S2.data[idx]) ** 2
        else:
            total += value ** 2
    for idx, value in zip(S2.indices, S2.data):
        if idx in S1.indices:
            total += 0.0  # already accounted for above
        else:
            total += value ** 2
    return total


def test_johnson_lindenstrauss(
    n: int = 40, min_n_features: int = 5, max_n_features: int = 100, eps: float = 0.05,
):
    """
    Test the the Johnson-Lindenstrauss Lemma holds between the sparse vectors and the
    randomly projected dense vectors. This uses the results found in 

    https://cs.stanford.edu/people/mmahoney/cs369m/Lectures/lecture1.pdf

    This checkes that the ratio of the distance between two vectors in the embedding
    to the distance between the original sparse vectors is between (1-eps) and (1+eps).
    The embedding dimension is determined by the requested error rate, eps.

    d = 24 \* log(n) / (3 \* eps\*\*2 - 2 \* eps \*\* 3)

    where n is the number of vectors.  This does the full n(n-1)/2 comparisions.

    Parameters
    ----------
    n : int, optional
        Number of sparse vectors to randomly generate. Default is 40
    min_n_features : int, optional
        Minimum number of features any given sparse vector may have. Default is 5
    max_n_features : int, optional
        Maximum number of features any given sparse vector may have. Default is 100
    eps : float, optional
        Desired error rate. Default is 0.05
    """
    # Johnson-Lindenstrauss Lemma says that the embedding dimension must be
    # at least bigger than this value.
    d = johnson_lindenstrauss_min_dim(n, eps=eps)
    # random_project needs multiple of 64. So make sure we satisfy minimum
    # requirement (hence the + 1)
    d = int(d // 64 + 1) * 64

    S = random_sparse_vectors(
        n, min_n_features=min_n_features, max_n_features=max_n_features
    )
    X = random_projection(
        S.data.astype(np.float32),
        S.indices.astype(np.int64),
        S.indptr.astype(np.int64),
        d,
    )

    for i in range(0, X.shape[0] - 1):
        for j in range(i + 1, X.shape[0]):
            sparse_distance2 = sparse_distance_squared(S[i], S[j])
            dense_distance2 = np.sum(np.square(X[i] - X[j]))
            ratio = dense_distance2 / sparse_distance2
            assert (1 - eps) <= ratio and ratio <= (
                1 + eps
            ), f"{ratio=:.4f} is outside of [{1-eps},{1+eps}]"


def test_distributional_johnson_lindenstrauss(
    n: int = 1000,
    min_n_features: int = 5,
    max_n_features: int = 100,
    eps: float = 0.1,
    delta_pad: float = 0.005,
):
    """
    Test the statistical guarantees of the Distributional Johnson-Lindenstrauss Lemma
    using a method for finding the best possible delta (failure rate) for a given
    epsilon (error rate) of the difference of the L2 norm between vectors in the
    original dimension (x) and the lower embedding dimension (Ax) described in

    M. Skorski. *Johnson-Lindenstrauss Transforms with Best Confidence*,
    Proceedings of Machine Learning Research **134**, 1 (2021)
    http://proceedings.mlr.press/v134/skorski21a/skorski21a.pdf

    This paper provides the optimal possible error probability

    delta(sparse_dim, embed_dim, eps) =
            P[abs(\|Ax\|\*\*2 - \|x\|\*\*2) > eps * \|x\|\*\*2]

    given the best possible matrix A and the worst possible data x.

    Parameters
    ----------
    n : int, optional
        Number of data points. Default is 1,000
    min_n_features : int, optional
        Minimum number of features any given vector may have. Default is 5
    max_n_features : int, optional
        Maximum number of features any given vector may have. Default is 100
    eps : float, optional
        Error rate. Default is 0.1
    delta_pad : float, optional
        Amount by which we pad the calculated best delta value. This is done to reduce
        the probability of failure given the vagaries of running statistical tests.
        Default is 0.005
    """
    S = random_sparse_vectors(
        n, min_n_features=min_n_features, max_n_features=max_n_features
    )

    for d in [64, 128, 256, 512, 1024, 2048]:
        X = random_projection(
            S.data.astype(np.float32),
            S.indices.astype(np.int64),
            S.indptr.astype(np.int64),
            d,
        )
        delta = (
            distributional_johnson_lindenstrauss_optimal_delta(2 ** 31 - 1, d, eps)
            + delta_pad
        )
        metric = np.zeros(n)
        for i in range(n):
            x2 = np.sum(np.square(X[i]))
            s2 = np.sum(np.square(S[i].data))
            metric[i] = np.abs(x2 / s2 - 1)
        failure_prob = metric[metric > eps].shape[0] / metric.shape[0]
        assert failure_prob <= delta, f"{failure_prob=:.4f} exceeded {delta=:.4f}"


def test_tf(d: int = 64):
    """
    Basic test that things look right. 

    Parameters
    ----------
    d : int, optional
        Embedding dimension. Default is 64
    """
    records = [
        {"id": "one", "features": [b"a", b"b", b"c"], "counts": [1, 2, 3]},
        {"id": "two", "features": [b"a", b"b", b"c"], "counts": [1, 2, 3]},
        {"id": "three", "features": [b"1", b"2", b"3"], "counts": [1, 2, 3]},
    ]

    embedder = TFVectorizer(d)
    X, ids = embedder(records)

    assert X.shape == (3, d)
    assert X.dtype == np.float32
    cosine_0_2 = X[0].dot(X[1])
    assert cosine_0_2 == approx(1.0), f"{cosine_0_2=:.5f} should be 1.0"
    assert ids.shape == (3,)
    assert ids[0] == "one"
    assert ids[1] == "two"
    assert ids[2] == "three"
    cosine_1_3 = X[0].dot(X[2])
    assert cosine_1_3 < 0.25, f"{cosine_1_3=:.5f} should be small"


def test_tf_filter(tmp_path, d: int = 64):
    """
    Testing the bloom filter & filter_out parameters for the TFVectorizer
    """
    records = [
        {
            "id": "one",
            "features": [b"a", b"b", b"c", b"d", b"e"],
            "counts": [1, 2, 3, 4, 5],
        },
        {"id": "two", "features": [b"a", b"b", b"c", b"d"], "counts": [1, 2, 3, 4]},
    ]

    """
    Test that with no filtering, the records have cosine != 1.0
    """
    embedder_nofilter = TFVectorizer(d)
    X, _ = embedder_nofilter(records)
    assert X[0].dot(X[1]) != approx(1.0)

    """
    Test that filtering out the "e" features, that the two vectors now have cosine = 1
    """
    # Add the feature b"e" to the filter.
    bloom_file = str(tmp_path / "tf_filter_out.bloom")
    bloom_filter = BloomFilter(200, 0.0001, bloom_file)
    bloom_filter.add(b"e")
    bloom_filter.close()

    embedder = TFVectorizer(d, filter=bloom_file, filter_out=True)
    X, _ = embedder(records)
    assert X[0].dot(X[1]) == approx(1.0), "'e' is filtered out. cosine should = 1.0"

    """
    Test that vectors are built only with features that are in the bloom filter.
    This will use different set of records, but the same bloom filter.
    """
    # It should already have b"e" in it.
    bloom_filter = BloomFilter.open(bloom_file, "rw")
    bloom_filter.update([b"d", b"c", b"b"])
    bloom_filter.close()

    # Only the b"a" & b"f" values are not in the filter and thus should be
    # removed before the embedding
    records = [
        {
            "id": "one",
            "features": [b"a", b"b", b"c", b"d", b"e"],
            "counts": [5, 4, 3, 2, 1],
        },
        {
            "id": "two",
            "features": [b"a", b"b", b"c", b"d", b"e", b"f"],
            "counts": [15, 4, 3, 2, 1, 40],
        },
    ]
    embedder = TFVectorizer(d, filter=bloom_file, filter_out=False)
    X, _ = embedder(records)
    assert X[0].dot(X[1]) == approx(1.0)


def test_tf_cms(tmp_path, d: int = 64):
    """
    Test that using a count-min sketch for filtering works properly
    """
    records = [
        {
            "id": "one",
            "features": [b"a", b"b", b"c", b"d", b"e"],
            "counts": [1, 2, 3, 4, 5],
        },
        {
            "id": "two",
            "features": [b"a", b"b", b"c", b"d", b"f"],
            "counts": [1, 2, 3, 4, 30],
        },
        {"id": "three", "features": [b"a", b"b", b"c", b"d"], "counts": [1, 2, 3, 4]},
    ]
    cms_file = str(tmp_path / "tf_cms.npz")
    cms = CountMin("linear", width=300)
    cms.update([b"a"] * 5)
    cms.update([b"b"] * 3)  # Matches minDF=3, should be included
    cms.update([b"c"] * 10)
    cms.update([b"d"] * 100)  # Matches maxDF=100, should be included
    cms.update([b"e"] * 500)  # Will exceed maxDF=100
    cms.update([b"f"] * 2)  # Below minDF=3
    cms.save(cms_file)

    """
    Test that with no filtering, the records have cosine != 1.0
    """
    embedder_nofilter = TFVectorizer(d)
    X_nofilter, _ = embedder_nofilter(records)
    for i in range(2):
        assert X_nofilter[i].dot(X_nofilter[2]) != approx(1.0)

    """
    Test that we drop "e" and "f" from records one & two. They should now match
    the unfiltered three record.
    """
    embedder = TFVectorizer(d, cms_file=cms_file, minDF=3, maxDF=100)
    X, _ = embedder(records)
    for i in range(2):
        assert X[i].dot(X_nofilter[2]) == approx(1.0), f"{i} was not filtered"


def test_tf_temp(d: int = 64):
    """
    Test that changing the temperature used during embedding changes the cosine in an
    expected way
    """
    records = [
        {
            "id": "one",
            "features": [b"a", b"b", b"c", b"d", b"e"],
            "counts": [1, 2, 3, 4, 15],
        },
        {
            "id": "two",
            "features": [b"a", b"b", b"c", b"d", b"e"],
            "counts": [15, 4, 3, 2, 1],
        },
        {
            "id": "three",
            "features": [b"a", b"b", b"c", b"d", b"e"],
            "counts": [1, 1, 1, 1, 1],
        },
    ]
    """
    Testing that using infinite temperature squashes all values to 1.0 and thus
    X_0 and X_1 should match X_2.
    """
    embedder_inf = TFVectorizer(d, temperature=np.inf)
    X, _ = embedder_inf(records)
    cosines = [0.0, 0.0, 0.0]
    for i in range(2):
        cosines[i] = X[i].dot(X[2])
        assert cosines[i] == approx(1.0), f"{i} failed to have cosine~1.0"
    cosines[2] = X[0].dot(X[1])

    """
    Test that as we lower the temperature the cosine decreases between the first two
    vectors and the third reference original vector (all ones) and also between just
    the first two vectors
    """
    for temp in [3.0, 2.0, 1.5, 1.0, 0.8]:
        embedder = TFVectorizer(d, temperature=temp)
        Y, _ = embedder(records)
        for i in range(2):
            cosine = Y[i].dot(X[2])
            assert (
                cosine < cosines[i]
            ), f"{temp} {i} has {cosine=:} less than previous {cosines[i]}"
            cosines[i] = cosine
        # Cosine between the first two vectors should also get worse as temp goes down
        cosine = Y[0].dot(Y[1])
        assert cosine < cosines[2]
        cosines[2] = cosine


def test_numba_idf():
    """
    Test that the numba function numba_idf works as expected
    """
    doc_freq = 100
    n_records = 150

    idf_py = np.log10(n_records / (doc_freq + 1))
    idf = numba_idf(doc_freq, n_records)

    assert idf_py == approx(idf)


def test_tfidf(tmp_path, d: int = 64):
    """
    Test that a tfidf vectorization causes one feature's tfidf to be zero and match
    a vector that does not have this feature
    """
    records = [
        {
            "id": "one",
            "features": [b"a", b"b", b"c", b"d", b"e"],
            "counts": [1, 2, 3, 4, 5],
        },
        {
            "id": "two",
            "features": [b"a", b"b", b"c", b"d", b"e"],
            "counts": [1, 2, 3, 4, 5],
        },
        {"id": "three", "features": [b"a", b"b", b"c", b"d"], "counts": [1, 2, 3, 4],},
    ]
    cms_file = str(tmp_path / "tf_cms.npz")
    cms = CountMin("linear", width=300)
    cms.n_added_records[1] = 500
    cms.update([b"a"] * 3)
    cms.update([b"b"] * 5)
    cms.update([b"c"] * 10)
    cms.update([b"d"] * 100)
    cms.update([b"e"] * 500)
    cms.save(cms_file)

    """
    Test that with no filtering, the records have cosine != 1.0
    """
    embedder = TFIDFVectorizer(d, cms_file=cms_file)
    X, _ = embedder(records)

    # Now the b"e" should contribute nothing to the vector since idf = 0.0
    assert X[0].dot(X[2]) == approx(1.0)
