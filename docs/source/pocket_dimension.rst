Pocket Dimension
================

The Basics
----------

Pocket Dimension provides a memory-efficient, dense, random projection of sparse
vectors from potentially very high dimensions (~2.1 billion) down to much lower
dimension dense vectors (~256). It does this by not storing the entire
(2.1 billion x 256) projection matrix in memory. Instead it calculates only the columns
it needs on the fly based upon the non-zero elements of any given sparse vector. This
function is implemented in Numba to speed things up.

Besides the random projection function, Pocket Dimension includes a small family of
vectorizers that convert record-level sparse feature counts into dense vectors:

* ``TFVectorizer`` for term-frequency weighting
* ``TFIDFVectorizer`` for term-frequency inverse-document-frequency weighting
* ``BM25Vectorizer`` for BM25-style weighting used in ranking-heavy retrieval tasks

The vectorizers all accept a count-min sketch via ``cms_file``. This can be a path to a
saved sketch on disk or an already instantiated sketch object, which is kept for backward
compatibility and allows a shared sketch to be reused in memory.

Usage
-----

::

    import numpy as np
    from pocket_dimension.vectorizer import TFVectorizer, BM25Vectorizer
    from sketchnu.countmin import CountMin

    # Make some data. "one" and "two" should be similar & "abc" should be different
    records = [
        {"id": "one", "features": [b"one", b"two", b"three"], "counts": [1, 2, 3]},
        {"id": "two", "features": [b"one", b"two", b"three"], "counts": [2, 3, 4]},
        {"id": "abc", "features": [b"abc", b"cde", b"efghi"], "counts": [9, 1, 3]},
    ]

    # Create the TFVectorizer. Let's project down to 128-d
    embedder = TFVectorizer(128)
    X, ids = embedder(records)

    # let's check cosine similarity
    cosine_one_two = X[0].dot(X[1])
    cosine_one_abc = X[0].dot(X[2])
    print(f"Vectors 'one' and 'two' have cosine similarity = {cosine_one_two:.4f}")
    print(f"Vectors 'one' and 'abc' have cosine similarity = {cosine_one_abc:.4f}")
    # Vectors 'one' and 'two' have cosine similarity = 0.9926
    # Vectors 'one' and 'abc' have cosine similarity = -0.0177

    # BM25 uses a count-min sketch to estimate document frequencies and corpus statistics.
    cms = CountMin("linear", width=4096)
    for rec in records:
        for feature, count in zip(rec["features"], rec["counts"]):
            cms.add(feature)  # BM25 uses the number of documents a term appears in, not the total count
        cms.n_added_records[1] += 1  # increment the number of 'documents' added to the sketch

    bm25 = BM25Vectorizer(128, cms_file=cms)
    X_bm25, ids_bm25 = bm25(records)

The BM25 vectorizer is especially useful when you want a ranking-oriented weighting that
accounts for per-document length normalization as well as inverse document frequency.
In API usage, ``cms_file`` may be provided either as a saved sketch path or as an in-memory
``CountMinLinear`` instance (or subclass such as ``CountMinLog8`` / ``CountMinLog16``).

The Details
-----------
Random projection is a dimension reduction technique that has some mathematical
guarantees thanks to the Johnson-Lindenstrauss Lemma, though in practice it is common
to get good results even if you blow past the mathematical guarantees to lower
dimensions. There are two different implementations within
`scikit-learn.random_projection`, `GaussianRandomProjection` and
`SparseRandomProjection`. The `GaussianRandomProjection` will make a dense projection
matrix which will quickly exhaust RAM if the one dimension is large. The
`SparseRandomProjection` will use a sparse matrix, but many of the values will
necessarily be zero which can affect the quality of the projection. Pocket Dimension
will generate only the needed columns of the projection matrix, but will make entries
for every row in that column.  So it is a combination of the two. Pocket Dimension
leverages a hash function to effectively create random, but repeatable,
:math:`\pm1/\sqrt d` entries which are scaled appropriately to preserve the magnitude
of the vector and :math:`d` is the embedding dimension.

If your vectors are sparse, then it is quick to calculate. The sparser the vector, the
faster the processing.

The Johnson-Lindenstrauss Lemma states that the distance squared between any pair of
vectors after they have been randomly projected down to a smaller dimension will be
within a multiple of :math:`1\pm\epsilon` of the original distance squared.

.. math::

    (1-\epsilon)\lVert \mathbf{u}-\mathbf{v} \rVert ^2 \leq \lVert \mathbf{A}\mathbf{u}-\mathbf{A}\mathbf{v} \rVert ^2 \leq (1+\epsilon)\lVert \mathbf{u}-\mathbf{v} \rVert^2

where :math:`\mathbf{A}` is the random projection matrix.
