Pocket Dimension
================

The Basics
----------

Pocket Dimension provides a memory-efficient, dense, random projection of sparse vectors from very high
dimensions (~2.1 billion) down to much lower dimension dense vectors (~256). It does this by not storing
the entire (2.1 billion x 256) projection in memory. Instead it calculates on the columns it needs on the
fly based upon the non-zero elements of any given sparse vector. This function is implemented in Numba to
speed up the calculations.

Besides the random projection function, Pocket Dimension comes with two classes that can create either
Term-Frequency or Term-Frequency, Inverse Document Frequency vectors into dense vectors from starting
records.

Usage
-----

::

    import numpy as np
    from pocket_dimension.pocket_dimension import TFVectorizer

    # Make some data. "one" and "two" should be similar & "three" should be different
    records = [
        {"id": "one", "features": [b"one", b"two", b"three"], "counts": [1, 2, 3]},
        {"id": "two", "features": [b"one", b"two", b"three"], "counts": [2, 3, 4]},
        {"id": "three", "features": [b"ab", b"cd", b"efghi"], "counts": [15, 1, 3]},
    ]

    # Create the TFVectorizer. Let's project down to 128-d
    embedder = TFVectorizer(128)
    X, ids = embedder(records)

    # let's check cosine similarity
    cosine_one_two = X[0].dot(X[1])
    cosine_one_three = X[0].dot(X[2])
    print(f"Vectors 'one' and 'two' have cosine similarity = {cosine_one_two:.4f}")
    print(f"Vectors 'one' and 'three'have cosine similarity = {cosine_one_three:.4f}")


The Details
-----------
Random projection is a dimension reduction technique that has some mathematical guarantees thanks to the
Johnson-Lindenstrauss Lemma, though in practice it is common to get good results even if you you blow past
the mathematical guarantees. Their are two different implementations within scikit-learn.random_projection,
GaussianRandomProjection and SparseRandomProjection. Unfortunately, these both use dense matrix representations
for the projection matrix. So if you want to randomly project a 2B dimensional sparse vector down to 128
dimensions, then you will quickly run out of RAM.

If your vectors are sparse, then there are 
