.. image:: ../../images/logo.png
    :align: center
    :alt: A small spiral galaxy inside a small glass sphere

==================================

Pocket Dimension provides a memory-efficient, dense, random projection of sparse vectors. This
random projection is used to take records {"id": str, "features": List[bytes],
"counts": List[int]}, convert them into sparse random vectors using scikit-learn's FeatureHasher,
and then project them down to lower dimensional dense vectors.

Pocket Dimension includes three vectorizer classes for this workflow:

* ``TFVectorizer`` for term-frequency vectors
* ``TFIDFVectorizer`` for term-frequency inverse-document-frequency vectors
* ``BM25Vectorizer`` for BM25-style scoring and ranking-friendly vector weights

The vectorizers accept a count-min sketch via the ``cms_file`` argument, which may be a path to a saved sketch or an already instantiated sketch object. This keeps older code paths working while allowing in-memory reuse of a shared sketch.

When the very large sparse universe becomes too inhospitable, escape into a cozy pocket dimension.

Quick example
=============

::

    from sketchnu.countmin import CountMin
    from pocket_dimension.vectorizer import BM25Vectorizer

    records = [
        {"id": "one", "features": [b"apple", b"banana"], "counts": [2, 1]},
        {"id": "two", "features": [b"apple", b"cherry"], "counts": [1, 3]},
    ]

    cms = CountMin("linear", width=200)
    for rec in records:
        for feature, count in zip(rec["features"], rec["counts"]):
            cms.add(feature) # BM25 uses the number of documents a term appears in, not the total count
        cms.n_added_records[1] += 1  # increment the number of 'documents' added to the sketch

    embedder = BM25Vectorizer(128, cms_file=cms)
    X, ids = embedder(records)

Documentation
=============
Documentation for the API and theoretical foundations of the algorithms can be
found at https://mhendrey.github.io/pocket_dimension

Installation
============
Pocket Dimension may be install using pip::

    pip install pocket_dimension

I'm working on a conda-forge version, but this uses pybloomfiltermmap3 which is currently only on PyPi.
