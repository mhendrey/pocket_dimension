.. image:: ../../images/logo.png
    :align: center
    :alt: A small spiral galaxy inside a small glass sphere

==================================

Pocket Dimension provides a memory-efficient, dense, random projection of sparse vectors. This
random projection is the used to be able to take records {"id": str, "features": List[bytes],
"counts": List[int]}, convert them into sparse random vectors using scikit-learn's FeatureHasher,
and then project them down to lower dimensional dense vectors.

When the very large sparse universe becomes too inhospitable, escape into a cozy pocket dimension.

Documentation
=============
Documentation for the API and theoretical foundations of the algorithms can be
found at https://mhendrey.github.io/pocket_dimension

Installation
============
Pocket Dimension may be install using pip::

    pip install pocket_dimension

I'm working on a conda-forge version, but this uses pybloomfiltermmap3 which is currently only on PyPi.
