.. Pocket Dimension documentation master file, created by
   sphinx-quickstart on Thu Sep 22 21:56:36 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

.. include:: ../../README.rst

Modules
=======

Pocket Dimension
----------------

Contains the Numba implementation of the random projection function and the TFVectorizer,
TFIDFVectorizer, and BM25Vectorizer classes that convert sparse term-weighted records into
dense vectors. The vectorizers also support count-min sketch inputs via ``cms_file``, either
from disk or from an already instantiated sketch object.

.. toctree::
   :maxdepth: 2

   pocket_dimension

.. toctree::
   :maxdepth: 2
   :caption: API

   modules

.. toctree::
   :maxdepth: 2
   :caption: Admin

   license
   help


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
