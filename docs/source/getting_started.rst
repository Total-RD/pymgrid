Getting Started
===============

.. _installation:

Installation
------------

The easiest way to install *pymgrid* is with pip:

.. code-block:: console

    $ pip install -U pymgrid

Alternatively, you can install from source. First clone the repo:

.. code-block:: bash

    $ git clone https://github.com/Total-RD/pymgrid.git

Then navigate to the root directory of pymgrid and call

.. code-block:: bash

    $ pip install .

Advanced Installation
---------------------

To use the included model predictive control algorithm <link> on microgrids containing gensets,
additional dependencies are required as the optimization problem becomes mixed integer.

The packages MOSEK and CVXOPT can both handle this case; you can install both by calling

.. code-block:: bash

    $ pip install pymgrid[genset_mpc]

Note that MOSEK requires a license; see https://www.mosek.com/ for details.
Academic and trial licenses are available.

Simple Example
--------------
See :doc:`examples/quick-start` for a simple example to get started.
