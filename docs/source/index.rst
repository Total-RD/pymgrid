.. pymgrid documentation master file, created by
   sphinx-quickstart on Sat Nov 19 12:49:18 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

*********************
pymgrid documentation
*********************

Important Notice
----------------

**The person that has been maintaining pymgrid since 2020 has moved future development to**
`python-microgrid <https://github.com/ahalev/python-microgrid>`__
**for the foreseeable future, a drop in replacement for pymgrid, and pymgrid may not be receiving any future updates.**
**Please open any new issues in python-microgrid. You can also view python-microgrid's documentation**
`at this link <https://python-microgrid.readthedocs.io/en/latest/>`__.

**Version**: |version|

*pymgrid* is a Python library to simulate tertiary control of electrical microgrids. *pymgrid* allows
users to create and customize microgrids of their choosing. These microgrids can then be controlled using a user-defined
algorithm or one of the control algorithms contained in *pymgrid*: rule-based control and model predictive control.

Environments corresponding to the OpenAI-Gym API are also provided, with both continuous and discrete action space
environments available. These environments can be used with your choice of reinforcement learning algorithm to train
a control algorithm.

*pymgrid* attempts to offer the simplest and most intuitive API possible, allowing the user to
focus on their particular application.

See the :doc:`getting_started` section for further information, including instructions on how to
:ref:`install <installation>` the project.

**Useful links**:
`Binary Installers <https://pypi.org/project/pymgrid/>`__ |
`Source Repository <https://github.com/Total-RD/pymgrid>`__


.. note::

   This project is under active development.

Contents
========

.. toctree::
   :maxdepth: 2

   getting_started
   examples/index
   reference/index

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
