.. _api.control:

Control Algorithms
==================

.. currentmodule:: pymgrid.algos

Control algorithms built into pymgrid, as well as references for external algorithms that can be deployed

Rule Based Control
------------------

Heuristic Algorithm that deploys modules via a priority list.

.. autosummary::
    :toctree: ../api/algos/

    RuleBasedControl

Model Predictive Control
------------------------

Algorithm that depends on a future forecast as well as a model of state transitions to determine optimal controls.


.. autosummary::
    :toctree: ../api/algos/

    ModelPredictiveControl

Reinforcement Learning
----------------------

Algorithms that treat a microgrid as a Markov process, and train a black-box policy by repeated interactions with
the environment. See :ref:`examples/rl-example` for an example of using reinforcement learning to train such an
algorithm.
