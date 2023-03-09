.. _acquisition:

Acquisition functions
=====================

Analytical acquisition functions
--------------------------------

.. _ei:

.. math::
    a_{EI} (\textbf{x}, y^*) = (\mu(\textbf{x}) - y^*) \Phi(z) + \sigma(\textbf{x}) \phi(z)

.. math::
    \text{where } z = \frac{\mu(\textbf{x}) - y^*}{\sigma(\textbf{x})}


.. math::
    a_{UCB} (\textbf{x}, \beta) = \mu(\textbf{x}) + \sqrt{\beta} \sigma(\textbf{x})

Monte Carlo acquisition functions
---------------------------------

.. math::
    a_{EI} = \max(ReLU(\mu(\textbf{x}) + \textbf{Lz} - y^*))

.. math::
    a_{UCB} = \max(\mu(\textbf{x}) + \sqrt{\frac{\beta \pi}{2}} | \textbf{Lz} |)