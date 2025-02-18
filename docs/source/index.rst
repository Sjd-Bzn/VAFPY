AFQMC documentation
===================

.. toctree::
    :maxdepth: 1
    :caption: Contents:

    constants
    determinant
    field
    logging
    propagate
    weight

Theory
------

We start from the Hamiltonian in second quantization

.. math::

   \hat H = \sum_{pq} h_{pq}^{(1)} \hat a_p^\dagger \hat a_q + \frac12 \sum_{pqrs} V_{pqrs}
      \hat a_p^\dagger \hat a_q^\dagger \hat a_s \hat a_r

where :math:`h^{(1)}_{pq}` and :math:`v_{pqrs}` are matrix elements of the one- and two-particle
Hamiltonian, respectively. :math:`\hat a_p^\dagger` and :math:`\hat a_p` are Fermionic
creation and annihilation operators.
We use split the Coulomb integral into

.. math::

   V_{pqrs} = \sum_g \mathcal L_{prg} \mathcal L^\ast_{sqg}

and then combine them to even and odd terms

.. math::

   L_{prg}^\text{e} &= \frac{1}{2} \bigl(\mathcal L_{prg} + \mathcal L^\ast_{rpg})
   \\
   L_{prg}^\text{o} &= \frac{i}{2} \bigl(\mathcal L_{prg} - \mathcal L^\ast_{rpg})

to rewrite it as

.. math::

   V_{pqrs} = \sum_g (L_{prg}^\text{e} L_{qsg}^\text{e} + L_{prg}^\text{o} L_{qsg}^\text{o}\bigr)
   =: \sum_g L^\text{full}_{prg} L^\text{full}_{qsg}

where the sum in over the new :math:`L^\text{full}_{prg}` runs over twice as many elements.
We can compress this expression using a singular-value decomposition

.. math::

   L^\text{full}_{prg} &= \sum_{g'} U_{prg'} \Sigma_{g'} V_{gg'}
   \\
   L^\text{trunc}_{prg} &= U_{prg}\Sigma_{g}~.

Next, we commute the Fermionic operators

.. math::

   \hat a_p^\dagger \hat a_q^\dagger \hat a_s \hat a_r
   = \hat a_p^\dagger \hat a_r \hat a_q^\dagger \hat a_s - \hat a_p^\dagger \hat a_s \delta_{qr}

so that the Hamiltonian reads

.. math::

   \hat H = \sum_{pq} \bigl(h^{(1)}_{pg} + h_{pq}^\text{SI}\bigr) \hat a_{p}^\dagger \hat a_q
   + \frac12 \sum_{g} \hat L^\text{trunc}_{g} \hat L^\text{trunc}_{g}

where the self-interaction term is

.. math::

   h_{pq}^\text{SI} = -\frac12 \sum_{rg} L^\text{trunc}_{prg} L^\text{trunc}_{rqg}

and the part of the two-particle operator is

.. math::

   \hat L^\text{trunc}_{g} = \sum_{pq} L^\text{trunc}_{pqg} \hat a^\dagger_{p} \hat a_q~.

Finally, we shift the values by the mean-field expectation

.. math::

   \hat L_{g} = \hat L^\text{trunc}_{g} - \bar L_{g}

which leads to a new Hamiltonian of

.. math::

   \hat H = \sum_{pq} \bigl(h^{(1)}_{pg} + h_{pq}^\text{MF} + h_{pq}^\text{SI}\bigr) \hat a_{p}^\dagger \hat a_q
   + \frac12 \sum_{g} \hat L_{g} \hat L_{g}

with

.. math::

   h_{pq}^\text{MF} = \sum_g \bar L_g \bigl( L_{pqg} + \frac12 \bar L_{g} \delta_{pq} \bigr)~.


Energy evaluation
-----------------

In AFQMC, we often want to compute energy expectation values of the from

.. math::

   E = \frac{\sum_w W_w \braket{\Psi_\text{T}|\hat H|\Psi_w}}{\sum_w W_w \braket{\Psi_\text{T}|\Psi_w}}

where :math:`\ket{\Psi_\text{T}}` is a trial Slater determinant and the determinant
:math:`\ket{\Psi_w}` and weight :math:`W_w` describe a walker :math:`w`.