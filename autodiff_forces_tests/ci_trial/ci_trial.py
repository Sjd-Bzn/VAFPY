"""AFQMC with a CI trial (here the FCI ground state) for the Gamma-point 16-orbital Hamiltonian: the propagation of
autodiff_forces.Model with the overlap, the force bias and the local energy taken from the trial
    |Psi_T> = sum_{I,J} C[I,J] |I_alpha J_beta>,
where I, J are strings of 4 occupied orbitals. A walker is one determinant Phi (nb x ne, the same for both spins):
    <Psi_T|Phi>  = sum_{I,J} C[I,J] det(Phi[occ_I]) det(Phi[occ_J])
    <Psi_T|E_pq|Phi> = sum_i dO/dPhi[p,i] Phi[q,i]          (E_pq = sum over spin of a+_p a_q)
    G[p,q] = <Psi_T|E_pq|Phi> / (2 <Psi_T|Phi>)            (one spin; for an HF trial G[p,q] = theta[q,p], p occupied)
Force bias as the compact kernel of functions.py with P_g = sum L[p,q,g] G[p,q], Q_g = sum conj(L[q,p,g]) G[p,q].
Local energy: Psi_T is an eigenstate of H with energy e0_trial, so <Psi_T|H + lam V|Phi>/<Psi_T|Phi>
= e0_trial + lam * 2 sum V[p,q] G[p,q] (exact only for an exact eigenstate; for other trials use only the V part).
Only V perturbations (no dL), Gamma point, compact layout."""
import sys
import numpy as np
sys.path.insert(0, "/fs/home/sajad/projects/git/AFQMC-autodiff-forces/source/vafpy_v3")
import functions as new
import autodiff_forces as af

jax, jnp = new._try_import_jax()


@jax.tree_util.register_pytree_node_class
class ModelCI(af.Model):
    ARRAYS = af.Model.ARRAYS + ("C", "occ_idx", "Lpq")
    STATIC = af.Model.STATIC + ("e0_trial",)

    def __init__(self, config, hamiltonian, trial, V, C, occ, e0_trial, Lc):
        super().__init__(config, hamiltonian, trial, V)
        if self.nk != 1:
            raise NotImplementedError("CI trial: Gamma point only")
        self.C = jnp.asarray(C, dtype=self.ctype)
        self.occ_idx = jnp.asarray(occ, dtype=jnp.int32)
        self.Lpq = jnp.asarray(Lc, dtype=self.ctype)                 # L[p,q,g] = Lc[b2=p, b1=q, g]
        self.e0_trial = float(e0_trial)

    def overlap1(self, d):
        a = jnp.linalg.det(d[self.occ_idx])                          # (n_strings,)
        return a @ self.C @ a

    def overlap(self, dets):
        return jax.vmap(self.overlap1)(dets) if dets.ndim == 3 else self.overlap1(dets)

    def green(self, dets):
        def one(d):
            g = jax.grad(self.overlap1, holomorphic=True)(d)
            return (g @ d.T) / (2 * self.overlap1(d))                # G[p,q]
        return jax.vmap(one)(dets)

    def force_bias(self, dets, P):
        G = self.green(dets)                                         # (w, p, q)
        Pg = jnp.einsum("pqg,wpq->gw", self.Lpq, G)
        Qg = jnp.einsum("qpg,wpq->gw", self.Lpq.conj(), G)
        S = jnp.einsum("wpp->w", G)
        Pg = Pg - P["c"] * S[None, :]
        Qg = Qg - P["cc"] * S[None, :]
        return jnp.concatenate([(Pg + Qg) * 0.5, (Pg - Qg) * 0.5j], axis=0)

    def e1_mixed(self, dets, V=None):
        V = self.V if V is None else V
        return 2 * jnp.einsum("pq,wpq->w", V, self.green(dets))

    def local_energy(self, lam, dets, P=None):
        return self.e0_trial + lam * self.e1_mixed(dets)

    def step(self, exp_half, e0, dets, weights, x, P=None):
        """Model.step with the force bias and the overlap of the CI trial."""
        P = self.pieces(0.0) if P is None else P
        fb = -2j * self.sqrt_tau * self.force_bias(dets, P)
        fb = jnp.where(jnp.abs(fb) > 1, 0.0, fb)
        arg = jnp.sum((x - 0.5 * fb) * fb, axis=0)
        field = 1j * self.sqrt_tau * self.k["aux"](P["M"], P["c"], P["cc"], self.eye, x - fb)
        half = exp_half @ dets
        result, addend = half, half
        for i in range(self.order):
            addend = (field @ addend) / (i + 1)
            result = result + addend
        new_dets = P["h0"] * (exp_half @ result)
        ratio = self.overlap(new_dets) / self.overlap(dets)
        factor = jnp.abs(ratio * jnp.exp(arg) * jnp.exp(self.dtau * e0))
        factor = jnp.where(factor < 10, factor, 0.0)
        new_w = factor * jnp.maximum(0.0, jnp.cos(jnp.angle(ratio))) * weights
        new_dets = new.cholesky_orthonormalize_complex_jax(new_dets)
        return new_dets, new_w.astype(self.ctype)
