from dataclasses import dataclass
from opt_einsum import contract_expression
import numpy.typing as npt


@dataclass
class Constants:
    L: npt.ArrayLike
    number_walker: int


class State:
    def __init__(self, constants):
        num_g = constants.L.shape[-1]
        self._make_auxiliary_field = contract_expression(
            "ijG,GN->ijN", constants.L, (num_g, constants.number_walker), constants=[0]
        )

    def new_auxiliary_field(self, force_bias):
        if force_bias is None:
            x = self.random_normal()
        else:
            x = self.random_normal() - force_bias
        return self._make_auxiliary_field(x)

    def random_normal(self):
        pass


def update_hyb(trial_0, trial, walker_mat, walker_weight, q_list, h_0, h_1, d_tau, e_0):
    NG = num_g
    new_walker_mat = np.zeros_like(walker_mat)
    new_walker_weight = np.zeros_like(walker_weight)
    theta_mat = []
    for i in range(NUM_WALKERS):
        theta_mat.append(theta(trial, walker_mat[i]))
    theta_mat = np.array(theta_mat)
    # fb_e_Q = -2j*SQRT_DTAU*contract('Nri,irG->NG',theta_mat, alpha_full_e,optimize='greedy')
    fb_e_Q = -2j * SQRT_DTAU * expr_fb_e(theta_mat)
    # fb_o_Q = -2j*SQRT_DTAU*contract('Nri,irG->NG',theta_mat, alpha_full_o,optimize='greedy')
    fb_o_Q = -2j * SQRT_DTAU * expr_fb_o(theta_mat)
    x_e_Q = np.random.randn(NG * NUM_WALKERS).reshape(NG, NUM_WALKERS)
    x_o_Q = np.random.randn(NG * NUM_WALKERS).reshape(NG, NUM_WALKERS)
    h_2 = expr_h2_e(x_e_Q - fb_e_Q.T) + expr_h2_o(x_o_Q - fb_o_Q.T)
    # h_2 = h_mf.two_body_e@(x_e_Q-fb_e_Q.T) + h_mf.two_body_o@(x_o_Q-fb_o_Q.T)
    for i in range(0, NUM_WALKERS):
        h = h_1 + SQRT_DTAU * 1j * h_2[:, :, i]
        addend = walker_mat[i]
        for j in range(order_trunc + 1):
            new_walker_mat[i] += addend
            addend = h @ addend / (j + 1)
        ovrlap_ratio = (
            np.linalg.det(overlap(trial, new_walker_mat[i])) ** 2
            / np.linalg.det(overlap(trial, walker_mat[i])) ** 2
        )
        alpha = phase(ovrlap_ratio)
        new_walker_weight[i] = (
            abs(
                ovrlap_ratio
                * (
                    np.exp(
                        np.dot(x_e_Q[:, i], fb_e_Q[i])
                        - np.dot(fb_e_Q[i], fb_e_Q[i] / 2)
                    )
                    * np.exp(
                        np.dot(x_o_Q[:, i], fb_o_Q[i])
                        - np.dot(fb_o_Q[i], fb_o_Q[i] / 2)
                    )
                )
            )
            * max(0, np.cos(alpha))
            * walker_weight[i]
        )
    return new_walker_mat, new_walker_weight
