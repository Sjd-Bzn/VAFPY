from dataclasses import dataclass
import numpy as np
import yaml


@dataclass
class Constants:
    number_electron: np.array
    H1: np.array
    L: np.array


@dataclass
class Energy:
    singularity: float
    ion_electron: float
    ewald: float
    one_electron: float
    hartree: float
    exchange: float
    paw_all_electron: float
    paw_pseudo: float
    atomic: float


def setup():
    with open("afqmc.yaml", "r") as yaml_file:
        content = yaml.safe_load(yaml_file)
    vasp = content["vasp"]
    constants = Constants(
        number_electron=np.array(vasp["number_electron"]),
        H1=np.load("H1.npy"),
        L=np.load("H2.npy"),
    )
    energy = Energy(
        singularity=vasp["singularity"],
        ion_electron=vasp["alpha Z        PSCENC"],
        ewald=vasp["Ewald energy   TEWEN"],
        one_electron=vasp["One el. energ  E1"],
        hartree=vasp["Hartree energ -DENC"],
        exchange=vasp["exchange       EXHF"],
        paw_all_electron=vasp["PAW double counting"][0],
        paw_pseudo=vasp["PAW double counting"][1],
        atomic=vasp["atomic energy  EATOM"],
    )
    return constants, energy


def vasp_consistency_check(constants, energy):
    assert len(constants.number_electron) == 1, "collinear systems not implemented"
    assert constants.H1.shape[2] == 1, "k-points not implemented"
    num_elec = constants.number_electron[0]
    L_occupied = constants.L[:, :num_elec, :num_elec]
    H2_occupied = np.einsum("gij,glk -> ijkl", L_occupied, L_occupied.conj())
    one_body = one_body_energy(num_elec, constants.H1)
    hartree = hartree_energy(num_elec, energy.singularity, H2_occupied)
    exchange = exchange_energy(H2_occupied)
    print("one body", one_body.real, energy.one_electron)
    print("hartree", hartree.real, energy.hartree)
    print("exchange", exchange.real, energy.exchange)
    print(
        (one_body.real + hartree.real + exchange.real),
        energy.one_electron + energy.hartree + energy.exchange,
    )


def one_body_energy(number_electron, H1):
    return np.trace(H1[:number_electron, :number_electron, 0])


def hartree_energy(number_electron, singularity, H2_occupied):
    return 2 * (np.einsum("iijj", H2_occupied) - number_electron**2 * singularity)


def exchange_energy(H2_occupied):
    return -np.einsum("ijij", H2_occupied)


def main():
    constants, energy = setup()
    vasp_consistency_check(constants, energy)


if __name__ == "__main__":
    main()
