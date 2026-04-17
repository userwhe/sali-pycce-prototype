# reproduce_fig1_layden.py
#
# Reproduce the style of Fig. 1 from
# "Efficient Quantum Error Correction of Dephasing Induced by a Common Fluctuator"
#
# Curves:
#   - repetition code n=3,5
#   - hardware-efficient code n=2,3,5
#
# Model:
#   theta ~ N(0, sigma)
#   H_E = sum_j g_j Z_j
#   <U>(X) = exp[- sigma^2 * H_super^2 / 2](X)
#
# Numerics:
#   - Monte Carlo average over g_j ~ Uniform[0,1]
#   - Hardware-efficient recovery via transpose channel Eq. (A5)
#   - Repetition-code recovery via standard syndrome projectors for Z-errors
#
# Notes:
#   - This reproduces the paper's numerical method, not necessarily a pixel-perfect copy.
#   - Increase N_SAMPLES for smoother curves.
#   - n=5 hardware-efficient can be slow.
#
# Dependencies:
#   pip install numpy scipy matplotlib

import itertools
import math
from dataclasses import dataclass

import numpy as np
import matplotlib.pyplot as plt


# ----------------------------
# Basic helpers
# ----------------------------

def basis_bits(n: int) -> np.ndarray:
    """Return array of shape (2^n, n) with computational basis bits."""
    d = 2 ** n
    bits = np.zeros((d, n), dtype=int)
    for state in range(d):
        for q in range(n):
            bits[state, q] = (state >> (n - 1 - q)) & 1
    return bits


def z_signs(bits: np.ndarray) -> np.ndarray:
    """Map computational basis bits to Z eigenvalues: |0> -> +1, |1> -> -1."""
    return 1 - 2 * bits


def energies_from_g(g: np.ndarray) -> np.ndarray:
    """Diagonal energies of H_E = sum_j g_j Z_j in the computational basis."""
    bits = basis_bits(len(g))
    z = z_signs(bits)
    return z @ g


def avg_channel_apply(rho: np.ndarray, energies: np.ndarray, sigma: float) -> np.ndarray:
    """
    Apply <U> from Eq. (A4):
        <U>(rho)_{jk} = exp[-sigma^2 (E_j - E_k)^2 / 2] rho_{jk}
    """
    dE = energies[:, None] - energies[None, :]
    damp = np.exp(-0.5 * sigma**2 * dE**2)
    return damp * rho


def projector(psi: np.ndarray) -> np.ndarray:
    return np.outer(psi, np.conjugate(psi))


def normalize(vec: np.ndarray) -> np.ndarray:
    nrm = np.linalg.norm(vec)
    if nrm == 0:
        raise ValueError("Zero vector cannot be normalized.")
    return vec / nrm


def plus_minus_product_states(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Logical basis for the phase-flip repetition code."""
    plus = np.array([1.0, 1.0], dtype=complex) / np.sqrt(2)
    minus = np.array([1.0, -1.0], dtype=complex) / np.sqrt(2)

    psi0 = plus
    psi1 = minus
    for _ in range(n - 1):
        psi0 = np.kron(psi0, plus)
        psi1 = np.kron(psi1, minus)
    return psi0, psi1


# ----------------------------
# Hardware-efficient code
# ----------------------------

@dataclass
class EfficientCode:
    n: int
    q: int
    psi0: np.ndarray
    psi1: np.ndarray
    PL: np.ndarray
    energies: np.ndarray
    H_pows: list[np.ndarray]     # diagonal entries of H^m
    Mplus: np.ndarray            # pseudoinverse of KL matrix


def build_efficient_code(g: np.ndarray, rcond: float = 1e-12) -> EfficientCode:
    """
    Construct the hardware-efficient code from Eqs. (6)-(8), with q = 2^(n-1)-1.

    We pair basis states j and (2^n - 1 - j). The vector z lies in the orthogonal
    complement of span{v_m} for odd m = 1,3,...,2q-1, where (v_m)_i = <i|H_E^m|i>.
    """
    n = len(g)
    d = 2 ** n
    q = 2 ** (n - 1) - 1
    half = q + 1  # = 2^(n-1)

    E = energies_from_g(g)

    # Build V with odd-power columns v_m, m=1,3,...,2q-1
    odd_ms = list(range(1, 2 * q + 1, 2))
    V = np.column_stack([E[:half] ** m for m in odd_ms])  # shape (half, q)

    # Find z orthogonal to columns of V
    # Equivalent to nullspace of V^T. Since dim(nullspace) >= 1, take smallest singular vector.
    _, _, vh = np.linalg.svd(V.T, full_matrices=True)
    z = np.real_if_close(vh[-1]).astype(float)

    # Normalize z so that ||z||_1 = 1, as in the paper
    l1 = np.sum(np.abs(z))
    if l1 < 1e-15:
        raise RuntimeError("Failed to construct nonzero z vector.")
    z = z / l1

    # Build amplitudes r_j for |0_L>, then reverse them for |1_L>
    r = np.zeros(d, dtype=float)
    for j in range(half):
        partner = d - 1 - j
        if z[j] >= 0:
            r[partner] = np.sqrt(z[j])
        else:
            r[j] = np.sqrt(-z[j])

    psi0 = r.astype(complex)
    psi1 = r[::-1].astype(complex)

    psi0 = normalize(psi0)
    psi1 = normalize(psi1)

    PL = projector(psi0) + projector(psi1)

    # Precompute diagonal powers H^m for m up to 2q
    H_pows = [np.ones(d)]
    for m in range(1, 2 * q + 1):
        H_pows.append(E ** m)

    # KL matrix M_{jk} from P_L H^{j+k} P_L = m_{jk} P_L
    M = np.zeros((q + 1, q + 1), dtype=float)
    probs0 = np.abs(psi0) ** 2
    for j in range(q + 1):
        for k in range(q + 1):
            m = j + k
            M[j, k] = float(np.sum(probs0 * H_pows[m]))

    Mplus = np.linalg.pinv(M, rcond=rcond)

    return EfficientCode(
        n=n,
        q=q,
        psi0=psi0,
        psi1=psi1,
        PL=PL,
        energies=E,
        H_pows=H_pows[: q + 1],
        Mplus=Mplus,
    )


def efficient_recovery_apply(rho: np.ndarray, code: EfficientCode) -> np.ndarray:
    """
    Transpose recovery, Eq. (A5):
        R(rho) = sum_{j,k=0}^q (M^+)_{jk} P_L H^j rho H^k P_L
    Since H is diagonal, H^j rho H^k is easy to evaluate elementwise.
    """
    out = np.zeros_like(rho, dtype=complex)
    PL = code.PL
    for j in range(code.q + 1):
        dj = code.H_pows[j]
        left_scaled = dj[:, None] * rho
        for k in range(code.q + 1):
            coeff = code.Mplus[j, k]
            if abs(coeff) < 1e-15:
                continue
            dk = code.H_pows[k]
            term = left_scaled * dk[None, :]
            out += coeff * (PL @ term @ PL)
    return out


def efficient_p_for_sigma(code: EfficientCode, sigma: float) -> float:
    """
    For hardware-efficient codes, Appendix A shows the recovered logical channel has the form
        rho_L -> (1-p) rho_L + p Z_L rho_L Z_L.
    Therefore the logical off-diagonal |0><1| is scaled by (1 - 2p).
    """
    rho01 = np.outer(code.psi0, np.conjugate(code.psi1))
    noisy = avg_channel_apply(rho01, code.energies, sigma)
    recovered = efficient_recovery_apply(noisy, code)

    eta = np.vdot(code.psi0, recovered @ code.psi1)   # should be real in this model
    p = 0.5 * (1.0 - float(np.real_if_close(eta)))
    return float(np.clip(p, 0.0, 1.0))


# ----------------------------
# Repetition code
# ----------------------------

@dataclass
class RepetitionCode:
    n: int
    t: int
    psi0: np.ndarray
    psi1: np.ndarray
    PL: np.ndarray
    energies: np.ndarray
    signs_list: list[np.ndarray]   # diagonal entries of each correctable Z-error operator


def physical_p_for_sigma(g: np.ndarray, sigma: float) -> float:
    """
    For a single unencoded physical qubit under CFD, the effective channel is
        rho -> (1-p) rho + p Z rho Z,
    with p = [1 - exp(-2 g^2 sigma^2)] / 2 for coupling strength g.

    Since a sampled device has couplings g_j, we compare against the best physical
    qubit in that device, i.e. the one with the smallest error probability.
    """
    g_abs = np.abs(np.asarray(g, dtype=float))
    p_each = 0.5 * (1.0 - np.exp(-2.0 * (g_abs ** 2) * (sigma ** 2)))
    return float(np.min(p_each))


def monte_carlo_physical_curve(
    n: int,
    sigmas: np.ndarray,
    n_samples: int = 300,
    seed: int = 1234,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute the average physical-qubit error probability by Monte Carlo over
    g_j ~ Uniform[0,1]. For each sampled register, use the physical qubit with
    the smallest error probability, matching the paper's pseudothreshold comparison.
    """
    rng = np.random.default_rng(seed)
    values = np.zeros((n_samples, len(sigmas)), dtype=float)

    for s in range(n_samples):
        g = rng.uniform(0.0, 1.0, size=n)
        for i, sigma in enumerate(sigmas):
            values[s, i] = physical_p_for_sigma(g, float(sigma))

        if show_progress and ((s + 1) % max(1, n_samples // 10) == 0):
            print(f"{'physical':10s} n={n}: sample {s+1}/{n_samples}")

    return values.mean(axis=0)


def error_signs_for_mask(n: int, mask: tuple[int, ...], bits: np.ndarray) -> np.ndarray:
    """
    Diagonal entries (+/-1) of product_{j in mask} Z_j in computational basis.
    """
    signs = np.ones(bits.shape[0], dtype=float)
    for q in mask:
        signs *= (1 - 2 * bits[:, q])
    return signs


def build_repetition_code(g: np.ndarray) -> RepetitionCode:
    """
    Build phase-flip repetition code:
        |0_L> = |+>^⊗n, |1_L> = |->^⊗n
    Recovery corrects all Z-errors of weight <= t=(n-1)//2.
    """
    n = len(g)
    if n % 2 == 0:
        raise ValueError("Repetition code here assumes odd n.")
    t = (n - 1) // 2
    psi0, psi1 = plus_minus_product_states(n)
    PL = projector(psi0) + projector(psi1)
    E = energies_from_g(g)
    bits = basis_bits(n)

    signs_list = []
    for w in range(t + 1):
        for mask in itertools.combinations(range(n), w):
            signs_list.append(error_signs_for_mask(n, mask, bits))

    return RepetitionCode(
        n=n,
        t=t,
        psi0=psi0,
        psi1=psi1,
        PL=PL,
        energies=E,
        signs_list=signs_list,
    )


def repetition_recovery_apply(rho: np.ndarray, code: RepetitionCode) -> np.ndarray:
    """
    Standard exact recovery for correctable Z-errors E_a:
        R(rho) = sum_a E_a P_a rho P_a E_a
    where P_a = E_a P_L E_a.
    Since E_a are diagonal Hermitian involutions, E_a^\dagger = E_a = E_a^{-1}.
    """
    out = np.zeros_like(rho, dtype=complex)
    for signs in code.signs_list:
        Epsi0 = signs * code.psi0
        Epsi1 = signs * code.psi1
        Pa = projector(Epsi0) + projector(Epsi1)
        term = Pa @ rho @ Pa
        term = signs[:, None] * term * signs[None, :]
        out += term
    return out


def repetition_p_for_sigma(code: RepetitionCode, sigma: float) -> float:
    """
    For repetition code under CFD, Appendix A states the logical channel has bit-flip form:
        rho_L -> (1-p) rho_L + p X_L rho_L X_L.
    So apply the channel to |0_L><0_L| and read out population in |1_L>.
    """
    rho0 = projector(code.psi0)
    noisy = avg_channel_apply(rho0, code.energies, sigma)
    recovered = repetition_recovery_apply(noisy, code)
    p = np.vdot(code.psi1, recovered @ code.psi1)
    return float(np.clip(np.real_if_close(p), 0.0, 1.0))


# ----------------------------
# Monte Carlo averaging
# ----------------------------

def monte_carlo_curve(
    curve_type: str,
    n: int,
    sigmas: np.ndarray,
    n_samples: int = 300,
    seed: int = 1234,
    show_progress: bool = True,
) -> np.ndarray:
    """
    Compute <p>(sigma) by Monte Carlo over g_j ~ Uniform[0,1].
    curve_type: "efficient" or "repetition"
    """
    rng = np.random.default_rng(seed)
    values = np.zeros((n_samples, len(sigmas)), dtype=float)

    for s in range(n_samples):
        g = rng.uniform(0.0, 1.0, size=n)

        if curve_type == "efficient":
            code = build_efficient_code(g)
            for i, sigma in enumerate(sigmas):
                values[s, i] = efficient_p_for_sigma(code, float(sigma))

        elif curve_type == "repetition":
            code = build_repetition_code(g)
            for i, sigma in enumerate(sigmas):
                values[s, i] = repetition_p_for_sigma(code, float(sigma))

        else:
            raise ValueError("curve_type must be 'efficient' or 'repetition'.")

        if show_progress and ((s + 1) % max(1, n_samples // 10) == 0):
            print(f"{curve_type:10s} n={n}: sample {s+1}/{n_samples}")

    return values.mean(axis=0)


# ----------------------------
# Main plotting routine
# ----------------------------

def main():
    # Increase these for smoother curves.
    N_SAMPLES = 900
    SIGMAS = np.linspace(0.0, 10.0, 256)
    SEED = 2026

    curves = {}

    # Physical qubits: use the same sampled register sizes for comparison
    for n in [1]:
        curves[f"physical n={n}"] = monte_carlo_physical_curve(
            n=n,
            sigmas=SIGMAS,
            n_samples=N_SAMPLES,
            seed=SEED + 5 * n,
        )

    # Repetition code: n=3,5
    for n in [3, 5]:
        curves[f"repetition n={n}"] = monte_carlo_curve(
            curve_type="repetition",
            n=n,
            sigmas=SIGMAS,
            n_samples=N_SAMPLES,
            seed=SEED + 10 * n,
        )

    # Hardware-efficient code: n=2,3,4,5
    for n in [2, 3, 4, 5]:
        curves[f"efficient n={n}"] = monte_carlo_curve(
            curve_type="efficient",
            n=n,
            sigmas=SIGMAS,
            n_samples=N_SAMPLES,
            seed=SEED + 100 * n,
        )

    # Plot
    plt.figure(figsize=(8.0, 5.2))

    # No explicit colors set, per plotting guideline.
    for label, ys in curves.items():
        plt.plot(SIGMAS, ys, linewidth=2, label=label)

    plt.xlabel(r"Noise strength $\sigma$")
    plt.ylabel(r"Average error probability $\langle p \rangle$")
    plt.title("average error probability vs noise strength")
    plt.ylim(0, 0.5)   
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()