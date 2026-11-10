"""
Diagnostic checks for MaternCovariance.

Run this file from the directory that contains the `deformation_gp` package:

    python check_matern_covariance.py

Nothing is saved to disk. Numerical diagnostics are printed to the console,
and figures are displayed with plt.show().
"""

import math

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.special import gamma, kv

from deformation_gp import MaternCovariance, set_bessel_scipy_parallel


# Use float64 so that the numerical comparisons are meaningful.
torch.set_default_dtype(torch.float64)

# Parallelism is unnecessary for these small diagnostic calculations.
set_bessel_scipy_parallel(parallel=False)


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------
def set_covariance_parameters(model, a=1.0, sigma2=1.0):
    """Set a and sigma^2 directly through the model's log-parameters."""
    with torch.no_grad():
        model.log_a.copy_(
            torch.tensor(math.log(float(a)), dtype=model.log_a.dtype)
        )
        model.log_sigma2.copy_(
            torch.tensor(math.log(float(sigma2)), dtype=model.log_sigma2.dtype)
        )
    return model


def make_general_matern(nu, a=1.0, sigma2=1.0, estimate_nu=False):
    """Create the general Matérn branch at a specified nu."""
    model = MaternCovariance(
        covariance_model="matern",
        estimate_nu=estimate_nu,
        nu_initial=float(nu),
        nu_lower=0.01,
        nu_upper=6.0,
        use_spatial_variance=False,
    )
    return set_covariance_parameters(model, a=a, sigma2=sigma2)


def make_closed_form_matern(name, a=1.0, sigma2=1.0):
    """Create one of the closed-form branches used in the implementation."""
    model = MaternCovariance(
        covariance_model=name,
        estimate_nu=False,
        use_spatial_variance=False,
    )
    return set_covariance_parameters(model, a=a, sigma2=sigma2)


def scipy_matern_correlation(distance, nu, a):
    """
    Reference Matérn correlation:

        rho(h) =
        2^(1-nu) / Gamma(nu)
        * z^nu K_nu(z),

        z = sqrt(2 nu) * a * h.
    """
    h = np.asarray(distance, dtype=np.float64)
    z = np.sqrt(2.0 * float(nu)) * float(a) * h

    result = np.ones_like(z)
    positive = z > 0.0

    result[positive] = (
        2.0 ** (1.0 - float(nu))
        / gamma(float(nu))
        * z[positive] ** float(nu)
        * kv(float(nu), z[positive])
    )
    return result


def status(value, tolerance, lower_is_better=True):
    if lower_is_better:
        return "PASS" if value <= tolerance else "CHECK"
    return "PASS" if value >= tolerance else "CHECK"


# ---------------------------------------------------------------------
# 1. Closed-form branches versus the general Matérn implementation
# ---------------------------------------------------------------------
def check_closed_forms():
    print("\n" + "=" * 72)
    print("1. Closed-form Matérn vs general Matérn")
    print("=" * 72)

    a = 1.3
    distances = torch.linspace(0.0, 5.0, 1001)

    cases = [
        (0.5, "exponential"),
        (1.5, "matern15"),
        (2.5, "matern25"),
    ]

    tolerance = 1e-10

    for nu, name in cases:
        closed_model = make_closed_form_matern(name, a=a)
        general_model = make_general_matern(nu, a=a)

        closed = closed_model.correlation_from_distance(distances)
        general = general_model.correlation_from_distance(distances)

        max_abs_error = torch.max(torch.abs(closed - general)).item()

        print(
            f"nu={nu:3.1f} | {name:11s} vs general | "
            f"max abs error = {max_abs_error:.3e} | "
            f"{status(max_abs_error, tolerance)}"
        )


# ---------------------------------------------------------------------
# 2. General Matérn versus the mathematical reference formula
# ---------------------------------------------------------------------
def check_against_scipy_reference():
    print("\n" + "=" * 72)
    print("2. General Matérn vs direct SciPy reference formula")
    print("=" * 72)

    a = 1.7

    # Start away from zero here so this test checks the Matérn formula itself,
    # separately from the implementation's special treatment near zero.
    distances_np = np.linspace(1e-3, 5.0, 1200)
    distances = torch.tensor(distances_np)

    nu_values = [0.1, 0.2, 0.5, 0.9, 1.5, 2.5, 4.5]
    tolerance = 1e-10

    for nu in nu_values:
        model = make_general_matern(nu, a=a)
        implemented = (
            model.correlation_from_distance(distances)
            .detach()
            .cpu()
            .numpy()
        )
        reference = scipy_matern_correlation(distances_np, nu=nu, a=a)

        max_abs_error = np.max(np.abs(implemented - reference))

        print(
            f"nu={nu:3.1f} | max abs error = {max_abs_error:.3e} | "
            f"{status(max_abs_error, tolerance)}"
        )


# ---------------------------------------------------------------------
# 3. Basic covariance-matrix properties
# ---------------------------------------------------------------------
def check_covariance_matrix_properties():
    print("\n" + "=" * 72)
    print("3. Covariance-matrix checks")
    print("=" * 72)

    torch.manual_seed(123)
    coordinates = torch.rand(60, 2)

    a = 2.0
    sigma2 = 1.7
    nu_values = [0.1, 0.5, 1.5, 2.5, 4.5]

    for nu in nu_values:
        model = make_general_matern(nu, a=a, sigma2=sigma2)

        covariance = model.covariance_matrix(
            coordinates,
            coordinates,
            coordinates,
            coordinates,
        )

        symmetry_error = torch.max(
            torch.abs(covariance - covariance.T)
        ).item()

        diagonal_error = torch.max(
            torch.abs(
                torch.diagonal(covariance)
                - torch.full((coordinates.shape[0],), sigma2)
            )
        ).item()

        eigenvalues = torch.linalg.eigvalsh(
            0.5 * (covariance + covariance.T)
        )
        minimum_eigenvalue = torch.min(eigenvalues).item()

        distance = torch.linspace(0.0, 5.0, 1000)
        correlation = model.correlation_from_distance(distance)
        largest_increase = torch.max(
            correlation[1:] - correlation[:-1]
        ).item()

        valid_range = bool(
            torch.all((correlation >= 0.0) & (correlation <= 1.0))
        )

        print(
            f"nu={nu:3.1f} | "
            f"sym.err={symmetry_error:.2e} | "
            f"diag.err={diagonal_error:.2e} | "
            f"min eig={minimum_eigenvalue:.3e} | "
            f"largest rho increase={largest_increase:.3e} | "
            f"rho in [0,1]={valid_range}"
        )

    print("\nInterpretation:")
    print("  - symmetry error should be essentially 0.")
    print("  - diagonal error should be essentially 0.")
    print("  - minimum eigenvalue should be >= 0, allowing tiny round-off error.")
    print("  - largest rho increase should be <= 0, allowing tiny round-off error.")


# ---------------------------------------------------------------------
# 4. Gradient checks for distance and nu
# ---------------------------------------------------------------------
def check_gradients():
    print("\n" + "=" * 72)
    print("4. Autograd gradient checks")
    print("=" * 72)

    a = 1.4
    nu0 = 1.2
    h0 = 0.7

    model = make_general_matern(
        nu0,
        a=a,
        sigma2=1.0,
        estimate_nu=True,
    )

    # Autograd derivative with respect to distance h.
    distance = torch.tensor([h0], requires_grad=True)
    correlation = model.correlation_from_distance(distance)[0]
    correlation.backward()

    autograd_dh = distance.grad.item()

    # Convert the raw_nu gradient to d rho / d nu.
    raw_nu = model.raw_nu.detach()
    sigmoid_raw = torch.sigmoid(raw_nu)
    dnu_draw = (
        (model.nu_upper - model.nu_lower)
        * sigmoid_raw
        * (1.0 - sigmoid_raw)
    ).item()

    autograd_dnu = model.raw_nu.grad.item() / dnu_draw

    # Numerical derivatives of the independent reference formula.
    eps_h = 1e-6
    finite_dh = (
        scipy_matern_correlation(h0 + eps_h, nu0, a)
        - scipy_matern_correlation(h0 - eps_h, nu0, a)
    ) / (2.0 * eps_h)

    eps_nu = 1e-5
    finite_dnu = (
        scipy_matern_correlation(h0, nu0 + eps_nu, a)
        - scipy_matern_correlation(h0, nu0 - eps_nu, a)
    ) / (2.0 * eps_nu)

    rel_error_h = abs(autograd_dh - finite_dh) / max(
        1.0, abs(finite_dh)
    )
    rel_error_nu = abs(autograd_dnu - finite_dnu) / max(
        1.0, abs(finite_dnu)
    )

    print(
        f"d rho / d h : autograd={autograd_dh:.10e}, "
        f"finite diff={float(finite_dh):.10e}, "
        f"error={rel_error_h:.3e}"
    )
    print(
        f"d rho / d nu: autograd={autograd_dnu:.10e}, "
        f"finite diff={float(finite_dnu):.10e}, "
        f"error={rel_error_nu:.3e}"
    )

    print("\nThe nu gradient is especially important when estimate_nu=True.")


# ---------------------------------------------------------------------
# 5. Near-zero stress test
# ---------------------------------------------------------------------
def check_near_zero_behavior():
    print("\n" + "=" * 72)
    print("5. Near-zero behavior of the general Matérn branch")
    print("=" * 72)

    a = 1.0
    threshold = 1e-5

    # The current implementation sets rho=1 whenever
    # z = sqrt(2 nu) * a * h <= 1e-5.
    # This test compares that approximation with the exact reference formula.
    nu_values = [0.05, 0.1, 0.2, 0.5, 1.5, 2.5]

    print(
        "Testing just below the implementation threshold "
        "z = sqrt(2*nu)*a*h = 0.99e-5"
    )

    for nu in nu_values:
        z = 0.99 * threshold
        h = z / (math.sqrt(2.0 * nu) * a)

        model = make_general_matern(nu, a=a)
        implemented = model.correlation_from_distance(
            torch.tensor([h])
        ).item()
        reference = float(
            scipy_matern_correlation(np.array([h]), nu, a)[0]
        )

        abs_error = abs(implemented - reference)
        flag = "OK" if abs_error < 1e-6 else "WARNING"

        print(
            f"nu={nu:4.2f} | h={h:.3e} | "
            f"implemented={implemented:.10f} | "
            f"reference={reference:.10f} | "
            f"abs error={abs_error:.3e} | {flag}"
        )

    print(
        "\nThis section is a stress test, not a general failure test. "
        "It specifically examines the hard near-zero cutoff used in "
        "_general_matern_correlation()."
    )


# ---------------------------------------------------------------------
# 6. Visual diagnostics
# ---------------------------------------------------------------------
def plot_nu_effect():
    a = 1.0
    distance = torch.linspace(0.0, 5.0, 800)
    nu_values = [0.5, 1.0, 1.5, 2.5, 4.0]

    plt.figure(figsize=(7, 5))

    for nu in nu_values:
        model = make_general_matern(nu, a=a)
        correlation = (
            model.correlation_from_distance(distance)
            .detach()
            .cpu()
            .numpy()
        )
        plt.plot(
            distance.numpy(),
            correlation,
            label=fr"$\nu={nu}$",
        )

    plt.xlabel("Distance h")
    plt.ylabel(r"Correlation $\rho(h)$")
    plt.title(r"Matérn correlation for different $\nu$ ($a=1$)")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()


def plot_nu_effect_near_origin():
    a = 1.0
    distance = torch.linspace(0.0, 0.5, 800)
    nu_values = [0.5, 1.0, 1.5, 2.5, 4.0]

    plt.figure(figsize=(7, 5))

    for nu in nu_values:
        model = make_general_matern(nu, a=a)
        correlation = (
            model.correlation_from_distance(distance)
            .detach()
            .cpu()
            .numpy()
        )
        plt.plot(
            distance.numpy(),
            correlation,
            label=fr"$\nu={nu}$",
        )

    plt.xlabel("Distance h")
    plt.ylabel(r"Correlation $\rho(h)$")
    plt.title(r"Near-origin behavior for different $\nu$")
    plt.ylim(0.75, 1.005)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()


def plot_a_effect():
    nu = 1.5
    distance = torch.linspace(0.0, 5.0, 800)
    a_values = [0.5, 1.0, 2.0, 4.0]

    plt.figure(figsize=(7, 5))

    for a in a_values:
        model = make_general_matern(nu, a=a)
        correlation = (
            model.correlation_from_distance(distance)
            .detach()
            .cpu()
            .numpy()
        )
        plt.plot(
            distance.numpy(),
            correlation,
            label=fr"$a={a}$",
        )

    plt.xlabel("Distance h")
    plt.ylabel(r"Correlation $\rho(h)$")
    plt.title(r"Matérn correlation for different $a$ ($\nu=1.5$)")
    plt.ylim(-0.02, 1.02)
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()


def plot_closed_form_error():
    a = 1.3
    distance_np = np.linspace(1e-4, 5.0, 1200)
    distance = torch.tensor(distance_np)

    cases = [
        (0.5, "exponential"),
        (1.5, "matern15"),
        (2.5, "matern25"),
    ]

    plt.figure(figsize=(7, 5))

    for nu, name in cases:
        closed_model = make_closed_form_matern(name, a=a)
        general_model = make_general_matern(nu, a=a)

        closed = (
            closed_model.correlation_from_distance(distance)
            .detach()
            .cpu()
            .numpy()
        )
        general = (
            general_model.correlation_from_distance(distance)
            .detach()
            .cpu()
            .numpy()
        )

        error = np.maximum(
            np.abs(closed - general),
            np.finfo(np.float64).tiny,
        )

        plt.semilogy(
            distance_np,
            error,
            label=fr"$\nu={nu}$",
        )

    plt.xlabel("Distance h")
    plt.ylabel("Absolute difference")
    plt.title("Closed-form vs general Matérn")
    plt.legend()
    plt.grid(alpha=0.25)
    plt.tight_layout()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    check_closed_forms()
    check_against_scipy_reference()
    check_covariance_matrix_properties()
    check_gradients()
    check_near_zero_behavior()

    plot_nu_effect()
    plot_nu_effect_near_origin()
    plot_a_effect()
    plot_closed_form_error()

    print("\n" + "=" * 72)
    print("Figures are displayed only; no result files are saved.")
    print("=" * 72)

    plt.show()


if __name__ == "__main__":
    main()
