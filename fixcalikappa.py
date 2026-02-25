import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import brentq

# =====================================================
# PARAMETERS
# =====================================================
params = {
    "alpha": 1.0,
    "beta": 1.0,
    "gamma_bar": 2.0,
    "kappa": 2.0,
    "eta": 0.08,
    "xi": 0.1,
    "delta": 0.02,
    "theta": 1.0,
    "rho": 0.05,
    "sigma": 0.3,
    "sigma_a": 0.4,
    "lambda": 0.05,
    "alpha_F": 1.5,
    "eta_F": 0.06,
    "delta_F": 0.02,
    "B": 2.0,
}

beta = params["beta"]
xi = params["xi"]
theta = params["theta"]
rho = params["rho"]
sigma_a = params["sigma_a"]

# =====================================================
# STEP 1: FDI
# =====================================================


def step1_fdi_equilibrium():
    pF = 1 / (rho + params["eta_F"])
    rF = params["alpha_F"] * pF
    xF_star = params["alpha_F"]**2 / \
        ((rho + params["eta_F"]) * (params["eta_F"] - params["delta_F"]))
    return pF, rF, xF_star


pf_val, rf_val, xf_star_val = step1_fdi_equilibrium()
print(f"pF = {pf_val:.4f}, rF = {rf_val:.4f}, xF* = {xf_star_val:.4f}")

# =====================================================
# STEP 3a: HJB (STABLE NUMERICS, KHÔNG ĐỔI PARAMS)
# =====================================================


def solve_h(a_grid, sR, sa, params, max_iter=2000, tol=1e-6):
    gamma_bar = params["gamma_bar"]
    kappa = params["kappa"]
    beta = params["beta"]
    rho = params["rho"]
    sigma_a = params["sigma_a"]
    alpha = params["alpha"]
    theta = params["theta"]
    xi = params["xi"]
    lamb = params["lambda"]

    p = 1 / (rho + params["eta"])
    Na = len(a_grid)
    da = a_grid[1] - a_grid[0]

    # Initial guess (smooth, decaying)
    h = 0.05 * np.exp(-a_grid / (kappa + 1e-8)) + 1e-3

    gamma_a = gamma_bar * a_grid**2 / (kappa**2 + a_grid**2 + 1e-12)
    gamma_prime = gamma_bar * (2 * a_grid * kappa**2) / \
        ((kappa**2 + a_grid**2)**2 + 1e-12)

    term1 = p * lamb * xf_star_val
    term2 = alpha * p * (sR + alpha * p * gamma_a)
    F_prime = gamma_prime * (term1 + term2)

    drift_coef = theta * sa - xi * a_grid

    damping = 0.2
    best_residual = np.inf
    best_h = h.copy()

    for it in range(max_iter):
        h_old = h.copy()
        h_new = np.zeros_like(h)

        h_x = np.zeros(Na)
        h_xx = np.zeros(Na)

        for i in range(1, Na-1):
            # upwind
            if drift_coef[i] >= 0:
                h_x[i] = (h[i] - h[i-1]) / da
            else:
                h_x[i] = (h[i+1] - h[i]) / da
            h_xx[i] = (h[i+1] - 2*h[i] + h[i-1]) / da**2

        for i in range(1, Na-1):
            num = (F_prime[i] +
                   beta**2 * h[i] * h_x[i] +
                   h_x[i] * drift_coef[i] +
                   (sigma_a**2 / 2) * h_xx[i] -
                   xi * h[i])

            if np.isfinite(num):
                h_new[i] = num / rho
            else:
                h_new[i] = h[i]

        h_new[0] = 0
        h_new[-1] = 0

        # chỉ chặn số học, KHÔNG đổi mô hình
        h_new = np.clip(h_new, -50, 50)

        res = np.max(np.abs(h_new - h_old))
        if res < best_residual:
            best_residual = res
            best_h = h_new.copy()

        h = damping * h_new + (1 - damping) * h_old

        if res < tol:
            break

    return best_h

# =====================================================
# STEP 3c: FP distribution (STABLE)
# =====================================================


def stationary_density(h, a_grid, sa):
    mu = beta**2 * h + theta * sa - xi * a_grid

    Phi = np.zeros_like(a_grid)
    for i in range(1, len(a_grid)):
        integrand = 2 * mu[:i+1] / sigma_a**2
        Phi[i] = np.trapezoid(integrand, a_grid[:i+1])

    Phi -= np.max(Phi)      # chống overflow
    Phi = np.clip(Phi, -700, 50)

    m = np.exp(Phi)
    Z = np.trapezoid(m, a_grid)
    m = m / Z if Z > 0 else np.ones_like(a_grid) / a_grid[-1]

    return m


# =====================================================
# STEP 4: CALIBRATION κ (A = 2κ, KHÔNG ĐỔI PARAMS)
# =====================================================
TARGET_TRAINING = 0.087
sR_baseline = 0.1
sa_baseline = 0.3   # giữ nguyên baseline của bạn


def training_share(kappa_val):
    params_tmp = params.copy()
    params_tmp["kappa"] = kappa_val

    A = 2 * kappa_val
    Na = 250
    a_grid = np.linspace(0, A, Na)

    h = solve_h(a_grid, sR_baseline, sa_baseline, params_tmp)
    m = stationary_density(h, a_grid, sa_baseline)

    idx = a_grid >= kappa_val
    share = np.trapezoid(m[idx], a_grid[idx])
    return share


def calibrate_kappa():
    print("\n=== Calibrating kappa ===")

    grid = np.linspace(0.5, 8.0, 10)
    vals = []

    for k in grid:
        s = training_share(k)
        vals.append(s)
        print(f"  kappa={k:.2f} → training={s:.4f}")

    for i in range(len(grid)-1):
        if (vals[i] - TARGET_TRAINING) * (vals[i+1] - TARGET_TRAINING) < 0:
            kL, kU = grid[i], grid[i+1]
            print(f"Bracket: [{kL:.2f}, {kU:.2f}]")

            def obj(k):
                return training_share(k) - TARGET_TRAINING

            k_star = brentq(obj, kL, kU, xtol=1e-4)
            return k_star

    print("No bracket found")
    return params["kappa"]


# =====================================================
# RUN
# =====================================================
kappa_star = calibrate_kappa()
print(f"\n>>> Calibrated kappa* = {kappa_star:.4f}")
print(f"Final training share = {training_share(kappa_star):.4f}")
