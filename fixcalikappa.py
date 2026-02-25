import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import root
from scipy.optimize import brentq
from scipy.stats import lognorm
# =====================================================
# PARAMETERS
# =====================================================
params = {
    "alpha": 1.0,  # guess
    "beta": 1.0,
    "gamma_bar": 2.0,  # guess
    "kappa": 5.0,  # guess
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
kappa = params["kappa"]
B = params["B"]

# GRID
A = 5 * kappa
Na = 400
a_grid = np.linspace(0, A, Na)
da = a_grid[1] - a_grid[0]

# =====================================================
# STEP 1: FDI EQUILIBRIUM
# =====================================================
def step1_fdi_equilibrium():
    pF = 1 / (rho + params["eta_F"])
    rF = params["alpha_F"]*pF
    xF_star = params["alpha_F"]**2 / \
        ((rho+params["eta_F"])*(params["eta_F"]-params["delta_F"]))
    return pF, rF, xF_star


# Gọi hàm để lấy giá trị
pf_val, rf_val,  xf_star_val = step1_fdi_equilibrium()
# In ra màn hình với định dạng 4 chữ số thập phân
print(f"Giá trị pF tính được: {pf_val:.4f}")
print(f"Giá trị pF tính được: {rf_val:.4f}")
print(f"Giá trị xF* tính được: {xf_star_val:.4f}")

# =====================================================
# STEP 2: INITIALIZE POLICY
# =====================================================


def step2_initialize_policy():
    Ba0 = B / 2
    BR0 = B - Ba0
    return Ba0, BR0
# =====================================================
# STEP 3a: Solve for h(a) = g'(a)
# =====================================================

def solve_h(a_grid, sR, sa, params):
    gamma_bar = params["gamma_bar"]
    kappa = params["kappa"]
    beta = params["beta"]
    rho = params["rho"]
    sigma_a = params["sigma_a"]
    alpha = params["alpha"]  
    theta = params["theta"]  
    xi = params["xi"]        
    lamb = params["lambda"]  

    p = 1 / (rho + params["eta"])   # p của domestic

    Na = len(a_grid)
    da = a_grid[1] - a_grid[0]

    # initial guess
    h = np.zeros(Na)

    # simple fixed-point iteration (stable for this model)
    for iteration in range(500):

        # controls from previous h
        gamma_a = gamma_bar * a_grid**2 / (kappa**2 + a_grid**2)
        r = sR + params["alpha"]*gamma_a * p
        e = beta * h

        gamma_prime = gamma_bar * (2 * a_grid * kappa**2) / (kappa**2 + a_grid**2)**2 #gamma'(a)

        F_prime = gamma_prime * (p * lamb * xf_star_val + alpha * p * (sR + alpha * p * gamma_a))

        # finite difference update (upwind)
        h_new = np.zeros_like(h)

        for i in range(1, Na-1):
            h_x = (h[i+1] - h[i-1]) / (2*da)  # central difference
            h_xx = (h[i+1] - 2*h[i] + h[i-1]) / da**2
            h_new[i] = ((F_prime[i] + beta**2 * h[i] * h_x + h_x * (theta * sa - xi * a_grid[i]) - xi * h[i] + (sigma_a**2/2) * h_xx))/ rho

        # boundary
        h_new[0] = 0        # h(0) = 0 (từ Neumann: g'(0)=0)
        h_new[-1] = 0        # h(A) = 0 (decay về 0 khi a→∞)

        # Check convergence
        if np.max(np.abs(h_new - h)) < 1e-8:
            print(f"Converged after {iteration+1} iterations")
            break

        # Relaxation để ổn định
        h = 0.3*h + 0.7*h_new
    
    return h

def recover_g(h, a_grid):  #đây là hàm g(a)
    g = cumulative_trapezoid(h, a_grid, initial=0.0)
    return g
# =====================================================
# STEP 3b — controls
# =====================================================
def compute_controls(h, a_grid, sR, params):
    gamma_bar = params["gamma_bar"]
    kappa = params["kappa"]
    beta = params["beta"]
    alpha = params["alpha"]
    p = 1 / (params["rho"] + params["eta"])
    
    gamma_a = gamma_bar * a_grid**2 / (a_grid**2 + kappa**2)  # S-shaped
    r_star = sR + alpha * gamma_a * p  # r*(a) = sR + αγ(a)p
    e_star = beta * h  # e*(a) = βg'(a)
    
    return r_star, e_star, gamma_a
# =====================================================
# STEP 3c — stationary FP distribution
# =====================================================
def stationary_density(h, g, a_grid, sa):
    # Drift
    mu = beta**2 * h + theta * sa - xi * a_grid  #muy(a)
    
    # Potential 
    Phi = beta**2 * g + theta * sa * a_grid - (xi/2) * a_grid**2
    
    # m(a) ∝ exp(2Φ/σ_a²)
    exponent  = (2 / sigma_a**2) * Phi
    exponent =  exponent  - np.max(exponent)  # numerical stability
    m = np.exp(exponent)
    
    # Normalize
    Z = np.trapezoid(m, a_grid) #tính tích phân trapeznoid tự hiểu 2 cận vì đã tính da
    m = m / Z
    
    return m, mu

# =====================================================
# STEP 3d — compute moments
# =====================================================
def compute_moments(a_grid, m, gamma_a, sR):
    alpha = params["alpha"]
    lamb = params["lambda"]  
    eta = params["eta"]  
    delta = params["delta"]  
    p = 1 / (rho + params["eta"])   # p của domestic
    abar = np.trapezoid(a_grid * m, a_grid)
    Gamma = np.trapezoid(gamma_a * m, a_grid)
    Gamma2 = np.trapezoid((gamma_a**2) * m, a_grid)
    # =====================================================
    # STEP 3e: 
    # =====================================================
    xD = (alpha * (sR * Gamma + alpha * p * Gamma2) + Gamma * lamb * xf_star_val) / (eta - delta)
    return abar, Gamma, Gamma2, xD
# =====================================================
# WRAPPER STEP 3
# =====================================================
def step3_equilibrium(sR, sa, a_grid, params):
    # 3a
    h = solve_h(a_grid, sR, sa, params)
    # recover g
    g = recover_g(h, a_grid)
    # 3b
    r_star, e_star, gamma_a = compute_controls(h, a_grid, sR, params)
    # 3c
    m, mu = stationary_density(h, g, a_grid, sa)
    # 3d
    abar, Gamma, Gamma2, xD = compute_moments(a_grid, m, gamma_a, sR)

    return {
        "h": h,
        "g": g,
        "r_star": r_star,
        "e_star": e_star,
        "mu":mu,
        "m": m,
        "abar": abar,
        "Gamma": Gamma,
        "Gamma2": Gamma2,
        "xD": xD,
    }

# =====================================================
# CALIBRATION TARGETS
# =====================================================
TARGET_TRAINING = 0.087   # % firms offer training muy(Kappa)
TARGET_RD = 0.051         # % firms do R&D
TARGET_RATIO = 2.0        # xF / xD ratio

# Baseline policy (từ paper)
sR_baseline = 0.1 #at  At Vietnam's baseline policy (sR ≈ 0.1) ý là đoạn này là giả định sau có thể thử đổi cái này 
sa_baseline = 0.3 # at Vietnam's current policy level (sa≈0.3) có vẻ đoạn này là stylized fact nè 

# =====================================================
# STEP 4 — Calibrate kappa từ training share
# =====================================================
def training_share(kappa_val, sR, sa, params):
    """Tính % firms có a >= kappa"""
    params_tmp = params.copy()
    params_tmp["kappa"] = kappa_val

    eq = step3_equilibrium(sR, sa, a_grid, params_tmp)
    m = eq["m"]

    # Firms với a >= kappa
    high_idx = a_grid >= kappa_val

    share = np.trapezoid(m[high_idx], a_grid[high_idx])
    return share

print(training_share(0.1, sR_baseline, sa_baseline, params))
print(training_share(10.0, sR_baseline, sa_baseline, params))

def calibrate_kappa(sR, sa, params):
    """Tìm kappa để training_share = 8.7%"""
    print("\n=== Calibrating kappa ===")

    def obj(kappa_val):
        share = training_share(kappa_val, sR, sa, params)
        print(f"  kappa={kappa_val:.3f} → training={share:.4f}")
        return share - TARGET_TRAINING

    # Search từ 1 đến 10
    kappa_star = brentq(obj, 1.0, 10.0)
    print(f"✓ kappa* = {kappa_star:.4f}")
    return kappa_star
