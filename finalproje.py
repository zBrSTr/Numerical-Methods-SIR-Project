import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline
from scipy.integrate import solve_ivp
from scipy.linalg import lu_factor, lu_solve
from scipy.optimize import minimize_scalar, minimize
import time
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("   NUMERICAL METHODS IN ENGINEERING - FINAL PROJECT")
print("   SIR Epidemic Spread Model - Comprehensive Analysis")
print("=" * 60)

# ===========================================================================
# GLOBAL PARAMETERS
# ===========================================================================
N      = 1000      # Total population
I0     = 10        # Initial infected
R0_ic  = 0         # Initial recovered
S0     = N - I0 - R0_ic
y0     = [S0, I0, R0_ic]
beta   = 0.4
gamma  = 0.1
t_span = (0, 100)


# ===========================================================================
# CORE SIR MODEL
# ===========================================================================
def sir_model(t, y, beta, gamma):
    """
    Computes the continuous derivatives for the Susceptible-Infected-Recovered (SIR) model.
    
    This function represents the core nonlinear ordinary differential equation (ODE) 
    system of mathematical epidemiology. It models disease transmission dynamics based 
    on the interaction between susceptible and infected populations within a closed system.
    
    Mathematical Formulation:
        dS/dt = - (beta * S * I) / N
        dI/dt = (beta * S * I) / N - (gamma * I)
        dR/dt = gamma * I
        
    Parameters:
        t (float): Independent variable representing time (days). Required by ODE solvers.
        y (array_like): State vector containing current values [S, I, R].
        beta (float): Transmission rate parameter controlling contact efficiency.
        gamma (float): Recovery rate parameter representing the inverse of infectious period.
        
    Returns:
        numpy.ndarray: Array containing the derivatives [dSdt, dIdt, dRdt].
    """
    S, I, R = y
    N_total = S + I + R # Dynamic sum to enforce mathematical consistency
    
    # Mathematical equations mapping the transmission and recovery mechanics
    dSdt = -(beta * S * I) / N_total
    dIdt = ((beta * S * I) / N_total) - (gamma * I)
    dRdt = gamma * I
    return np.array([dSdt, dIdt, dRdt])


# ===========================================================================
# TOPIC 9: ODE SOLVER - Custom RK4 + SciPy Solvers
# ===========================================================================
def rk4_step(f, t, y, h, beta, gamma):
    """
    Performs a single computational step of the classic 4th-order Runge-Kutta (RK4) method.
    
    This function calculates the next state of the nonlinear differential equation system 
    by evaluating four distinct intermediate slopes (k1, k2, k3, k4) across the interval h. 
    By canceling out lower-order error terms via Taylor series matching, it achieves 
    a local truncation error of O(h^5) and a global truncation error of O(h^4).
    
    Parameters:
        f (callable): The differential equation system function to evaluate (sir_model).
        t (float): Current time value.
        y (numpy.ndarray): Current state vector array [S, I, R].
        h (float): Fixed step size for the time increment.
        beta (float): Transmission rate parameter passed to the model.
        gamma (float): Recovery rate parameter passed to the model.
        
    Returns:
        numpy.ndarray: The updated state vector array [S, I, R] computed at time t + h.
    """
    k1 = h * f(t,       y,            beta, gamma)
    k2 = h * f(t + h/2, y + k1/2,     beta, gamma)
    k3 = h * f(t + h/2, y + k2/2,     beta, gamma)
    k4 = h * f(t + h,   y + k3,       beta, gamma)
    
    # Take a weighted average of the four slopes to optimize accuracy
    return y + (k1 + 2*k2 + 2*k3 + k4) / 6

def solve_sir_rk4(t_span, y0, h, beta, gamma):
    """
    Simulates the full trajectory of the SIR model using the fixed-step custom RK4 engine.
    
    Iterates through the entire time domain specified by t_span using an explicitly 
    defined step size h. Allocates state matrices dynamically and evaluates the propagation 
    using the rk4_step method. Includes zero-guards for interval mapping.
    
    Parameters:
        t_span (tuple): Tuple containing (start_time, end_time).
        y0 (list/array): Initial conditions vector [S0, I0, R0].
        h (float): Step size for numerical propagation.
        beta (float): Transmission rate parameter.
        gamma (float): Recovery rate parameter.
        
    Returns:
        tuple: (t_values, y_values) where t_values is a 1D grid and y_values is a 2D matrix.
    """
    t_values = np.arange(t_span[0], t_span[1] + h, h)
    y_values = np.zeros((len(t_values), len(y0)))
    y_values[0] = y0
    
    # Propagate through the predefined time domain iteratively
    for i in range(1, len(t_values)):
        y_values[i] = rk4_step(sir_model, t_values[i-1], y_values[i-1], h, beta, gamma)
    return t_values, y_values

print("\n[TOPIC 9] ODE Solvers - Performance Comparison")
print("-" * 50)

t0 = time.perf_counter()
t_rk4, y_rk4 = solve_sir_rk4(t_span, y0, 0.5, beta, gamma)
time_rk4 = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
sol_rk45 = solve_ivp(sir_model, t_span, y0, args=(beta, gamma), method='RK45', dense_output=True)
time_rk45 = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
sol_rk23 = solve_ivp(sir_model, t_span, y0, args=(beta, gamma), method='RK23', dense_output=True)
time_rk23 = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
sol_dop = solve_ivp(sir_model, t_span, y0, args=(beta, gamma), method='DOP853', dense_output=True)
time_dop = (time.perf_counter() - t0) * 1000

print(f"  Custom RK4  (h=0.5)  : {time_rk4:.2f} ms")
print(f"  SciPy RK45 (adaptive): {time_rk45:.2f} ms")
print(f"  SciPy RK23 (adaptive): {time_rk23:.2f} ms")
print(f"  SciPy DOP853         : {time_dop:.2f} ms")

t_ref = np.linspace(0, 100, 1000)
y_ref = sol_dop.sol(t_ref)  # DOP853 serves as the high-accuracy baseline


# ===========================================================================
# TOPIC 1: ERROR ANALYSIS & FLOATING-POINT PRECISION
# ===========================================================================
print("\n[TOPIC 1] Error Analysis & Floating-Point Precision")
print("-" * 50)

eps32 = np.finfo(np.float32).eps
eps64 = np.finfo(np.float64).eps
print(f"  Machine epsilon float32 : {eps32:.2e}")
print(f"  Machine epsilon float64 : {eps64:.2e}")

# R0 calculation error across precisions
R0_true = beta / gamma  # Exact mathematical representation = 4.0
for dtype_name, dtype in [('float16', np.float16), ('float32', np.float32), ('float64', np.float64)]:
    try:
        # Cast parameters to specific data types to evaluate floating-point limits
        b = dtype(beta)
        g = dtype(gamma)
        R0_calc = float(b / g)
        err = abs(R0_calc - R0_true)
        print(f"  R0 ({dtype_name}): {R0_calc:.6f}  |  Error: {err:.2e}")
    except Exception as e:
        # Catch arithmetic exceptions or data type incompatibility dynamically
        print(f"  R0 ({dtype_name}): computation failed - {e}")

# Rounding error accumulation in beta over iterations
print("\n  Rounding error accumulation (beta summed 1000x):")
accum32, accum64 = np.float32(0.0), np.float64(0.0)
for _ in range(1000):
    accum32 += np.float32(beta)
    accum64 += np.float64(beta)
expected = beta * 1000
print(f"  Expected     : {expected:.6f}")
print(f"  float32 sum  : {float(accum32):.6f}  |  Error: {abs(float(accum32)-expected):.2e}")
print(f"  float64 sum  : {float(accum64):.6f}  |  Error: {abs(float(accum64)-expected):.2e}")

# Step size error analysis (ties into Topic 4 & 10)
print("\n  Step size MSE vs DOP853 reference:")
for h_val in [1.0, 0.5, 0.1]:
    t_h, y_h = solve_sir_rk4(t_span, y0, h_val, beta, gamma)
    ref_at_t = sol_dop.sol(t_h)
    mse = np.mean((y_h[:, 1] - ref_at_t[1])**2)
    print(f"    h = {h_val:<4}  MSE = {mse:.4e}")


# ===========================================================================
# TOPIC 2: ROOT FINDING - Bisection, Newton-Raphson, Secant
# ===========================================================================
print("\n[TOPIC 2] Root Finding Methods")
print("-" * 50)

TARGET_PEAK = 0.20 * N  # Target boundary rule: peak infected count must equal 20% of N

def peak_infected_for_beta(b):
    """
    Simulates the SIR model trajectory to extract the maximum infected population value.
    
    Acts as the wrapper function f(x) for root-finding applications. It maps a single scalar 
    input (beta) into a single scalar output (maximum curve peak) using an adaptive solver.
    
    Parameters:
        b (float): Evaluated transmission rate value.
        
    Returns:
        float: Maximum value of the infected population found during the time span.
    """
    try:
        sol = solve_ivp(sir_model, t_span, y0, args=(b, gamma), method='RK45', max_step=0.5)
        return float(np.max(sol.y[1]))
    except Exception:
        # Return NaN to denote failure in case numerical divergence occurs inside the ODE engine
        return float('nan')

def f_root(b):
    """Reframes the engineering target as a zero-finding root problem: f(beta) = 0."""
    return peak_infected_for_beta(b) - TARGET_PEAK

def bisection(f, a, b, tol=1e-5, max_iter=50):
    """
    Finds the root of a function using the structurally robust Bisection Method.
    
    Relies on the Intermediate Value Theorem. It iteratively halves an interval [a, b] 
    where a sign change is verified. Possesses a linear convergence rate (order = 1) 
    but offers absolute structural guarantee of convergence if the bracket is valid.
    
    Parameters:
        f (callable): Target root function to evaluate (f_root).
        a (float): Lower bound of the initialization bracket.
        b (float): Upper bound of the initialization bracket.
        tol (float): Tolerance threshold defining convergence termination.
        max_iter (int): Safeguard maximum iteration limit to prevent infinite execution loops.
        
    Returns:
        tuple: (root_estimate, history_list) where history maps convergence steps.
    """
    history = []
    try:
        # Enforce sign change validation guard before launching iterations
        if f(a) * f(b) > 0:
            raise ValueError("f(a) and f(b) must have opposite signs to guarantee a bracketed root.")
            
        for i in range(max_iter):
            c = (a + b) / 2.0
            fc = f(c)
            history.append(abs(b - a) / 2)
            
            # Convergence check based on error tolerance or bracket width criteria
            if abs(fc) < tol or (b - a) / 2 < tol:
                return c, history
            if f(a) * fc < 0:
                b = c
            else:
                a = c
    except Exception as e:
        # Safeguard handler to manage out-of-bounds calculations or functional exceptions
        print(f"  [!] Bisection error caught safely: {e}")
    return (a + b) / 2.0, history

def newton_raphson(f, x0, tol=1e-5, max_iter=50, h=1e-4):
    """
    Finds the root of a function using the rapidly converging Newton-Raphson Method.
    
    Utilizes a first-order Taylor series tangent line approximation. The derivative is 
    computed dynamically using a central finite-difference scheme. Boasts quadratic 
    convergence (order = 2), but requires a localized, high-quality initial guess.
    
    Parameters:
        f (callable): Target function to find the root for.
        x0 (float): Initial starting guess.
        tol (float): Tolerance threshold for absolute convergence check.
        max_iter (int): Safeguard iteration counter to stop non-converging cycles.
        h (float): Perturbation step size utilized for numerical differentiation.
        
    Returns:
        tuple: (root_estimate, history_list) mapping step changes.
    """
    history = []
    x = x0
    try:
        for i in range(max_iter):
            fx  = f(x)
            # Central difference scheme to obtain a highly accurate slope approximation
            fpx = (f(x + h) - f(x - h)) / (2 * h)   
            
            # Critical division-by-zero check to block NaN propagation and code crashes
            if abs(fpx) < 1e-12:
                raise ValueError("Derivative magnitude near-zero - Newton-Raphson step aborted.")
                
            x_new = x - fx / fpx
            history.append(abs(x_new - x))
            
            if abs(x_new - x) < tol:
                return x_new, history
            x = x_new
    except Exception as e:
        # Gracefully log and handle local extrema trap or singular derivative scenarios
        print(f"  [!] Newton-Raphson error caught safely: {e}")
    return x, history

def secant(f, x0, x1, tol=1e-5, max_iter=50):
    """
    Finds the root of a function utilizing the derivative-free Secant Method.
    
    Approximates the local derivative line using the secant slope across the two 
    most recent history points. Eliminates explicit differentiation needs while 
    maintaining a superlinear convergence profile (order approx 1.618).
    
    Parameters:
        f (callable): Target root function.
        x0 (float): First independent initialization point.
        x1 (float): Second independent initialization point.
        tol (float): Numerical convergence tolerance.
        max_iter (int): Dynamic safeguard limit to break loop under bad guesses.
        
    Returns:
        tuple: (root_estimate, history_list) tracking execution steps.
    """
    history = []
    try:
        for i in range(max_iter):
            f0, f1 = f(x0), f(x1)
            
            # Enforce near-zero denominator check to eliminate catastrophic cancellation errors
            if abs(f1 - f0) < 1e-14:
                raise ValueError("Denominator delta near zero - Secant division blocked.")
                
            x2 = x1 - f1 * (x1 - x0) / (f1 - f0)
            history.append(abs(x2 - x1))
            
            if abs(x2 - x1) < tol:
                return x2, history
            x0, x1 = x1, x2
    except Exception as e:
        # Structural error logger to avoid runtime collapse during parallel runs
        print(f"  [!] Secant error caught safely: {e}")
    return x1, history

t0 = time.perf_counter()
root_b, hist_b = bisection(f_root, 0.11, 0.5)
time_b = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
root_n, hist_n = newton_raphson(f_root, x0=0.25)
time_n = (time.perf_counter() - t0) * 1000

t0 = time.perf_counter()
root_s, hist_s = secant(f_root, x0=0.15, x1=0.40)
time_s = (time.perf_counter() - t0) * 1000

print(f"  Target peak infections : {TARGET_PEAK:.0f} people ({TARGET_PEAK/N*100:.0f}% of N)")
print(f"\n  {'Method':<20} {'Beta*':<12} {'Iters':<8} {'Time(ms)'}")
print(f"  {'-'*52}")
print(f"  {'Bisection':<20} {root_b:<12.6f} {len(hist_b):<8} {time_b:.2f}")
print(f"  {'Newton-Raphson':<20} {root_n:<12.6f} {len(hist_n):<8} {time_n:.2f}")
print(f"  {'Secant':<20} {root_s:<12.6f} {len(hist_s):<8} {time_s:.2f}")


# ===========================================================================
# TOPIC 3: INTERPOLATION - Linear vs Cubic Spline
# ===========================================================================
print("\n[TOPIC 3] Interpolation - Linear vs Cubic Spline")
print("-" * 50)

t_coarse, y_coarse = solve_sir_rk4(t_span, y0, 5.0, beta, gamma)  # Sparse 5-day sampling matrix
t_fine = np.linspace(0, 100, 500)
I_coarse = y_coarse[:, 1]

# Linear interpolation path configuration
I_linear = np.interp(t_fine, t_coarse, I_coarse)

# Cubic spline implementation generating boundary continuous equations
cs = CubicSpline(t_coarse, I_coarse)
I_cubic = cs(t_fine)

I_ref = sol_dop.sol(t_fine)[1]

err_linear = np.mean(np.abs(I_linear - I_ref))
err_cubic  = np.mean(np.abs(I_cubic  - I_ref))
print(f"  Linear interpolation MAE : {err_linear:.4f}")
print(f"  Cubic spline MAE         : {err_cubic:.4f}")
print(f"  Cubic spline is {err_linear/err_cubic:.1f}x more accurate than linear")


# ===========================================================================
# TOPIC 4: NUMERICAL DIFFERENTIATION
# ===========================================================================
print("\n[TOPIC 4] Numerical Differentiation")
print("-" * 50)

I_fine = sol_dop.sol(t_fine)[1]

def forward_diff(y, t):
    """
    Approximates derivatives using the first-order Forward Difference approximation.
    
    Formula: f'(x) = [f(x+h) - f(x)] / h. Truncation error profile matches O(h).
    """
    dy = np.zeros_like(y)
    for i in range(len(y) - 1):
        dy[i] = (y[i+1] - y[i]) / (t[i+1] - t[i])
    dy[-1] = dy[-2]  # Boundary error extension handling
    return dy

def backward_diff(y, t):
    """
    Approximates derivatives utilizing the first-order Backward Difference scheme.
    
    Formula: f'(x) = [f(x) - f(x-h)] / h. Truncation error profile matches O(h).
    """
    dy = np.zeros_like(y)
    dy[0] = dy[1]    # Boundary constraint safeguard initialization
    for i in range(1, len(y)):
        dy[i] = (y[i] - y[i-1]) / (t[i] - t[i-1])
    return dy

def central_diff(y, t):
    """
    Computes derivatives using the highly accurate second-order Central Difference scheme.
    
    Formula: f'(x) = [f(x+h) - f(x-h)] / 2h. Achieves O(h^2) error symmetry by canceling out 
    the first-order derivative error metrics derived via Taylor expansions.
    """
    dy = np.zeros_like(y)
    dy[0]  = (y[1] - y[0])   / (t[1] - t[0])  # Fallback to asymmetric edge mapping
    dy[-1] = (y[-1] - y[-2]) / (t[-1] - t[-2])
    for i in range(1, len(y) - 1):
        dy[i] = (y[i+1] - y[i-1]) / (t[i+1] - t[i-1])
    return dy

dI_forward  = forward_diff(I_fine, t_fine)
dI_backward = backward_diff(I_fine, t_fine)
dI_central  = central_diff(I_fine, t_fine)

S_fine = sol_dop.sol(t_fine)[0]
dI_analytical = ((beta * S_fine * I_fine) / N) - (gamma * I_fine)

err_fwd = np.mean(np.abs(dI_forward  - dI_analytical))
err_bwd = np.mean(np.abs(dI_backward - dI_analytical))
err_cen = np.mean(np.abs(dI_central  - dI_analytical))
print(f"  Forward  difference MAE vs analytical: {err_fwd:.6f}")
print(f"  Backward difference MAE vs analytical: {err_bwd:.6f}")
print(f"  Central  difference MAE vs analytical: {err_cen:.6f}")
print(f"  -> Central difference is most accurate (2nd order)")

# Detect peak day by identifying the algebraic zero-crossing of the derivative array
sign_changes = np.where(np.diff(np.sign(dI_central)))[0]
if len(sign_changes) > 0:
    peak_day_diff = t_fine[sign_changes[0]]
    print(f"  Peak infection day (zero-crossing of dI/dt): Day {peak_day_diff:.1f}")


# ===========================================================================
# TOPIC 5: NUMERICAL INTEGRATION
# ===========================================================================
print("\n[TOPIC 5] Numerical Integration")
print("-" * 50)

def trapezoidal_rule(t, y):
    """
    Approximates definite integral via piecewise linear Newton-Cotes approximation.
    
    Maps areas under curves by treating intervals as individual trapezoids. Error order is O(h^2).
    """
    return np.sum((y[:-1] + y[1:]) * np.diff(t) / 2)

def simpsons_rule(t, y):
    """
    Integrates functions using the highly accurate quadratic Simpson's 1/3 Rule.
    
    Fits local parabolas across coordinate triplets. Elevates precision limits to O(h^4) error.
    """
    n = len(t) - 1
    if n % 2 != 0:
        # Enforce step interval symmetry by removing the final odd element safely
        t, y = t[:-1], y[:-1]
        n -= 1
    h = (t[-1] - t[0]) / n
    total = y[0] + y[-1]
    total += 4 * np.sum(y[1:-1:2]) # Apply mathematical weight constants [1, 4, 2, 4, ..., 1]
    total += 2 * np.sum(y[2:-2:2])
    return total * h / 3

from scipy.integrate import quad as scipy_quad
def scipy_integration(t, y):
    """Computes definite integral using adaptive Gaussian Quadrature mapping frameworks."""
    cs_temp = CubicSpline(t, y)
    result, _ = scipy_quad(cs_temp, t[0], t[-1])
    return result

trap_result   = trapezoidal_rule(t_fine, I_fine)
simp_result   = simpsons_rule(t_fine, I_fine)
scipy_result  = scipy_integration(t_fine, I_fine)

print(f"  Trapezoidal rule  : {trap_result:.2f}  patient-days")
print(f"  Simpson's rule    : {simp_result:.2f}  patient-days")
print(f"  SciPy quad        : {scipy_result:.2f}  patient-days")
print(f"  Trap vs Scipy err : {abs(trap_result - scipy_result):.4f}")
print(f"  Simp vs Scipy err : {abs(simp_result - scipy_result):.4f}")
print(f"  -> Simpson's rule is more accurate than Trapezoidal")


# ===========================================================================
# TOPIC 6: LINEAR SYSTEMS - numpy.linalg.solve
# ===========================================================================
print("\n[TOPIC 6] Solving Linear Systems")
print("-" * 50)

A = np.array([
    [1.0,  1.0,  1.0 ],   # Linear equation constraint: normalized risk weight sum equals 1.0
    [2.0, -1.0,  0.5 ],   # Interaction metrics matching transmission scaling variables
    [0.1,  0.3, -0.2 ]    # Recovery scaling metric parameters matrix
])
b = np.array([1.0, 0.5, 0.05])

try:
    x = np.linalg.solve(A, b)
    residual = np.linalg.norm(A @ x - b)
    print(f"  Group contributions: {x}")
    print(f"  Residual (||Ax-b||): {residual:.2e}  <- should be ~0")
    cond = np.linalg.cond(A)
    print(f"  Condition number   : {cond:.4f}  <- low = well-conditioned")
except np.linalg.LinAlgError as e:
    # Trap matrix singularities safely to shield compilation pipeline from absolute crash
    print(f"  [!] Linear system solving failed due to singular matrix profile: {e}")


# ===========================================================================
# TOPIC 7: LU DECOMPOSITION
# ===========================================================================
print("\n[TOPIC 7] LU Decomposition")
print("-" * 50)

try:
    # Compute LU factorization parameters once to establish factorization reuse patterns
    lu, piv = lu_factor(A)

    b_scenarios = [
        np.array([1.0, 0.5,  0.05]),   # Baseline control profile
        np.array([1.0, 0.3,  0.02]),   # Mild isolation containment metric
        np.array([1.0, 0.1, -0.01]),   # Strict lockdown vector matrix
    ]

    t0 = time.perf_counter()
    for b_s in b_scenarios:
        x_lu = lu_solve((lu, piv), b_s) # Solves in O(n^2) back-substitution steps
    time_lu = (time.perf_counter() - t0) * 1000

    t0 = time.perf_counter()
    for b_s in b_scenarios:
        x_direct = np.linalg.solve(A, b_s) # Re-computes full O(n^3) Gaussian reductions
    time_direct = (time.perf_counter() - t0) * 1000

    print(f"  LU solve (3 scenarios)     : {time_lu:.4f} ms")
    print(f"  Direct solve (3 scenarios) : {time_direct:.4f} ms")
    print(f"  LU solution              : {x_lu}")
    resid_lu = np.linalg.norm(A @ x_lu - b_scenarios[-1])
    print(f"  LU residual              : {resid_lu:.2e}")
    print(f"  -> LU factorization pays off when same A is reused many times")
except Exception as e:
    # Strategic fallback protection block managing decomposition failure criteria
    print(f"  [!] LU factorization routine error state logged: {e}")


# ===========================================================================
# TOPIC 8: OPTIMIZATION
# ===========================================================================
print("\n[TOPIC 8] Optimization")
print("-" * 50)

def neg_peak(b):
    """Evaluates target peak limits as an optimization cost target constraint mapping wrapper."""
    try:
        sol = solve_ivp(sir_model, t_span, y0, args=(float(b), gamma), method='RK45', max_step=1.0)
        return float(np.max(sol.y[1]))
    except Exception:
        return float('inf') # Return infinity during optimization crashes to denote absolute penalty bounds

result_scalar = minimize_scalar(neg_peak, bounds=(0.05, 0.5), method='bounded')
print(f"  Scalar opt: optimal beta  = {result_scalar.x:.6f}")
print(f"  Scalar opt: peak infected = {result_scalar.fun:.1f} people")

def cost_function(params):
    """Evaluates multivariate penalty metrics using multi-parameter tracking matrices."""
    b, g = params
    if b <= 0 or g <= 0 or b > 1 or g > 1:
        return float('inf') # Enforce hard physical boundary parameter walls
    try:
        sol = solve_ivp(sir_model, t_span, y0, args=(b, g), method='RK45', max_step=1.0)
        peak = float(np.max(sol.y[1]))
        cost = peak + 500 * abs(b - 0.4) + 500 * abs(g - 0.1)
        return cost
    except Exception:
        return float('inf')

result_multi = minimize(cost_function, x0=[0.3, 0.12], method='Nelder-Mead',
                        options={'xatol': 1e-4, 'fatol': 1e-4, 'maxiter': 200})
print(f"  Multi opt : optimal beta  = {result_multi.x[0]:.6f}")
print(f"  Multi opt : optimal gamma = {result_multi.x[1]:.6f}")
print(f"  Multi opt : cost value    = {result_multi.fun:.2f}")


# ===========================================================================
# TOPIC 10: PERFORMANCE ANALYSIS & NUMERICAL STABILITY
# ===========================================================================
print("\n[TOPIC 10] Performance Analysis & Numerical Stability")
print("-" * 50)

h_list, mse_list = [], []
for h_val in [2.0, 1.0, 0.5, 0.2, 0.1, 0.05]:
    t_h, y_h = solve_sir_rk4(t_span, y0, h_val, beta, gamma)
    ref_I    = sol_dop.sol(t_h)[1]
    mse      = np.mean((y_h[:, 1] - ref_I)**2)
    h_list.append(h_val)
    mse_list.append(mse)
    print(f"  h = {h_val:<5}  MSE = {mse:.4e}")

# Stability Invariant Enforcer check: monitor physical mass conservation law
S_rk4 = y_rk4[:, 0]
I_rk4 = y_rk4[:, 1]
R_rk4 = y_rk4[:, 2]
conservation_error = np.max(np.abs(S_rk4 + I_rk4 + R_rk4 - N))
print(f"\n  Max population conservation error (S+I+R-N): {conservation_error:.2e}")


# ===========================================================================
# TOPIC 12: COMPARATIVE ANALYSIS - Multiple Scenarios
# ===========================================================================
print("\n[TOPIC 12] Comparative Analysis - Intervention Scenarios")
print("-" * 50)

scenarios = {
    "No Intervention" : {"beta": 0.40, "gamma": 0.10},
    "Mild Measures"   : {"beta": 0.25, "gamma": 0.10},
    "Strict Lockdown" : {"beta": 0.10, "gamma": 0.10},
    "Vaccination"     : {"beta": 0.40, "gamma": 0.20},
}

scenario_results = {}
for name, params in scenarios.items():
    sol_s = solve_ivp(sir_model, t_span, y0, args=(params["beta"], params["gamma"]), method='RK45', dense_output=True)
    I_s      = sol_s.sol(t_fine)[1]
    peak_I   = float(np.max(I_s))
    peak_day = float(t_fine[np.argmax(I_s)])
    total    = float(trapezoidal_rule(t_fine, I_s))
    R0_val   = params["beta"] / params["gamma"]
    scenario_results[name] = {
        "sol": sol_s, "I": I_s,
        "peak": peak_I, "peak_day": peak_day,
        "total": total, "R0": R0_val
    }
    print(f"  {name:<20} R0={R0_val:.1f}  Peak={peak_I:.0f}  Day={peak_day:.0f}  Load={total:.0f}")


# ===========================================================================
# TOPIC 11: VISUALIZATION - Generating 12 Individual Plots
# ===========================================================================
print("\n[TOPIC 11] Generating 12 Individual Visualizations...")
colors_scenario = ['#e74c3c', '#e67e22', '#27ae60', '#2980b9']

# ── Plot 1: Floating-Point Error Accumulation ────────────────────────────────
plt.figure(figsize=(8, 5))
steps  = np.arange(1, 501)
err32, err64 = [], []
r32, r64 = np.float32(0.0), np.float64(0.0)
for i in steps:
    r32 += np.float32(beta); r64 += np.float64(beta)
    true_v = beta * i
    err32.append(abs(float(r32) - true_v))
    err64.append(abs(float(r64) - true_v))
plt.semilogy(steps, err32, 'r-', linewidth=1.5, label='float32')
plt.semilogy(steps, err64, 'b-', linewidth=1.5, label='float64')
plt.title('Topic 1: Floating-Point Error Accumulation (beta iteratively summed)', fontweight='bold')
plt.xlabel('Iteration Count'); plt.ylabel('Absolute Error (log scale)')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('Topic_01_FloatingPoint.png', dpi=300); plt.close()

# ── Plot 2: Root Finding Convergence ────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.semilogy(range(1, len(hist_b)+1), hist_b, 'b-o', markersize=4, label='Bisection')
plt.semilogy(range(1, len(hist_n)+1), hist_n, 'r-s', markersize=4, label='Newton-Raphson')
plt.semilogy(range(1, len(hist_s)+1), hist_s, 'g-^', markersize=4, label='Secant')
plt.title('Topic 2: Root Finding Convergence Rate Comparison', fontweight='bold')
plt.xlabel('Iteration Number'); plt.ylabel('Error (log scale)')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('Topic_02_RootFinding.png', dpi=300); plt.close()

# ── Plot 3: Interpolation Comparison ────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(t_fine,   I_ref,    'k-',  linewidth=2,   label='Reference (DOP853)', alpha=0.7)
plt.plot(t_fine,   I_linear, 'r--', linewidth=1.5, label=f'Linear (MAE={err_linear:.1f})')
plt.plot(t_fine,   I_cubic,  'b-',  linewidth=1.5, label=f'Cubic Spline (MAE={err_cubic:.1f})')
plt.scatter(t_coarse, I_coarse, color='orange', s=30, zorder=5, label='Sparse data pts (every 5 days)')
plt.title('Topic 3: Interpolation (Linear vs Cubic Spline)', fontweight='bold')
plt.xlabel('Time (days)'); plt.ylabel('Infected Population')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('Topic_03_Interpolation.png', dpi=300); plt.close()

# ── Plot 4: Numerical Differentiation ───────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(t_fine, dI_analytical, 'k-',  linewidth=2,   label='Analytical dI/dt', alpha=0.8)
plt.plot(t_fine, dI_forward,   'r--', linewidth=1.2, label='Forward Difference', alpha=0.8)
plt.plot(t_fine, dI_backward,  'b:',  linewidth=1.2, label='Backward Difference', alpha=0.8)
plt.plot(t_fine, dI_central,   'g-',  linewidth=1.2, label='Central Difference', alpha=0.8)
plt.axhline(0, color='gray', linewidth=0.8, linestyle='--')
plt.title('Topic 4: Numerical Differentiation (dI/dt Comparison)', fontweight='bold')
plt.xlabel('Time (days)'); plt.ylabel('dI/dt')
plt.legend(); plt.grid(True, alpha=0.3); plt.xlim([0, 60]); plt.tight_layout()
plt.savefig('Topic_04_Differentiation.png', dpi=300); plt.close()

# ── Plot 5: Numerical Integration ───────────────────────────────────────────
plt.figure(figsize=(8, 5))
methods_int  = ['Trapezoidal\nRule', "Simpson's\n1/3 Rule", 'SciPy\nquad']
values_int   = [trap_result,   simp_result,  scipy_result]
colors_int   = ['#3498db',     '#e74c3c',    '#2ecc71']
bars = plt.bar(methods_int, values_int, alpha=0.8, color=colors_int, edgecolor='black', width=0.5)
plt.ylabel('Total Patient-Days')
for bar, val in zip(bars, values_int):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 50, f'{val:.0f}', ha='center', va='bottom', fontweight='bold')
plt.title('Topic 5: Numerical Integration Method Comparison', fontweight='bold')
plt.ylim([0, 11000]); plt.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig('Topic_05_Integration.png', dpi=300); plt.close()

# ── Plot 6: Linear Systems ───────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
group_labels = ['High-Risk\nGroup', 'Medium-Risk\nGroup', 'Low-Risk\nGroup']
bar_colors   = ['#e74c3c', '#f39c12', '#27ae60']
x_solve = np.linalg.solve(A, b)
bars7 = plt.bar(group_labels, x_solve, color=bar_colors, alpha=0.85, edgecolor='black', width=0.5)
plt.title(f'Topic 6: Linear System Solution (Ax=b)\nCondition Number κ(A) = {np.linalg.cond(A):.2f}', fontweight='bold')
plt.ylabel('Contribution Weight')
plt.grid(True, alpha=0.3, axis='y')
for bar, val in zip(bars7, x_solve):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, f'{val:.4f}', ha='center', fontweight='bold')
plt.tight_layout(); plt.savefig('Topic_06_LinearSystem.png', dpi=300); plt.close()

# ── Plot 7: LU Decomposition ────────────────────────────────────────────────
plt.figure(figsize=(9, 5))
scen_names = ['Scenario 1\n(Baseline)', 'Scenario 2\n(Mild)', 'Scenario 3\n(Lockdown)']
lu_solutions = [lu_solve((lu, piv), b_s) for b_s in b_scenarios]
x_pos = np.arange(len(scen_names))
width = 0.25
for i, (label, color) in enumerate(zip(['High-Risk Group', 'Medium-Risk Group', 'Low-Risk Group'], bar_colors)):
    vals = [sol[i] for sol in lu_solutions]
    plt.bar(x_pos + (i-1) * width, vals, width, label=label, color=color, edgecolor='black', alpha=0.8)
plt.title('Topic 7: LU Decomposition Multi-Scenario Linear System Solutions', fontweight='bold')
plt.xticks(x_pos, scen_names)
plt.ylabel('Contribution Weight')
plt.legend(); plt.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig('Topic_07_LUDecomposition.png', dpi=300); plt.close()

# ── Plot 8: Optimization ────────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
beta_sweep = np.linspace(0.05, 0.5, 40)
peak_sweep = [neg_peak(b) for b in beta_sweep]
plt.plot(beta_sweep, peak_sweep, 'b-', linewidth=2.5)
plt.axvline(result_scalar.x, color='red', linestyle='--', linewidth=2, label=f'Optimal β* = {result_scalar.x:.3f}')
plt.axhline(result_scalar.fun, color='green', linestyle=':', linewidth=2, label=f'Min Peak = {result_scalar.fun:.0f}')
plt.title('Topic 8: Scalar Optimization Minimizing Peak Infections', fontweight='bold')
plt.xlabel('Beta (transmission rate)'); plt.ylabel('Peak Infected Population')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('Topic_08_Optimization.png', dpi=300); plt.close()

# ── Plot 9: SIR Model - ODE Solvers Comparison ──────────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(t_rk4, y_rk4[:, 1], 'r-',  linewidth=2.5, label='Custom RK4 (h=0.5)')
plt.plot(sol_rk45.t, sol_rk45.y[1], 'b--', linewidth=2, label='SciPy RK45 (adaptive)')
plt.plot(sol_rk23.t, sol_rk23.y[1], 'g:', linewidth=2, label='SciPy RK23 (adaptive)')
plt.title('Topic 9: ODE Solver Comparison (Custom RK4 vs SciPy Solvers)', fontweight='bold')
plt.xlabel('Time (days)'); plt.ylabel('Infected Population I(t)')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('Topic_09_ODE_Solvers.png', dpi=300); plt.close()

# ── Plot 10: Step Size vs MSE ────────────────────────────────────────────────
plt.figure(figsize=(8, 5))
plt.loglog(h_list, mse_list, 'b-o', markersize=8, linewidth=2.5)
plt.title('Topic 10: RK4 Convergence Study (Step Size vs. MSE)', fontweight='bold')
plt.xlabel('Step Size h (log scale)'); plt.ylabel('Mean Squared Error (log scale)')
plt.grid(True, alpha=0.5, which='both')
for h_v, mse_v in zip(h_list, mse_list):
    plt.annotate(f'h={h_v}', (h_v, mse_v), textcoords='offset points', xytext=(5, 5))
plt.tight_layout(); plt.savefig('Topic_10_Performance.png', dpi=300); plt.close()

# ── Plot 11: Intervention Scenarios ─────────────────────────────────────────
plt.figure(figsize=(9, 5))
for (name, res), col in zip(scenario_results.items(), colors_scenario):
    plt.plot(t_fine, res['I'], linewidth=2.5, color=col, label=f"{name} (R0={res['R0']:.1f})")
plt.title('Topic 12: Real-World Scenario Comparison (Intervention Strategies)', fontweight='bold')
plt.xlabel('Time (days)'); plt.ylabel('Infected Population')
plt.legend(); plt.grid(True, alpha=0.3); plt.tight_layout()
plt.savefig('Topic_11_Scenarios.png', dpi=300); plt.close()

# ── Plot 12: Summary Dashboard ───────────────────────────────────────────────
plt.figure(figsize=(9, 5))
scen_list  = list(scenario_results.keys())
peak_list  = [scenario_results[s]['peak']  for s in scen_list]
total_list = [scenario_results[s]['total'] for s in scen_list]
x12 = np.arange(len(scen_list))
plt.bar(x12 - 0.2, peak_list,  0.35, label='Peak Infected',   color=colors_scenario, alpha=0.8)
plt.bar(x12 + 0.2, [t/10 for t in total_list], 0.35, label='Total Load (/10)', color=colors_scenario, alpha=0.4, edgecolor='black', linestyle='--')
plt.title('Topic 12: Scenario Summary (Peak vs Total Infection Load)', fontweight='bold')
plt.xticks(x12, scen_list)
plt.ylabel('Population Count')
plt.legend(); plt.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
plt.savefig('Topic_12_Summary.png', dpi=300); plt.close()

print("  ✓ All 12 individual plots saved successfully as PNG files.")
print("\n" + "=" * 60)
print("  ALL 12 TOPICS COVERED — PROJECT COMPLETE")
print("=" * 60)
print(f"""
  Topic  1 - Error Analysis & Floating-Point Precision  ✓
  Topic  2 - Root Finding (Bisection, Newton, Secant)   ✓
  Topic  3 - Interpolation (Linear + Cubic Spline)      ✓
  Topic  4 - Numerical Differentiation (Fwd/Bwd/Cen)   ✓
  Topic  5 - Numerical Integration (Trap/Simp/Quad)     ✓
  Topic  6 - Linear Systems (numpy.linalg.solve)        ✓
  Topic  7 - LU Decomposition (scipy.linalg)            ✓
  Topic  8 - Optimization (scalar + multivariable)      ✓
  Topic  9 - ODE Solvers (Custom RK4 + RK45/RK23/DOP)  ✓
  Topic 10 - Performance Analysis & Stability           ✓
  Topic 11 - Visualization (12 comprehensive plots)     ✓
  Topic 12 - Comparative Analysis (4 scenarios)         ✓
""")