# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: Python [conda env:base] *
#     language: python
#     name: conda-base-py
# ---

# %%
#Needed libraries
import numpy as np
from scipy.constants import G, c, pi
from scipy.integrate import solve_ivp
from scipy.interpolate import CubicSpline, PchipInterpolator
import matplotlib.pyplot as plt
import time
import os
import mpmath as mp
from concurrent.futures import ProcessPoolExecutor

#All the constants needed
Msol = 1.98847 * 10**30 #kg
R0 = 2 * G * Msol / (c**2 * 1000) #km
P0SI = c**8 / (32 * pi * G**3 * Msol**2) #Pascals
P0 = P0SI / (1.602176634 * 10**32) #MeV/fm^3

# ---------------------------------------------------------------------------
# k2(beta, yR) -- Eq. (2) of Postnikov, Prakash & Lattimer (2010), 1004.5098.
# Evaluated in arbitrary precision below beta=0.05 because the closed-form
# expression suffers catastrophic cancellation as beta -> 0 (subtracting two
# nearly-equal large terms in double precision). Low-mass/large-radius stars
# from a central-pressure sweep routinely land in that regime, so this isn't
# an edge case you can ignore.
# ---------------------------------------------------------------------------
def _k2_formula(beta, yR, dps=None):
    if dps is None:
        b, y = beta, yR
        term1 = 2 - y + 2*b*(y - 1)
        bracket = (2*b*(6 - 3*y + 3*b*(5*y - 8)
                         + 2*b**2*(13 - 11*y + b*(3*y - 2)
                                   + 2*b**2*(1 + y)))
                   + 3*(1 - 2*b)**2*term1*np.log(1 - 2*b))
        return (8.0/5.0)*b**5*(1 - 2*b)**2*term1 / bracket
    else:
        with mp.workdps(dps):
            b, y = mp.mpf(beta), mp.mpf(yR)
            term1 = 2 - y + 2*b*(y - 1)
            bracket = (2*b*(6 - 3*y + 3*b*(5*y - 8)
                             + 2*b**2*(13 - 11*y + b*(3*y - 2)
                                       + 2*b**2*(1 + y)))
                       + 3*(1 - 2*b)**2*term1*mp.log(1 - 2*b))
            return float(mp.mpf(8)/5*b**5*(1 - 2*b)**2*term1/bracket)

def love_number_k2(beta, yR, small_beta_cutoff=0.05):
    if beta < small_beta_cutoff:
        return _k2_formula(beta, yR, dps=50)
    return _k2_formula(beta, yR)


def _finalize_results(results, r_cut_km):
    """Shared post-processing for both the serial and parallel sweep paths:
    builds Lambda, finds Mmax, and selects the physically stable branch."""
    results = np.array(results)
    if results.size == 0:
        raise RuntimeError("No central pressures produced a valid stellar solution — check Rbar_max or the EOS table.")
    Pcs, Rs, Ms, betas, yRs, k2s, lams = results.T

    #Dimensionless tidal deformability, Lambda = lambda / M_geom^5.
    M_geom_km = Ms * (R0 / 2.0)
    Lambdas = lams / M_geom_km**5

    idx_max = np.argmax(Ms)

    # Index of the model closest to 1.4 Msun


    def interpolate_at_mass(Ms, Rs, Lambdas, k2s, betas, lams, yRs, stable, M_target=1.4):
        s = stable
        M_stab = Ms[s]
        order = np.argsort(M_stab)  # ensure strictly increasing for Pchip
        M_sorted = M_stab[order]
    
        R_of_M      = PchipInterpolator(M_sorted, Rs[s][order])
        Lambda_of_M = PchipInterpolator(M_sorted, Lambdas[s][order])
        k2_of_M     = PchipInterpolator(M_sorted, k2s[s][order])
        beta_of_M   = PchipInterpolator(M_sorted, betas[s][order])
    
        if not (M_sorted[0] <= M_target <= M_sorted[-1]):
            raise ValueError(f"M={M_target} outside stable branch range "
                              f"[{M_sorted[0]:.3f}, {M_sorted[-1]:.3f}] Msun")
    
        return {
            "R1.4": float(R_of_M(M_target)),
            "Lambda1.4": float(Lambda_of_M(M_target)),
            "k2_1.4": float(k2_of_M(M_target)),
            "beta1.4": float(beta_of_M(M_target)),
        }
    
    valid = Rs < r_cut_km
    if not np.any(valid[:idx_max + 1]):
        stable = slice(idx_max, idx_max + 1)
    else:
        start = idx_max
        while start > 0 and (Ms[start - 1] < Ms[start]) and valid[start - 1]:
            start -= 1
        stable = slice(start, idx_max + 1)
    
    return {
        "Pcs": Pcs,
        "Rs": Rs,
        "Ms": Ms,
        "betas": betas,
        "yRs": yRs,
        "k2s": k2s,
        "lambdas": lams,
        "Lambdas": Lambdas,
    
        "Mmax": Ms[idx_max],
        "Rmax": Rs[idx_max],
        "idx_max": idx_max,
    
        "R1.4": Rs[i14],
        "Lambda1.4": Lambdas[i14],
        "k2_1.4": k2s[i14],
        "beta1.4": betas[i14],
        "lambda1.4": lams[i14],
        "yR1.4": yRs[i14],
    
        "stable": stable,
    }


# ---------------------------------------------------------------------------
# Parallel sweep. The Pbar_c sweep inside EOStoObservables is embarrassingly
# parallel -- each central pressure integrates a fully independent star.
# ProcessPoolExecutor needs the worker function to be a plain top-level
# function (closures like the ones inside EOStoObservables can't reliably be
# pickled to send to worker processes), so this is a self-contained,
# module-level re-implementation of the same TOV+y physics, with the EOS
# interpolants built ONCE per worker process (via the initializer) rather
# than once per central pressure.
# ---------------------------------------------------------------------------
_worker = {}  # populated once per worker process by _init_worker

def _init_worker(EOS):
    global _worker
    tabEOS = np.loadtxt(EOS, skiprows=4)
    energydbar = tabEOS[:, 0] / P0
    pressurebar = tabEOS[:, 1] / P0
    logPbar, logebar = np.log(pressurebar), np.log(energydbar)
    logE_of_logP = PchipInterpolator(logPbar, logebar, extrapolate=False)
    _worker["logE_of_logP"] = logE_of_logP
    _worker["dlogE_dlogP"] = logE_of_logP.derivative(1)
    _worker["Pbar_min"] = pressurebar.min()
    _worker["Pbar_max"] = pressurebar.max()

def _pressureToEnergyd_w(Pbar_eval):
    return np.exp(_worker["logE_of_logP"](np.log(Pbar_eval)))

def _cs2_from_Pbar_w(Pbar_eval):
    lp = np.log(Pbar_eval)
    slope = _worker["dlogE_dlogP"](lp)
    if slope <= 0:
        return 1e10
    ebar = np.exp(_worker["logE_of_logP"](lp))
    return float((Pbar_eval / ebar) / slope)

def _TOV_w(Rbar, y):
    Mbar, Pbar, yL = y
    Pbar_eval = max(Pbar, _worker["Pbar_min"])
    ebar = _pressureToEnergyd_w(Pbar_eval)
    cs2 = _cs2_from_Pbar_w(Pbar_eval)

    dMbar_dRbar = ebar * Rbar**2
    dPbar_dRbar = -1 * (Mbar * ebar / (2 * Rbar**2)) * (1 + Pbar / ebar) * (1 + Rbar**3 * Pbar / Mbar) * (1 - Mbar / Rbar)**-1

    ELAM = 1.0 / (1.0 - Mbar / Rbar)
    y_elam_term = 1.0 + 0.5 * Rbar**2 * (Pbar - ebar)
    Q_bracket = 5*ebar + 9*Pbar + (ebar + Pbar) / cs2
    r2Q = 0.5 * Rbar**2 * ELAM * Q_bracket - 6*ELAM \
          - (ELAM**2 / Rbar**2) * (Mbar + Rbar**3 * Pbar)**2
    dyL_dRbar = -(1.0 / Rbar) * (yL**2 + yL * ELAM * y_elam_term + r2Q)

    return [dMbar_dRbar, dPbar_dRbar, dyL_dRbar]

def _surface_w(Rbar, y):
    return y[1]
_surface_w.terminal = True
_surface_w.direction = -1

def _collapsed_star_task(Pbar_c):
    """Top-level worker: integrate one star. Runs in a worker process;
    reads EOS interpolants from _worker, which _init_worker populated once
    when that process started."""
    Pbar_min, Pbar_max = _worker["Pbar_min"], _worker["Pbar_max"]
    if not (Pbar_min <= Pbar_c <= Pbar_max):
        return None
    ebar_c = _pressureToEnergyd_w(Pbar_c)
    Rbar0 = 1e-6
    Mbar0 = ebar_c * Rbar0**3 / 3
    y0 = [Mbar0, Pbar_c, 2.0]
    sol = solve_ivp(_TOV_w, (Rbar0, 5000), y0, method='RK45', events=_surface_w,
                     rtol=1e-8, atol=1e-10)
    if not sol.success or len(sol.t_events[0]) == 0:
        return None
    Rbarf = sol.t_events[0][0]
    Mbarf, Pbarf, yRbar = sol.y_events[0][0]
    R = Rbarf * R0
    beta = Mbarf / (2.0 * Rbarf)
    try:
        k2 = love_number_k2(beta, yRbar)
    except (ZeroDivisionError, ValueError):
        return None
    if not np.isfinite(k2):
        return None
    lam = (2.0 / 3.0) * k2 * R**5
    return (Pbar_c, R, Mbarf, beta, yRbar, k2, lam)


def EOStoObservables_parallel(EOS, r_cut_km=50.0, n_workers=None, n_points=3000, chunksize=4):
    """
    Same physics and same return format as EOStoObservables, but sweeps
    central pressures across multiple worker processes. On a single large
    EOS table (hundreds to thousands of Pbar_c points, each an independent
    solve_ivp call), this is the parallelism that actually matters --
    parallelizing across EOS *files* instead only helps once you have at
    least as many files as CPU cores.

    n_workers defaults to os.cpu_count(). Must be called from inside
    `if __name__ == "__main__":` (or imported from a saved .py file, not
    pasted into a notebook cell) -- see note below.
    """
    tabEOS = np.loadtxt(EOS, skiprows=4)
    pressurebar = tabEOS[:, 1] / P0
    Pbar_min, Pbar_max = pressurebar.min(), pressurebar.max()
    Pbar_c_values = np.logspace(np.log10(Pbar_min) + 2, np.log10(Pbar_max) - 0.1, n_points)

    n_workers = n_workers or os.cpu_count()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init_worker,
                              initargs=(EOS,)) as ex:
        raw = list(ex.map(_collapsed_star_task, Pbar_c_values, chunksize=chunksize))

    results = [r for r in raw if r is not None]
    return _finalize_results(results, r_cut_km)


# This is where you replace the " " place with whatever tabular EOS file name
def EOStoObservables(EOS, r_cut_km=50.0):
    tabEOS = np.loadtxt(EOS, skiprows = 4)
    energyd = tabEOS[:,0]
    pressure = tabEOS[:,1]
    numd = tabEOS[:,2]
    #Scaling the values in the tabEOS
    energydbar = energyd / P0
    pressurebar = pressure / P0
    assert np.all(np.diff(pressurebar) > 0), "Pressure column is not strictly increasing"
    assert np.all(np.diff(energydbar) > 0), "Energy density column is not strictly increasing"
    #Interpolate log(ebar) vs log(Pbar) with a MONOTONIC (Pchip) interpolant.
    #Real EOS tables are close to power laws and often have kinks (e.g. where
    #a crust table is stitched to a core model) -- an ordinary cubic spline
    #in *linear* P,e space rings/overshoots at those kinks, and since cs2 is
    #a *derivative* of this interpolant, that ringing gets amplified into
    #wild cs2 swings and a noisy/spiky k2(beta) curve. Log-log + Pchip (which
    #is built to never overshoot) fixes both the shape mismatch and the
    #ringing.
    logPbar = np.log(pressurebar)
    logebar = np.log(energydbar)
    logE_of_logP = PchipInterpolator(logPbar, logebar, extrapolate=False)
    dlogE_dlogP = logE_of_logP.derivative(1)

    def pressureToEnergyd(Pbar_eval):
        return np.exp(logE_of_logP(np.log(Pbar_eval)))

    Pbar_min = pressurebar.min()
    Pbar_max = pressurebar.max()

    def cs2_from_Pbar(Pbar_eval):
        # cs2 = dP/de = (e/P) / (dloge/dlogP)
        lp = np.log(Pbar_eval)
        slope = dlogE_dlogP(lp)
        if slope <= 0:
            return 1e10  # near-incompressible fallback
        ebar = np.exp(logE_of_logP(lp))
        return float((Pbar_eval / ebar) / slope)

    #TOV + y(r) equations, all in Mbar/Rbar/Pbar/ebar variables
    def TOV(Rbar, y):
        Mbar, Pbar, yL = y
        Pbar_eval = max(Pbar, Pbar_min)
        ebar = pressureToEnergyd(Pbar_eval)
        cs2 = cs2_from_Pbar(Pbar_eval)

        dMbar_dRbar = ebar * Rbar**2
        dPbar_dRbar = -1 * (Mbar * ebar / (2 * Rbar**2)) * (1 + Pbar / ebar) * (1 + Rbar**3 * Pbar / Mbar) * (1 - Mbar / Rbar)**-1

        # e^lambda = 1/(1-2m/r) becomes 1/(1-Mbar/Rbar) in this scaling,
        # exactly as in the (1-Mbar/Rbar)**-1 factor above.
        ELAM = 1.0 / (1.0 - Mbar / Rbar)

        # Derived by substituting r=R0*Rbar, m=(R0/2)*Mbar, p=Pbar/(8*pi*R0^2),
        # rho=ebar/(8*pi*R0^2) into Eqs. (3)-(6) of 1004.5098 -- R0 cancels
        # completely, as it must since y is dimensionless and its ODE can only
        # depend on the dimensionless state variables.
        y_elam_term = 1.0 + 0.5 * Rbar**2 * (Pbar - ebar)
        Q_bracket = 5*ebar + 9*Pbar + (ebar + Pbar) / cs2
        r2Q = 0.5 * Rbar**2 * ELAM * Q_bracket - 6*ELAM \
              - (ELAM**2 / Rbar**2) * (Mbar + Rbar**3 * Pbar)**2

        dyL_dRbar = -(1.0 / Rbar) * (yL**2 + yL * ELAM * y_elam_term + r2Q)

        return [dMbar_dRbar, dPbar_dRbar, dyL_dRbar]
    
    #Pressure tracking for solver to stop when it reaches 0
    def surface(Rbar, y):
        Mbar, Pbar, yL = y
        return Pbar
    surface.terminal = True
    surface.direction = -1
    
    #Initial Conditions
    def collapsed_star(Pbar_c):
        #Make sure Pbar_c is in the EOS
        assert Pbar_min <= Pbar_c <= Pbar_max, f"Pbar_c={Pbar_c} outside table range [{Pbar_min}, {Pbar_max}]"
        ebar_c = pressureToEnergyd(Pbar_c)
        Rbar0 = 10**-6
        Mbar0 = ebar_c * Rbar0**3 / 3
        y0 = [Mbar0, Pbar_c, 2.0]  # y(0) = 2, Eq. (12) boundary condition -- invariant under rescaling
        #Just to make the code work
        Rbar_max = 5000
        #Solving
        sol = solve_ivp(TOV, (Rbar0, Rbar_max), y0, method='RK45', events=surface, rtol = 10**-8, atol = 10**-10)
        #In case the solution didnt trigger the event, meaning its working wrong and reached dummy Rbar_max limit
        if not sol.success or len(sol.t_events[0]) == 0:
            return None
        #Getting solutions at the exact surface crossing (more accurate than the last integrator step)
        Rbarf = sol.t_events[0][0]
        Mbarf, Pbarf, yRbar = sol.y_events[0][0]
        R = Rbarf * R0
        M = Mbarf

        #Compactness: 2*m_geom/r_geom = Mbar/Rbar in this scaling (see derivation
        #above), so beta = GM/(Rc^2) = m_geom/r_geom = Mbar/(2*Rbar)
        beta = Mbarf / (2.0 * Rbarf)
        try:
            k2 = love_number_k2(beta, yRbar)
        except (ZeroDivisionError, ValueError):
            return None  # pathological yR from the noisy low-density branch
        if not np.isfinite(k2):
            return None
        lam = (2.0 / 3.0) * k2 * R**5  # km^5, G=c=1 convention

        return R, M, beta, yRbar, k2, lam
    
    #Sweep over all Pbar_c except those too close to the surface and the highest densities
    Pbar_c_values = np.logspace(np.log10(Pbar_min) + 2, np.log10(Pbar_max) - 0.1, 3000)
    results = []
    for Pc in Pbar_c_values:
        out = collapsed_star(Pc)
        if out is not None:
            R, M, beta, yRbar, k2, lam = out
            results.append((Pc, R, M, beta, yRbar, k2, lam))
    results = [r for r in results if r is not None]
    return _finalize_results(results, r_cut_km)


def _compare_eos_worker(args):
    """Top-level (picklable) helper: run the *serial* per-file sweep for one
    EOS. Must stay serial in here -- this already runs inside a worker
    process spawned by compare_eos's ProcessPoolExecutor, and daemonic
    worker processes are not allowed to spawn their own child processes, so
    calling EOStoObservables_parallel from in here would raise."""
    fname, r_cut_km = args
    return EOStoObservables(fname, r_cut_km=r_cut_km)


def compare_eos(eos_files, labels=None, r_cut_km=50.0, savepath="eos_comparison.png", n_workers=None):
    """
    Overlay multiple EOS curves on the same M-R / Lambda-M / Lambda-R axes,
    in the style of papers like the one you're checking against (panel a:
    M vs R; panel b: Lambda vs M and Lambda vs R side by side).

    eos_files: list of paths to tabulated EOS files (same format as
               EOStoObservables expects)
    labels: optional list of legend labels (defaults to filenames)
    n_workers: processes to spread the *files* across (defaults to
               os.cpu_count()). Each worker runs one file's full sweep
               serially -- see _compare_eos_worker docstring for why.
    """
    if labels is None:
        labels = [f.split("/")[-1] for f in eos_files]

    #A fixed color cycle so each EOS gets a distinct, consistent color
    #across all three panels (matplotlib's default cycle repeats after 10,
    #which is enough for a typical EOS comparison; pass your own list of
    #hex colors here if you have more than 10 EOS).
    colors = plt.rcParams['axes.prop_cycle'].by_key()['color']

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    #One task per file, computed in its own process. Order is preserved by
    #executor.map, so results[i] always corresponds to eos_files[i].
    tasks = [(fname, r_cut_km) for fname in eos_files]
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        results = list(executor.map(_compare_eos_worker, tasks))

    all_results = {}
    for i, (label, out) in enumerate(zip(labels, results)):
        color = colors[i % len(colors)]
        all_results[label] = out
        s = out["stable"]

        axes[0].plot(out["Rs"][s], out["Ms"][s], color=color, label=label)
        axes[1].plot(out["Ms"][s], out["Lambdas"][s], color=color, label=label)
        axes[2].plot(out["Rs"][s], out["Lambdas"][s], color=color, label=label)

    axes[0].set_xlabel(r"R [km]"); axes[0].set_ylabel(r"M [$M_\odot$]")
    axes[0].set_ylim(bottom=0)

    axes[1].set_xlabel(r"M [$M_\odot$]"); axes[1].set_ylabel(r"$\Lambda$")
    axes[1].set_yscale("log")

    axes[2].set_xlabel(r"R [km]"); axes[2].set_ylabel(r"$\Lambda$")
    axes[2].set_yscale("log")

    axes[0].legend(fontsize=8, loc="best")
    plt.tight_layout()
    plt.savefig(savepath, dpi=150)
    return all_results


if __name__ == "__main__":
    t0 = time.time()
    out = EOStoObservables_parallel("EOSBetaNL3.dat")
    print(f"Swept {len(out['Pcs'])} central pressures in {time.time()-t0:.1f}s")
    print(f"Mmax = {out['Mmax']:.3f} Msun at R = {out['Rmax']:.3f} km")

    i14 = np.argmin(np.abs(out["Ms"] - 1.4))
    print(f"Near M=1.4 Msun: R={out['Rs'][i14]:.3f} km  beta={out['betas'][i14]:.4f}  "
          f"yR={out['yRs'][i14]:.4f}  k2={out['k2s'][i14]:.4f}  lambda={out['lambdas'][i14]:.4e} km^5")

    s = out["stable"]
    fig, axes = plt.subplots(1, 4, figsize=(19, 4))
    axes[0].plot(out["Rs"][s], out["Ms"][s])
    axes[0].set_xlabel("R [km]"); axes[0].set_ylabel(r"M [$M_\odot$]"); axes[0].set_title("Mass-Radius (stable branch)")

    axes[1].plot(out["betas"][s], out["k2s"][s])
    axes[1].set_xlabel(r"$\beta = M/R$"); axes[1].set_ylabel(r"$k_2$"); axes[1].set_title("Love number")

    axes[2].plot(out["Ms"][s], out["lambdas"][s])
    axes[2].set_yscale("log")
    axes[2].set_xlabel(r"M [$M_\odot$]"); axes[2].set_ylabel(r"$\lambda$ [km$^5$]"); axes[2].set_title("Tidal deformability")

    axes[3].plot(out["Ms"][s], out["Lambdas"][s])
    axes[3].set_yscale("log")
    axes[3].set_xlabel(r"M [$M_\odot$]"); axes[3].set_ylabel(r"$\Lambda$"); axes[3].set_title("Dimensionless tidal deformability")

    plt.tight_layout()
    plt.savefig("mrk2_test.png", dpi=120)
    print("Saved mrk2_test.png")
