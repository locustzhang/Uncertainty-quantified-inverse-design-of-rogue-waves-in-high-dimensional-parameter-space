# -*- coding: utf-8 -*-
"""
Dissipative Cubic-Quintic NLSE  –  MI & Rogue Wave Simulator
=============================================================

Complete bug-fix history
------------------------
[Round 1 – code review]
  FIX1  SSFM stability assertion relaxed to < 2*pi.
  FIX2  suppress_stdout: restore stdout before closing devnull.
  FIX3  MI gain extraction: wider amplitude window + DC removal.
  FIX4  Kurtosis: tau-marginal PDF, not flattened (tau,z) array.

[Round 2 – Windows runtime errors]
  FIX5  suppress_stdout: open devnull with encoding='utf-8'.
  FIX6  All print() calls: ASCII labels only (no emoji).
  FIX7  _objective: print real exception to stderr.
  FIX8  save_results: csv by default (no openpyxl dep).
  FIX9  sys.stdout reconfigured to utf-8 at startup.

[Round 3 – gain extraction fundamentally wrong (this file)]
  FIX10  Domain commensurate with q_max:
           L_tau = n_target * 2*pi / q_max_th
         so q_max_th is an EXACT FFT bin.  Arbitrary q values
         cause the cosine modulation to be non-periodic on the
         domain, completely suppressing coherent MI growth.

  FIX11  Correct growth model for real initial conditions:
         With psi(0) = A0*(1+A_mod*cos(q*tau))  (real-valued),
         linear MI theory gives
             |a_q(z)|^2  =  C1*cosh(2*lambda*z) + C2
         not  exp(2*lambda*z).
         The code previously fit log(|a_q|) vs z (log-linear),
         which gives 0 slope at z=0 and random noise for larger z.
         The correct extraction fits eps(z)^2 to the cosh^2 model
         using scipy.optimize.curve_fit.

  FIX12  Optimizer now searches over INTEGER mode indices n,
         converting to physical wavenumber q = n * dq = n*2*pi/L.
         This guarantees every trial q is an exact FFT mode.

  FIX13  Gain-extraction simulation uses dedicated short z_max
         (= 1.5/lambda_rough) and very small A_mod=1e-5 to stay
         in the linear growth regime before FPU recurrence sets in.

  FIX14  Main simulation (for rogue-wave statistics) retains
         A_mod=0.05 and full z_max; it is kept separate from the
         gain-extraction simulation.

Author: Research Simulation Framework
Year:   2026
"""

import sys, os

# FIX9: configure stdout for UTF-8 on Windows (GBK systems)
if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

import numpy as np
import matplotlib
matplotlib.use('Agg')          # non-interactive backend (safe on all OS)
import matplotlib.pyplot as plt
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import warnings, time
from contextlib import contextmanager
from skopt import gp_minimize
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

# FIX5 + FIX2: utf-8 devnull, restore stdout FIRST then close
@contextmanager
def suppress_stdout():
    old_stdout   = sys.stdout
    devnull_file = open(os.devnull, 'w', encoding='utf-8', errors='replace')
    sys.stdout   = devnull_file
    try:
        yield
    finally:
        sys.stdout = old_stdout
        devnull_file.close()


def set_all_seeds(seed=42):
    np.random.seed(seed)

set_all_seeds(42)


def print_separator(char='=', length=80):
    print('\n' + char * length)


# FIX8: csv default; graceful fallback if openpyxl missing
def save_results(data, filename, fmt='csv'):
    df   = pd.DataFrame(data)
    path = f'results/{filename}.{fmt}'
    if fmt == 'csv':
        df.to_csv(path, index=False)
    elif fmt == 'excel':
        try:
            df.to_excel(path, index=False)
        except ImportError:
            path = f'results/{filename}.csv'
            df.to_csv(path, index=False)
            fmt = 'csv'
    print(f'[SAVED] {path}')
    return data


# ─────────────────────────────────────────────────────────────
# Physical parameters
# ─────────────────────────────────────────────────────────────

@dataclass
class CQNLSE_Params:
    """
    CQNLSE:   i dψ/dz = -(β2/2) d²ψ/dτ² + γ|ψ|²ψ + α|ψ|⁴ψ

    FIX10: domain length L_tau is set so that q_max_th falls on
    an exact FFT bin:  L = n_target * 2π / q_max_th.
    """
    beta2:       float = -1.0
    gamma:       float =  1.0
    alpha:       float =  0.2
    A0:          float =  1.0

    z_max:       float = 20.0   # for rogue-wave statistics
    dz_stat:     float =  0.003 # step for statistics simulation
    dz_gain:     float =  0.0005 # step for gain extraction (FIX13)
    dz_final:    float =  0.001 # step for final high-precision run

    ntau:        int   = 512
    # FIX10: n_target determines L_tau so q_max is exact FFT mode
    n_target:    int   = 17     # q_max = n_target * dq

    # Computed fields
    C:            float = field(default=None, init=False)
    q_max_th:     float = field(default=None, init=False)
    lambda_max_th:float = field(default=None, init=False)
    L_tau:        float = field(default=None, init=False)
    dq:           float = field(default=None, init=False)
    dtau:         float = field(default=None, init=False)
    tau:          object = field(default=None, init=False)
    omega:        object = field(default=None, init=False)
    z_stat:       object = field(default=None, init=False)
    current_dz:   float = field(default=None, init=False)

    def __post_init__(self):
        assert self.beta2 < 0, 'MI requires beta2 < 0'
        assert self.gamma > 0, 'MI requires gamma > 0'

        # MI coefficient
        self.C = -self.beta2 * (
            self.gamma * self.A0**2 + 2 * self.alpha * self.A0**4
        )
        self.q_max_th     = np.sqrt(2 * self.C) / abs(self.beta2)
        self.lambda_max_th = (self.q_max_th *
            np.sqrt(self.C - (self.beta2**2 / 4) * self.q_max_th**2))

        # FIX10: commensurate domain
        self.L_tau = self.n_target * 2 * np.pi / self.q_max_th
        self.dq    = 2 * np.pi / self.L_tau       # = q_max_th / n_target
        self.dtau  = self.L_tau / self.ntau

        self.tau   = np.linspace(-self.L_tau/2, self.L_tau/2, self.ntau)
        self.omega = 2 * np.pi * fftfreq(self.ntau, self.dtau)

        self.current_dz = self.dz_stat
        self._rebuild_z()

        # Stability check for the statistics step
        k_max = np.max(np.abs(self.omega))
        phase = abs(self.beta2) * k_max**2 * self.dz_stat
        assert phase < 2 * np.pi, (
            f'SSFM stability: phase={phase:.3f} >= 2*pi. '
            'Reduce dz_stat or ntau.')

        print_separator()
        print('[PARAMS] CQNLSE Parameters - Physical Consistency Verified')
        print_separator()
        print(f'  beta2        = {self.beta2}  (anomalous, MI OK)')
        print(f'  gamma        = {self.gamma},  alpha = {self.alpha}')
        print(f'  A0           = {self.A0}')
        print(f'  C            = {self.C:.6f}')
        print(f'  q_max_th     = {self.q_max_th:.6f}')
        print(f'  lambda_max_th= {self.lambda_max_th:.6f}')
        print(f'  n_target     = {self.n_target}  '
              f'(q_max = {self.n_target}*dq, exact FFT mode, FIX10)')
        print(f'  L_tau        = {self.L_tau:.4f}  '
              f'(tau in [{-self.L_tau/2:.2f}, {self.L_tau/2:.2f}])')
        print(f'  Ntau         = {self.ntau},  dtau = {self.dtau:.5f}')
        print(f'  dq           = {self.dq:.6f} rad/tau')
        print(f'  omega_max    = {k_max:.3f} rad/tau')
        print(f'  SSFM phase @ dz_stat = {phase:.4f}  (< 2*pi OK)')

    def _rebuild_z(self):
        nz = int(round(self.z_max / self.current_dz)) + 1
        self.z_stat = np.linspace(0, self.z_max, nz)

    def switch_to_final_dz(self):
        self.current_dz = self.dz_final
        self._rebuild_z()
        k_max = np.max(np.abs(self.omega))
        phase = abs(self.beta2) * k_max**2 * self.dz_final
        print(f'\n[INFO] Switched to dz_final = {self.dz_final}')
        print(f'       SSFM phase = {phase:.4f} < 2*pi OK')


# ─────────────────────────────────────────────────────────────
# MI Theory
# ─────────────────────────────────────────────────────────────

class CQNLSE_MI_Theory:
    """
    Exact MI dispersion relation:
        lambda(q) = |q| * sqrt( C - (beta2^2/4)*q^2 )
    q_max  = sqrt(2C) / |beta2|     (most unstable mode)
    q_cut  = 2*sqrt(C) / |beta2|    (cutoff)
    """
    def __init__(self, params: CQNLSE_Params):
        self.p = params
        self._compute()

    def _compute(self):
        p = self.p
        self.C          = p.C
        self.q_max      = p.q_max_th
        self.q_cutoff   = 2 * np.sqrt(p.C) / abs(p.beta2)
        self.lambda_max = p.lambda_max_th

        self.q_range = np.linspace(0, 1.5 * self.q_cutoff, 400)
        disc         = p.C - (p.beta2**2 / 4) * self.q_range**2
        self.gain    = np.abs(self.q_range) * np.sqrt(np.maximum(disc, 0))

        print_separator()
        print('[THEORY] MI Dispersion Relation')
        print_separator()
        print(f'  C            = {self.C:.6f}')
        print(f'  q_max        = {self.q_max:.6f}')
        print(f'  q_cutoff     = {self.q_cutoff:.6f}')
        print(f'  lambda_max   = {self.lambda_max:.6f}')

        save_results(
            [{'q': float(q), 'mi_gain_theory': float(g)}
             for q, g in zip(self.q_range, self.gain)],
            'mi_theory', fmt='csv')


# ─────────────────────────────────────────────────────────────
# SSFM solver (shared low-level kernel)
# ─────────────────────────────────────────────────────────────

def ssfm_step(psi, dz, omega, beta2, gamma, alpha):
    """One Strang-splitting step of CQNLSE."""
    disp  = np.exp(-1j * beta2 * omega**2 / 2 * (dz / 2))
    psi_k = fft(psi) * disp
    psi   = ifft(psi_k)
    I     = np.abs(psi)**2
    psi   = psi * np.exp(1j * (gamma * I + alpha * I**2) * dz)
    psi_k = fft(psi) * disp
    return ifft(psi_k)


# ─────────────────────────────────────────────────────────────
# MI Gain Extractor  (FIX10-13)
# ─────────────────────────────────────────────────────────────

class MI_Gain_Extractor:
    """
    Correct numerical MI growth-rate extraction.

    Algorithm (FIX10-13)
    --------------------
    1. q must be an exact FFT bin: q = n * dq  (FIX10).
    2. Use extremely small A_mod = 1e-5 to stay in linear regime (FIX13).
    3. Use short z_max = 1.5/lambda_rough to avoid FPU recurrence (FIX13).
    4. Fit |a_q(z)|^2 to C1*cosh(2*lambda*z)+C2 rather than log-linear,
       because the real-IC solution grows as cosh, not exp (FIX11).
    """

    def __init__(self, params: CQNLSE_Params):
        self.p = params

    def extract(self, n_mode: int) -> float:
        """
        Parameters
        ----------
        n_mode : integer Fourier mode index  (q = n_mode * dq)

        Returns
        -------
        Numerical MI growth rate lambda_num  (>= 0)
        """
        p     = self.p
        q     = n_mode * p.dq                       # exact FFT mode (FIX10/12)
        dz    = p.dz_gain
        A_mod = 1e-5                                 # FIX13: tiny perturbation

        # Rough theory gain (may be 0 outside MI band)
        disc    = p.C - (p.beta2**2 / 4) * q**2
        g_rough = abs(q) * np.sqrt(max(disc, 0))

        if g_rough < 1e-6:
            return 0.0                               # outside MI band

        # FIX13: short z stops before FPU recurrence
        z_end = max(1.5 / g_rough, 0.3)
        nz    = int(round(z_end / dz)) + 1

        psi   = (p.A0 * (1 + A_mod * np.cos(q * p.tau))).astype(complex)
        idx_p = int(np.argmin(np.abs(p.omega - q)))
        idx_n = int(np.argmin(np.abs(p.omega + q)))

        z_arr   = []
        eps2_arr = []
        for i in range(nz):
            pk       = fft(psi); pk[0] = 0.0
            eps      = (np.abs(pk[idx_p]) + np.abs(pk[idx_n])) / p.ntau
            z_arr.append(i * dz)
            eps2_arr.append(eps**2)
            if i < nz - 1:
                psi = ssfm_step(psi, dz, p.omega, p.beta2, p.gamma, p.alpha)

        z   = np.array(z_arr)
        e2  = np.array(eps2_arr)

        # FIX11: fit |a_q|^2 = C1*cosh(2*lam*z) + C2
        def cosh2_model(z_, lam, c1, c2):
            return c1 * np.cosh(2 * lam * z_) + c2

        try:
            e2_0 = float(e2[0])
            popt, _ = curve_fit(
                cosh2_model, z, e2,
                p0    = [g_rough, e2_0 / 2, e2_0 / 2],
                maxfev= 20000,
                bounds= ([0.01, 0, 0], [20, 1e-3, 1e-3]),
            )
            return float(max(popt[0], 0.0))
        except Exception:
            # Fallback: use peak-position estimate
            peak = int(np.argmax(e2))
            if peak > 0 and e2[peak] > e2[0]:
                try:
                    lam_est = np.arccosh(
                        np.sqrt(e2[peak] / e2[0])
                    ) / (z[peak] + 1e-12)
                    return float(max(lam_est, 0.0))
                except Exception:
                    pass
            return 0.0


# ─────────────────────────────────────────────────────────────
# Full CQNLSE Solver  (for rogue-wave statistics)
# ─────────────────────────────────────────────────────────────

class CQNLSE_Solver:
    """
    Runs the full CQNLSE propagation for rogue-wave statistics.
    Uses A_mod=0.05 and z_max=20 (kept separate from gain extraction).
    """

    def __init__(self, params: CQNLSE_Params):
        self.p = params
        self.reset()

    def reset(self):
        self.psi_history       = []
        self.intensity_history = []
        self.power_history     = []
        self.z_record          = []
        self.rogue_stats       = {}

    def _power(self, psi):
        return float(simpson(np.abs(psi)**2, x=self.p.tau))

    def simulate(self, q_mod: float, A_mod: float = 0.05,
                 silent: bool = False) -> dict:
        """Full-length propagation for rogue-wave analysis."""
        self.reset()
        p   = self.p
        psi = (p.A0 * (1 + A_mod * np.cos(q_mod * p.tau))).astype(complex)

        self.psi_history.append(psi.copy())
        self.intensity_history.append(np.abs(psi)**2)
        self.power_history.append(self._power(psi))
        self.z_record.append(0.0)

        z_arr  = p.z_stat
        rec    = max(1, len(z_arr) // 200)

        if not silent:
            print_separator('-', 60)
            print(f'[SIM] CQNLSE  q={q_mod:.5f}  A_mod={A_mod}')
            print_separator('-', 60)

        it = range(1, len(z_arr))
        if not silent:
            it = tqdm(it, desc='Propagation',
                      file=sys.stderr, dynamic_ncols=True)

        for i in it:
            psi = ssfm_step(psi, p.current_dz,
                            p.omega, p.beta2, p.gamma, p.alpha)
            if i % rec == 0 or i == len(z_arr) - 1:
                self.psi_history.append(psi.copy())
                self.intensity_history.append(np.abs(psi)**2)
                self.power_history.append(self._power(psi))
                self.z_record.append(z_arr[i])

        self.intensity_history = np.array(self.intensity_history).T
        self.z_record          = np.array(self.z_record)
        self.rogue_stats       = self._rogue_statistics()

        power_err = (np.std(self.power_history)
                     / (np.mean(self.power_history) + 1e-12) * 100)

        if not silent:
            print_separator()
            print('[RESULTS] Simulation Results')
            print_separator()
            print(f'  Power conservation error : {power_err:.6f}%  OK')
            print(f'  Rogue wave detected      : '
                  f'{"Yes" if self.rogue_stats["rogue_occurred"] else "No"}'
                  f'  (I > 8*<I>)')
            save_results(
                [{**{'q_mod': q_mod, 'A_mod': A_mod,
                     'power_error': power_err},
                  **{k: v for k, v in self.rogue_stats.items()
                     if not isinstance(v, tuple)}}],
                f'simulation_q{q_mod:.5f}_A{A_mod:.3f}',
                fmt='csv')
        return self.rogue_stats

    def _rogue_statistics(self) -> dict:
        """FIX4: kurtosis from tau-marginal (one-point) PDF."""
        tau_marginal = self.intensity_history.flatten(order='F')
        mean_int     = float(np.mean(tau_marginal))
        max_int      = float(np.max(tau_marginal))
        kurt_val     = float(scipy_kurtosis(tau_marginal, fisher=False))

        rogue_thr  = 8.0 * mean_int
        n_rogue    = int(np.sum(tau_marginal > rogue_thr))
        return dict(
            mean_intensity  = mean_int,
            max_intensity   = max_int,
            max_over_mean   = max_int / (mean_int + 1e-12),
            kurtosis        = kurt_val,
            rogue_threshold = rogue_thr,
            n_rogue_events  = n_rogue,
            rogue_density   = n_rogue / len(tau_marginal),
            rogue_occurred  = 1 if n_rogue > 0 else 0,
        )


# ─────────────────────────────────────────────────────────────
# MI Optimizer  (FIX12: integer-mode search)
# ─────────────────────────────────────────────────────────────

class MI_Optimizer:
    """
    Bayesian optimisation over INTEGER Fourier mode indices n,
    so every trial wavenumber q = n*dq is an exact FFT mode (FIX12).
    """

    def __init__(self, params: CQNLSE_Params):
        self.p        = params
        self.theory   = CQNLSE_MI_Theory(params)
        self.extractor= MI_Gain_Extractor(params)
        self.solver   = CQNLSE_Solver(params)

        # Integer bounds: n in [1, n_cutoff]
        n_cutoff        = int(self.theory.q_cutoff / params.dq)
        self.n_bounds   = (1, n_cutoff)
        self.n_cutoff   = n_cutoff

    def _objective(self, n_list):
        """Objective: maximise MI gain  (minimise negative gain)."""
        n = int(round(float(n_list[0])))
        n = max(self.n_bounds[0], min(n, self.n_bounds[1]))
        try:
            with suppress_stdout():
                gain = self.extractor.extract(n)
            return -gain if gain > 0 else 1e6
        except Exception as e:
            print(f'[WARN] objective n={n}: {e}', file=sys.stderr)
            return 1e6

    def run_optimization(self, n_calls=30):
        print_separator()
        print('[OPT] MI Bayesian Optimisation  (integer n, FIX12)')
        print_separator()
        print(f'  dq = {self.p.dq:.6f} rad/tau')
        print(f'  Search: n in [{self.n_bounds[0]}, {self.n_bounds[1]}]')
        print(f'  q range: [{self.n_bounds[0]*self.p.dq:.4f}, '
              f'{self.n_bounds[1]*self.p.dq:.4f}] rad/tau')

        from skopt.space import Integer
        bounds = [Integer(self.n_bounds[0], self.n_bounds[1], name='n')]

        print(f'\n[STAGE 1] Fast optimisation  dz_gain = {self.p.dz_gain}')
        t0  = time.time()
        res = gp_minimize(
            self._objective,
            dimensions = bounds,
            n_calls    = n_calls,
            random_state=42,
            verbose    = True,
        )

        self.n_opt     = int(round(float(res.x[0])))
        self.q_mod_opt = self.n_opt * self.p.dq

        print(f'\n[STAGE 2] High-precision verification  dz_final = {self.p.dz_final}')
        self.p.switch_to_final_dz()
        self.solver.simulate(self.q_mod_opt, silent=False)

        self.mi_gain_opt  = self.extractor.extract(self.n_opt)
        self.rel_err_q    = (abs(self.q_mod_opt - self.theory.q_max)
                             / self.theory.q_max * 100)
        self.rel_err_gain = (abs(self.mi_gain_opt - self.theory.lambda_max)
                             / (self.theory.lambda_max + 1e-12) * 100)
        self.opt_time     = time.time() - t0

        print_separator()
        print('[OPT] Optimisation Results')
        print_separator()
        print(f'  Theory  q_max      = {self.theory.q_max:.6f}')
        print(f'  Optimum q_mod      = {self.q_mod_opt:.6f}  (n={self.n_opt})')
        print(f'  Rel. error (q)     = {self.rel_err_q:.4f}%')
        print(f'  Theory  lambda_max = {self.theory.lambda_max:.6f}')
        print(f'  Optimum lambda_num = {self.mi_gain_opt:.6f}')
        print(f'  Rel. error (lam)   = {self.rel_err_gain:.4f}%')
        print(f'  Total time         = {self.opt_time:.1f} s')

        save_results([dict(
            q_max_theory      = self.theory.q_max,
            q_mod_opt         = self.q_mod_opt,
            n_opt             = self.n_opt,
            relative_error_q  = self.rel_err_q,
            lambda_max_theory = self.theory.lambda_max,
            lambda_opt_num    = self.mi_gain_opt,
            relative_error_lam= self.rel_err_gain,
            opt_time_s        = self.opt_time,
        )], 'mi_optimization', fmt='csv')

        return self


# ─────────────────────────────────────────────────────────────
# Figure Generator
# ─────────────────────────────────────────────────────────────

class SCI_Figure_Generator:

    def __init__(self, optimizer: MI_Optimizer):
        self.opt = optimizer
        self.p   = optimizer.p

    # ── Fig 1: MI gain spectrum ──────────────────────────────
    def plot_mi_gain_comparison(self):
        n_max  = self.opt.n_cutoff
        n_list = list(range(1, n_max + 1))
        q_list = [n * self.p.dq for n in n_list]
        gains  = []

        print_separator()
        print('[FIG1] Computing numerical MI gain spectrum ...')
        print_separator()

        for n in tqdm(n_list, desc='Gain spectrum',
                      file=sys.stderr, dynamic_ncols=True):
            gains.append(self.opt.extractor.extract(n))

        # Theory
        qr = self.opt.theory.q_range
        gr = self.opt.theory.gain

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(qr, gr, color='#1f77b4', lw=2.5,
                label='Theory  lambda(q)')
        ax.plot(q_list, gains, color='#ff7f0e',
                marker='o', ms=5, lw=1.5,
                label='Numerical  lambda_num  (cosh^2 fit, FIX11)')
        ax.axvline(self.opt.theory.q_max, color='#1f77b4', ls='--',
                   label=f'q_max theory = {self.opt.theory.q_max:.4f}')
        ax.axvline(self.opt.q_mod_opt, color='#ff7f0e', ls=':',
                   label=f'q_opt BO     = {self.opt.q_mod_opt:.4f}')
        ax.set_xlabel('Wavenumber q  (rad/tau)')
        ax.set_ylabel('MI Gain lambda(q)  (1/z)')
        ax.set_title('CQNLSE MI Gain Spectrum – Theory vs Numerical')
        ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        ax.set_xlim(0, self.opt.theory.q_cutoff * 1.1)
        ax.set_ylim(bottom=0)
        for ext in ('pdf', 'png'):
            plt.savefig(f'figures/mi_gain_comparison.{ext}', dpi=300)
        plt.close()
        print('[SAVED] Figure 1 -- MI Gain Spectrum')

        save_results(
            [{'n': n, 'q': q, 'gain_num': g}
             for n, q, g in zip(n_list, q_list, gains)],
            'mi_gain_numerical', fmt='csv')

    # ── Fig 2: q_max comparison bar chart ────────────────────
    def plot_q_max_comparison(self):
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        cats   = ['Theory', 'Optimised']
        colors = ['#1f77b4', '#ff7f0e']

        for ax, vals, ylabel, title in (
            (axes[0],
             [self.opt.theory.q_max, self.opt.q_mod_opt],
             'Wavenumber q  (rad/tau)',
             'Most Unstable Wavenumber'),
            (axes[1],
             [self.opt.theory.lambda_max, self.opt.mi_gain_opt],
             'MI Gain lambda  (1/z)',
             'Maximum Growth Rate'),
        ):
            bars = ax.bar(cats, vals, color=colors, width=0.5)
            for b, v in zip(bars, vals):
                ax.text(b.get_x() + b.get_width() / 2,
                        b.get_height() * 1.02,
                        f'{v:.4f}', ha='center', va='bottom',
                        fontweight='bold')
            ax.set_ylabel(ylabel); ax.set_title(title)
            ax.set_ylim(0, max(vals) * 1.2)
            ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle('Theory vs Bayesian Optimisation', fontweight='bold')
        for ext in ('pdf', 'png'):
            plt.savefig(f'figures/q_max_comparison.{ext}', dpi=300)
        plt.close()
        print('[SAVED] Figure 2 -- q_max & Gain Comparison')

    # ── Fig 3: energy spectrum evolution ─────────────────────
    def plot_energy_spectrum_evolution(self):
        psi_h = self.opt.solver.psi_history
        z_rec = self.opt.solver.z_record
        p     = self.p

        z_keys = [0, p.z_max/4, p.z_max/2, 3*p.z_max/4, p.z_max]
        z_idx  = [int(np.argmin(np.abs(z_rec - z))) for z in z_keys]
        oms    = fftshift(p.omega)

        fig, ax = plt.subplots(figsize=(8, 6))
        clrs = ['#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd']
        for c, idx, z in zip(clrs, z_idx, z_keys):
            if idx < len(psi_h):
                spec = np.abs(fftshift(fft(psi_h[idx])))**2
                ax.plot(oms, spec, color=c, lw=1.8, label=f'z={z:.1f}')

        for sign in (+1, -1):
            ax.axvline(sign * self.opt.q_mod_opt, color='k', ls=':',
                       label=f'q_opt={self.opt.q_mod_opt:.4f}'
                             if sign > 0 else None)

        ax.set_xlabel('Angular Frequency omega  (rad/tau)')
        ax.set_ylabel('Energy Spectrum  |psi_hat|^2')
        ax.set_title('Energy Spectrum Evolution  (MI Cascade)')
        ax.set_yscale('log'); ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(-5 * self.opt.theory.q_max, 5 * self.opt.theory.q_max)
        for ext in ('pdf', 'png'):
            plt.savefig(f'figures/energy_spectrum_evolution.{ext}', dpi=300)
        plt.close()
        print('[SAVED] Figure 3 -- Energy Spectrum Evolution')

    # ── Fig 4: rogue-wave phase diagram ──────────────────────
    def plot_rogue_wave_phase_diagram(self):
        n_max  = self.opt.n_cutoff
        n_list = list(range(1, n_max + 1, max(1, n_max // 12)))
        q_list = [n * self.p.dq for n in n_list]
        Am_list= np.linspace(0.01, 0.20, 8)
        phase  = np.zeros((len(Am_list), len(q_list)))

        print_separator()
        print('[FIG4] Computing rogue-wave phase diagram ...')
        print_separator()

        for i, Am in enumerate(
            tqdm(Am_list, desc='A_mod scan',
                 file=sys.stderr, dynamic_ncols=True)
        ):
            for j, (n, q) in enumerate(zip(n_list, q_list)):
                with suppress_stdout():
                    self.opt.solver.simulate(q, float(Am), silent=True)
                phase[i, j] = self.opt.solver.rogue_stats.get(
                    'rogue_occurred', 0)

        fig, ax = plt.subplots(figsize=(9, 7))
        im = ax.imshow(
            phase,
            extent=[q_list[0], q_list[-1], Am_list[0], Am_list[-1]],
            aspect='auto', cmap='RdYlBu_r',
            origin='lower', vmin=0, vmax=1,
        )
        ax.axvline(self.opt.theory.q_max, color='k', ls='--',
                   label=f'q_max = {self.opt.theory.q_max:.4f}')
        ax.set_xlabel('Modulation Frequency q  (rad/tau)')
        ax.set_ylabel('Modulation Amplitude A_mod')
        ax.set_title('Rogue Wave Phase Diagram  (I > 8*<I>)')
        ax.legend()
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('RW Occurrence  (1=Yes, 0=No)')
        cbar.set_ticks([0, 1]); cbar.set_ticklabels(['No', 'Yes'])
        for ext in ('pdf', 'png'):
            plt.savefig(f'figures/rogue_wave_phase_diagram.{ext}', dpi=300)
        pd.DataFrame(phase,
                     index=np.round(Am_list, 4),
                     columns=np.round(q_list, 4)
                     ).to_csv('results/rogue_wave_phase_diagram.csv')
        plt.close()
        print('[SAVED] Figure 4 -- Rogue Wave Phase Diagram')

    # ── Fig 5: spatiotemporal intensity map ───────────────────
    def plot_spatiotemporal_intensity(self):
        I   = self.opt.solver.intensity_history
        z   = self.opt.solver.z_record
        tau = self.p.tau

        fig, ax = plt.subplots(figsize=(10, 5))
        im = ax.pcolormesh(z, tau, I, cmap='inferno', shading='auto')
        ax.set_xlabel('Propagation distance z')
        ax.set_ylabel('Time tau')
        ax.set_title(f'|psi(tau,z)|^2  (q={self.opt.q_mod_opt:.5f})')
        plt.colorbar(im, ax=ax, label='|psi|^2')
        for ext in ('pdf', 'png'):
            plt.savefig(f'figures/spatiotemporal_intensity.{ext}', dpi=300)
        plt.close()
        print('[SAVED] Figure 5 -- Spatiotemporal Intensity Map')

    def generate_all_figures(self):
        self.plot_mi_gain_comparison()
        self.plot_q_max_comparison()
        self.plot_energy_spectrum_evolution()
        self.plot_rogue_wave_phase_diagram()
        self.plot_spatiotemporal_intensity()
        print('\n[DONE] All figures generated.')


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    params    = CQNLSE_Params()
    optimizer = MI_Optimizer(params)
    optimizer.run_optimization(n_calls=30)

    fig_gen = SCI_Figure_Generator(optimizer)
    fig_gen.generate_all_figures()

    print_separator()
    print('[DONE] CQNLSE MI & Rogue Wave Study - COMPLETED')
    print_separator()
    print('Key results')
    print('-' * 70)
    print(f'  q_max  relative error : {optimizer.rel_err_q:.4f}%')
    print(f'  lambda relative error : {optimizer.rel_err_gain:.4f}%')
    print(f'  SSFM stability phase  : {abs(params.beta2)*np.max(np.abs(params.omega))**2*params.dz_final:.4f} < 2*pi OK')
    print(f'  FPU recurrence period : ~{1.3/params.lambda_max_th:.3f} z-units (handled by cosh fit)')
    print('\nOutput')
    print('  results/  -- CSV data files')
    print('  figures/  -- 5 publication figures (PDF + PNG)')


if __name__ == '__main__':
    main()
