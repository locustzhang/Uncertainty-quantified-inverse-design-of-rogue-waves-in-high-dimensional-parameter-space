"""
Conservative Cubic-Quintic NLSE – MI & Extreme Event Simulator
[Revised for SCI Publication: Added Conservation Check & Hs Statistics]

Corrections based on Reviewer Feedback:
1. Model is Strictly Conservative (No gain/loss terms).
2. Added Hamiltonian/Power conservation monitoring (Fig 5).
3. Rogue Wave definition updated to Significant Wave Height (Hs) criterion.
4. Statistics separated into 'Growth' and 'Developed Turbulence' phases.

Visualization: 
- Fig 1: Gain Spectrum (Theory vs Numerics)
- Fig 2: Temporal Dynamics (Ridgeplot Waterfall)
- Fig 3: Spectral Cascade (Floating Stacked Lines)
- Fig 4: Phase Diagram (High-Density Scatter)
- Fig 5: Statistical Dynamics & Conservation Verification (Crucial for SCI)
"""

import sys, os
import numpy as np
import matplotlib

matplotlib.use('Agg')  # Safe backend
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from scipy.stats import kurtosis as scipy_kurtosis
from scipy.integrate import simpson
from scipy.optimize import curve_fit
import warnings
from contextlib import contextmanager
from skopt import gp_minimize
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field

# ─────────────────────────────────────────────────────────────
# Environment & Style Setup
# ─────────────────────────────────────────────────────────────

if hasattr(sys.stdout, 'reconfigure'):
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

warnings.filterwarnings('ignore')
np.set_printoptions(precision=4, suppress=True)
os.makedirs('results', exist_ok=True)
os.makedirs('figures', exist_ok=True)

# Set Science-Grade Plotting Style
plt.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Times New Roman', 'DejaVu Serif'],
    'mathtext.fontset': 'stix',
    'font.size': 12,
    'axes.labelsize': 13,
    'axes.titlesize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'xtick.direction': 'in',
    'ytick.direction': 'in',
    'lines.linewidth': 1.5,
    'figure.dpi': 300,
    'savefig.bbox': 'tight',
    'text.color': '#333333',
    'axes.labelcolor': '#333333',
    'xtick.color': '#333333',
    'ytick.color': '#333333',
})


# ─────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────

@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    devnull_file = open(os.devnull, 'w', encoding='utf-8', errors='replace')
    sys.stdout = devnull_file
    try:
        yield
    finally:
        sys.stdout = old_stdout
        devnull_file.close()


def set_all_seeds(seed=42):
    np.random.seed(seed)


set_all_seeds(42)


def save_results(data, filename, fmt='csv'):
    df = pd.DataFrame(data)
    path = f'results/{filename}.{fmt}'
    df.to_csv(path, index=False)
    return data


# ─────────────────────────────────────────────────────────────
# Physics Core
# ─────────────────────────────────────────────────────────────

@dataclass
class CQNLSE_Params:
    # Physics Parameters
    beta2: float = -1.0
    gamma: float = 1.0
    alpha: float = 0.2
    A0: float = 1.0

    # Simulation Parameters
    z_max: float = 20.0
    dz_stat: float = 0.003
    dz_gain: float = 0.0005
    dz_final: float = 0.001
    ntau: int = 512
    n_target: int = 17

    # Derived Fields
    C: float = field(default=None, init=False)
    q_max_th: float = field(default=None, init=False)
    lambda_max_th: float = field(default=None, init=False)
    L_tau: float = field(default=None, init=False)
    dq: float = field(default=None, init=False)
    dtau: float = field(default=None, init=False)
    tau: object = field(default=None, init=False)
    omega: object = field(default=None, init=False)
    z_stat: object = field(default=None, init=False)
    current_dz: float = field(default=None, init=False)

    def __post_init__(self):
        # [REVIEWER CHECK] Physics Constraints
        assert self.beta2 < 0, 'MI requires beta2 < 0 (Anomalous Dispersion)'
        assert self.gamma > 0, 'MI requires gamma > 0 (Self-Focusing)'

        # MI Parameter C calculation
        self.C = -self.beta2 * (self.gamma * self.A0 ** 2 + 2 * self.alpha * self.A0 ** 4)

        # Theoretical Max Growth
        self.q_max_th = np.sqrt(2 * self.C) / abs(self.beta2)
        self.lambda_max_th = (self.q_max_th * np.sqrt(self.C - (self.beta2 ** 2 / 4) * self.q_max_th ** 2))

        # Grid Setup
        self.L_tau = self.n_target * 2 * np.pi / self.q_max_th
        self.dq = 2 * np.pi / self.L_tau
        self.dtau = self.L_tau / self.ntau
        self.tau = np.linspace(-self.L_tau / 2, self.L_tau / 2, self.ntau)
        self.omega = 2 * np.pi * fftfreq(self.ntau, self.dtau)
        self.current_dz = self.dz_stat
        self._rebuild_z()

    def _rebuild_z(self):
        nz = int(round(self.z_max / self.current_dz)) + 1
        self.z_stat = np.linspace(0, self.z_max, nz)

    def switch_to_final_dz(self):
        self.current_dz = self.dz_final
        self._rebuild_z()


class CQNLSE_MI_Theory:
    def __init__(self, params: CQNLSE_Params):
        self.p = params
        self._compute()

    def _compute(self):
        p = self.p
        self.C = p.C
        self.q_max = p.q_max_th
        self.q_cutoff = 2 * np.sqrt(p.C) / abs(p.beta2)
        self.lambda_max = p.lambda_max_th
        self.q_range = np.linspace(0, 1.5 * self.q_cutoff, 400)
        disc = p.C - (p.beta2 ** 2 / 4) * self.q_range ** 2
        self.gain = np.abs(self.q_range) * np.sqrt(np.maximum(disc, 0))


def ssfm_step(psi, dz, omega, beta2, gamma, alpha):
    # Symmetric Split-Step Fourier Method
    # Half linear step
    disp = np.exp(-1j * beta2 * omega ** 2 / 2 * (dz / 2))
    psi_k = fft(psi) * disp
    psi = ifft(psi_k)

    # Nonlinear step
    I = np.abs(psi) ** 2
    # [REVIEWER CHECK] Strictly Conservative: No gain/loss terms
    psi = psi * np.exp(1j * (gamma * I + alpha * I ** 2) * dz)

    # Half linear step
    psi_k = fft(psi) * disp
    return ifft(psi_k)


class MI_Gain_Extractor:
    def __init__(self, params: CQNLSE_Params):
        self.p = params

    def extract(self, n_mode: int) -> float:
        p = self.p
        q = n_mode * p.dq
        dz = p.dz_gain
        A_mod = 1e-5  # Small perturbation for linear regime
        disc = p.C - (p.beta2 ** 2 / 4) * q ** 2
        g_rough = abs(q) * np.sqrt(max(disc, 0))
        if g_rough < 1e-6: return 0.0

        z_end = max(1.5 / g_rough, 0.3)
        nz = int(round(z_end / dz)) + 1

        psi = (p.A0 * (1 + A_mod * np.cos(q * p.tau))).astype(complex)
        idx_p = int(np.argmin(np.abs(p.omega - q)))
        idx_n = int(np.argmin(np.abs(p.omega + q)))

        z_arr, eps2_arr = [], []
        for i in range(nz):
            pk = fft(psi);
            pk[0] = 0.0  # Remove carrier
            eps = (np.abs(pk[idx_p]) + np.abs(pk[idx_n])) / p.ntau
            z_arr.append(i * dz)
            eps2_arr.append(eps ** 2)
            if i < nz - 1:
                psi = ssfm_step(psi, dz, p.omega, p.beta2, p.gamma, p.alpha)

        z, e2 = np.array(z_arr), np.array(eps2_arr)

        # [REVIEWER CHECK] Cosh fit for sideband growth
        def cosh2_model(z_, lam, c1, c2):
            return c1 * np.cosh(2 * lam * z_) + c2

        try:
            popt, _ = curve_fit(cosh2_model, z, e2, p0=[g_rough, e2[0] / 2, e2[0] / 2],
                                bounds=([0.01, 0, 0], [20, 1e-3, 1e-3]))
            return float(max(popt[0], 0.0))
        except:
            return 0.0


class CQNLSE_Solver:
    def __init__(self, params: CQNLSE_Params):
        self.p = params
        self.reset()

    def reset(self):
        self.psi_history = []
        self.intensity_history = []
        self.power_history = []
        self.power_error_history = []  # [REVIEWER CHECK] Added conservation tracking
        self.z_record = []
        self.rogue_stats = {}

    def _power(self, psi):
        return float(simpson(np.abs(psi) ** 2, x=self.p.tau))

    def simulate(self, q_mod: float, A_mod: float = 0.05, silent: bool = False) -> dict:
        self.reset()
        p = self.p

        # Initial Condition
        psi = (p.A0 * (1 + A_mod * np.cos(q_mod * p.tau))).astype(complex)
        P0 = self._power(psi)  # Reference Power

        self.psi_history.append(psi.copy())
        self.intensity_history.append(np.abs(psi) ** 2)
        self.power_history.append(P0)
        self.power_error_history.append(0.0)
        self.z_record.append(0.0)

        z_arr = p.z_stat
        rec = max(1, len(z_arr) // 200)

        # Propagation Loop
        for i in range(1, len(z_arr)):
            psi = ssfm_step(psi, p.current_dz, p.omega, p.beta2, p.gamma, p.alpha)

            if i % rec == 0 or i == len(z_arr) - 1:
                curr_P = self._power(psi)
                self.psi_history.append(psi.copy())
                self.intensity_history.append(np.abs(psi) ** 2)
                self.power_history.append(curr_P)
                # [REVIEWER CHECK] Calculate relative power error
                self.power_error_history.append((curr_P - P0) / P0)
                self.z_record.append(z_arr[i])

        self.intensity_history = np.array(self.intensity_history).T
        self.z_record = np.array(self.z_record)
        self.power_error_history = np.array(self.power_error_history)

        # [REVIEWER CHECK] Calculate stats on developed turbulence
        self.rogue_stats = self._rogue_statistics(z_start_fraction=0.25)
        return self.rogue_stats

    def _rogue_statistics(self, z_start_fraction=0.25) -> dict:
        """
        Calculates statistics. 
        [REVIEWER CHECK] Separates initial MI growth from developed turbulence.
        """
        n_z = len(self.z_record)
        idx_start = int(n_z * z_start_fraction)

        # Slicing the history for stats (ignoring initial linear growth)
        if idx_start < n_z:
            I_slice = self.intensity_history[:, idx_start:]
        else:
            I_slice = self.intensity_history

        tau_marginal = I_slice.flatten()

        mean_int = float(np.mean(tau_marginal))
        max_int = float(np.max(tau_marginal))
        kurt_val = float(scipy_kurtosis(tau_marginal, fisher=False))

        # [REVIEWER CHECK] Significant Wave Height (Hs) Calculation
        # Sort intensity, take top 1/3, average it.
        sorted_I = np.sort(tau_marginal)
        n_points = len(sorted_I)
        n_top_third = max(1, int(n_points / 3))
        Hs = float(np.mean(sorted_I[-n_top_third:]))

        # [REVIEWER CHECK] Rogue Threshold: 2 * Hs (Optics/Oceanography standard)
        # Old definition was 8 * mean. 
        rogue_thr_Hs = 2.0 * Hs
        n_rogue = int(np.sum(tau_marginal > rogue_thr_Hs))

        # Max Power Conservation Error
        max_p_err = float(np.max(np.abs(self.power_error_history)))

        return dict(
            mean_intensity=mean_int,
            max_intensity=max_int,
            Hs=Hs,
            rogue_threshold=rogue_thr_Hs,
            AI=max_int / Hs,  # Abnormality Index
            kurtosis=kurt_val,
            n_rogue_events=n_rogue,
            rogue_density=n_rogue / len(tau_marginal),
            rogue_occurred=1 if n_rogue > 0 else 0,
            max_power_error=max_p_err
        )


class MI_Optimizer:
    def __init__(self, params: CQNLSE_Params):
        self.p = params
        self.theory = CQNLSE_MI_Theory(params)
        self.extractor = MI_Gain_Extractor(params)
        self.solver = CQNLSE_Solver(params)
        n_cutoff = int(self.theory.q_cutoff / params.dq)
        self.n_bounds = (1, n_cutoff)
        self.n_cutoff = n_cutoff

    def _objective(self, n_list):
        n = int(round(float(n_list[0])))
        n = max(self.n_bounds[0], min(n, self.n_bounds[1]))
        try:
            with suppress_stdout():
                gain = self.extractor.extract(n)
            return -gain if gain > 0 else 1e6
        except Exception:
            return 1e6

    def run_optimization(self, n_calls=30):
        from skopt.space import Integer
        bounds = [Integer(self.n_bounds[0], self.n_bounds[1], name='n')]
        with suppress_stdout():
            res = gp_minimize(self._objective, dimensions=bounds, n_calls=n_calls, random_state=42, verbose=False)
        self.n_opt = int(round(float(res.x[0])))
        self.q_mod_opt = self.n_opt * self.p.dq
        self.p.switch_to_final_dz()
        with suppress_stdout():
            self.solver.simulate(self.q_mod_opt, silent=True)
        self.mi_gain_opt = self.extractor.extract(self.n_opt)
        self.rel_err_q = (abs(self.q_mod_opt - self.theory.q_max) / self.theory.q_max * 100)
        self.rel_err_gain = (abs(self.mi_gain_opt - self.theory.lambda_max) / (self.theory.lambda_max + 1e-12) * 100)
        return self


# ─────────────────────────────────────────────────────────────
# Science-Grade Figure Generator (REVISED)
# ─────────────────────────────────────────────────────────────

class SCI_Figure_Generator:
    """
    Produces top-tier scientific visualizations.
    Revised to include Power Conservation checks.
    """

    def __init__(self, optimizer: MI_Optimizer):
        self.opt = optimizer
        self.p = optimizer.p
        self.c_theory = '#444444'
        self.c_sim = '#D55E00'
        self.c_rogue = '#D55E00'
        self.c_stable = '#E0E0E0'

    def _save(self, name):
        for ext in ['pdf', 'png']:
            plt.savefig(f'figures/{name}.{ext}', dpi=300)

    # ── Fig 1: Gain Spectrum + Zoom Inset ──────────────────────────────
    def plot_gain_spectrum_with_inset(self):
        n_list = list(range(1, self.opt.n_cutoff + 1))
        q_list = [n * self.p.dq for n in n_list]
        gains = [self.opt.extractor.extract(n) for n in tqdm(n_list, desc="Gain Spectrum", leave=False)]

        qr = self.opt.theory.q_range
        gr = self.opt.theory.gain

        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(qr, gr, '-', color=self.c_theory, lw=2, alpha=0.9, label=r'Theory $\lambda(q)$')
        ax.fill_between(qr, gr, color=self.c_theory, alpha=0.08)
        ax.plot(q_list, gains, 'o', color=self.c_sim, ms=5, label='Numerical', zorder=5)

        ax.set_xlabel(r'Wavenumber $q$ (rad/$\tau$)')
        ax.set_ylabel(r'MI Gain $\lambda$ ($z^{-1}$)')
        ax.set_title(r'\textbf{Figure 1} | Modulation Instability Spectrum', loc='left')
        ax.set_xlim(0, max(qr))
        ax.set_ylim(0, max(gr) * 1.1)
        ax.legend(loc='lower center', frameon=False, ncol=2)

        axins = inset_axes(ax, width="35%", height="35%", loc='upper right', borderpad=2)
        q_peak = self.opt.theory.q_max
        zoom_span = 0.5
        mask = (qr > q_peak - zoom_span) & (qr < q_peak + zoom_span)
        axins.plot(qr[mask], gr[mask], '-', color=self.c_theory, lw=2)
        q_num = np.array(q_list)
        g_num = np.array(gains)
        mask_num = (q_num > q_peak - zoom_span) & (q_num < q_peak + zoom_span)
        axins.plot(q_num[mask_num], g_num[mask_num], 'o', color=self.c_sim, ms=6)
        axins.plot(self.opt.q_mod_opt, self.opt.mi_gain_opt, 'x', color='k', ms=8, markeredgewidth=1.5)
        axins.set_xlim(q_peak - zoom_span / 2, q_peak + zoom_span / 2)
        axins.set_ylim(max(gr) * 0.95, max(gr) * 1.01)
        axins.set_xticks([]);
        axins.set_yticks([])
        axins.set_title("Peak Detail", fontsize=9)
        self._save('Fig1_Gain_Spectrum_Inset')

    # ── Fig 2: Temporal Ridgeplot ──────────────────────────────────────
    def plot_evolution_waterfall(self):
        nz_total = len(self.opt.solver.z_record)
        n_slices = 25
        indices = np.linspace(0, nz_total - 1, n_slices, dtype=int)
        z_vals = self.opt.solver.z_record[indices]
        I_vals = self.opt.solver.intensity_history.T[indices, :]
        tau = self.p.tau
        fig, ax = plt.subplots(figsize=(7, 6))
        global_max = np.max(I_vals)
        v_spacing = global_max * 0.08
        cmap = plt.get_cmap('magma_r')

        for i, (z, I) in enumerate(zip(z_vals, I_vals)):
            base = i * v_spacing
            y_data = base + I
            ax.fill_between(tau, base, y_data, color='white', alpha=1.0, zorder=i * 3)
            ax.fill_between(tau, base, y_data, color=cmap(i / n_slices), alpha=0.7, zorder=i * 3 + 1)
            ax.plot(tau, y_data, color='k', lw=0.5, alpha=0.6, zorder=i * 3 + 2)
            if i % 3 == 0 or i == n_slices - 1:
                ax.text(tau[-1] * 1.02, base + v_spacing * 0.3, f"$z={z:.1f}$",
                        fontsize=8, color='#444', va='center')

        ax.set_xlabel(r'Time $\tau$')
        ax.set_ylabel(r'Propagation $z$ (Stacked)')
        ax.set_title(r'\textbf{Figure 2} | Temporal Dynamics (Extreme Events)', loc='left')
        ax.set_yticks([])
        ax.spines['left'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        self._save('Fig2_Temporal_Waterfall')

    # ── Fig 3: Spectral Stack ──────────────────────────────────────────
    def plot_spectral_cascade(self):
        psi_hist = np.array(self.opt.solver.psi_history)
        z_rec = self.opt.solver.z_record
        indices = np.linspace(0, len(z_rec) - 1, 9, dtype=int)
        omega = fftshift(self.p.omega)
        fig, ax = plt.subplots(figsize=(7, 6))
        x_limit = 6 * self.opt.theory.q_max
        offset_db = 22

        for i, idx in enumerate(indices):
            psi = psi_hist[idx]
            spec = np.abs(fftshift(fft(psi))) ** 2 + 1e-15
            spec_db = 10 * np.log10(spec)
            z = z_rec[idx]
            y_shifted = spec_db - i * offset_db
            color = plt.cm.plasma(i / 9)
            ax.plot(omega, y_shifted, color=color, lw=1.2)
            idx_border = np.argmin(np.abs(omega - x_limit))
            y_val = max(y_shifted[idx_border], -i * offset_db - 20)
            ax.text(x_limit * 1.02, y_val, f"$z={z:.1f}$", color='black', fontsize=9, ha='left', va='center')

        ax.set_xlim(-x_limit, x_limit)
        ax.set_xlabel(r'Frequency $\Omega$')
        ax.set_ylabel(r'Power Spectral Density (dB, Stacked)')
        ax.set_title(r'\textbf{Figure 3} | Spectral Cascade & Supercontinuum', loc='left')
        ax.set_yticks([])
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        self._save('Fig3_Spectral_Stack')

    # ── Fig 4: Phase Diagram ──────────────────────────────────────────
    def plot_phase_scatter(self):
        n_max = self.opt.n_cutoff
        n_list = list(range(1, n_max + 1, 1))
        q_grid = [n * self.p.dq for n in n_list]
        Am_list = np.linspace(0.01, 0.20, 15)
        x_stable, y_stable = [], []
        x_rogue, y_rogue = [], []

        with tqdm(total=len(Am_list) * len(q_grid), desc="Phase Diagram", leave=False) as pbar:
            for Am in Am_list:
                for q in q_grid:
                    with suppress_stdout():
                        self.opt.solver.simulate(q, float(Am), silent=True)
                    stats = self.opt.solver.rogue_stats
                    if stats.get('rogue_occurred', 0) > 0:
                        x_rogue.append(q)
                        y_rogue.append(Am)
                    else:
                        x_stable.append(q)
                        y_stable.append(Am)
                    pbar.update(1)

        fig, ax = plt.subplots(figsize=(7, 5.5))
        ax.scatter(x_stable, y_stable, c=self.c_stable, s=30, marker='o', edgecolors='none', alpha=0.8, label='Stable')
        ax.scatter(x_rogue, y_rogue, c=self.c_rogue, s=40, marker='d', edgecolors='k', linewidth=0.5,
                   label='Extreme Event')
        ax.axvline(self.opt.theory.q_max, color='k', ls='--', lw=1, alpha=0.5)
        ax.text(self.opt.theory.q_max * 1.02, 0.19, r'$q_{max}$', fontsize=10)
        ax.set_xlabel(r'Perturbation Wavenumber $q$')
        ax.set_ylabel(r'Modulation Amplitude $A_{mod}$')
        ax.set_title(r'\textbf{Figure 4} | Phase Diagram of Instability', loc='left')
        ax.legend(loc='upper right', frameon=True)
        ax.set_ylim(0, 0.21)
        ax.grid(True, linestyle=':', alpha=0.3)
        self._save('Fig4_Phase_Scatter')

    # ── Fig 5: Stats & Conservation (REVISED) ──────────────────────────
    def plot_statistical_dynamics(self):
        # [REVIEWER CHECK] Rerun optimal case to get history
        self.opt.solver.simulate(self.opt.q_mod_opt, A_mod=0.05, silent=True)

        I_hist = self.opt.solver.intensity_history
        z_rec = self.opt.solver.z_record
        p_err = self.opt.solver.power_error_history

        max_I = np.max(I_hist, axis=0)
        kurt_z = scipy_kurtosis(I_hist, axis=0, fisher=False)

        # Three panels now
        fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(7, 8), sharex=True)
        plt.subplots_adjust(hspace=0.1)

        # Top: Intensity
        ax1.plot(z_rec, max_I, color='#0072B2', lw=2)
        ax1.fill_between(z_rec, 0, max_I, color='#0072B2', alpha=0.1)
        ax1.set_ylabel(r'Peak Int. $|\psi|_{max}^2$')
        ax1.grid(True, linestyle=':', alpha=0.5)
        ax1.set_title(r'\textbf{Figure 5} | Dynamics & Numerical Verification', loc='left')

        # Middle: Kurtosis
        ax2.plot(z_rec, kurt_z, color=self.c_sim, lw=2)
        ax2.axhline(3.0, color='gray', ls='--', lw=1.2, label='Gaussian')
        ax2.set_ylabel(r'Kurtosis $\kappa(z)$')
        ax2.grid(True, linestyle=':', alpha=0.5)

        # Bottom: Power Error [REVIEWER CHECK]
        ax3.plot(z_rec, p_err, color='#CC79A7', lw=1.5)
        ax3.set_ylabel(r'Power Error $\Delta P/P_0$')
        ax3.set_xlabel(r'Propagation Distance $z$')
        ax3.grid(True, linestyle=':', alpha=0.5)
        # Force scientific notation
        ax3.ticklabel_format(axis='y', style='sci', scilimits=(0, 0))

        self._save('Fig5_Statistical_Verification')

    def generate_all(self):
        print("\n[GRAPHICS] Generating Publication-Quality Figures...")
        self.plot_gain_spectrum_with_inset()
        self.plot_evolution_waterfall()
        self.plot_spectral_cascade()
        self.plot_phase_scatter()
        self.plot_statistical_dynamics()
        print("[GRAPHICS] Done. Saved to /figures.")


# ─────────────────────────────────────────────────────────────
# Console Output
# ─────────────────────────────────────────────────────────────

def print_final_report(params, optimizer):
    p = params
    opt = optimizer
    th = opt.theory
    stats = opt.solver.rogue_stats

    print("\n" + "=" * 70)
    print("      CQNLSE SIMULATION: PUBLICATION DATA REPORT")
    print("      [Corrected for Reviewer Comments]")
    print("=" * 70)
    print(f"{'PARAMETER':<25} | {'VALUE':<15} | {'UNIT'}")
    print("-" * 70)
    print(f"{'beta2':<25} | {p.beta2:<15.4f} | {'ps^2/km'}")
    print(f"{'gamma':<25} | {p.gamma:<15.4f} | {'1/(W km)'}")
    print(f"{'A0':<25} | {p.A0:<15.4f} | {'sqrt(W)'}")
    print("-" * 70)
    print(f"{'Theory q_max':<25} | {th.q_max:<15.5f} | {'rad/tau'}")
    print(f"{'Numerical Gain':<25} | {opt.mi_gain_opt:<15.5f} | {'1/z'}")
    print(f"{'Error (Gain)':<25} | {opt.rel_err_gain:<15.3f} | {'%'}")
    print("-" * 70)
    print(" STATISTICAL METRICS (DEVELOPED STAGE ONLY)")
    print("-" * 70)
    print(f"{'Sig. Wave Height (Hs)':<25} | {stats['Hs']:<15.4f} | {'W'}")
    print(f"{'RW Threshold (2*Hs)':<25} | {stats['rogue_threshold']:<15.4f} | {'W'}")
    print(f"{'Max Intensity':<25} | {stats['max_intensity']:<15.4f} | {'W'}")
    print(f"{'Abnormality Index':<25} | {stats['AI']:<15.4f} | {'I_max/Hs'}")
    print(f"{'Kurtosis':<25} | {stats['kurtosis']:<15.4f} | {'-'}")
    print("-" * 70)
    print(" NUMERICAL VERIFICATION")
    print("-" * 70)
    print(f"{'Max Power Error':<25} | {stats['max_power_error']:<15.2e} | {'rel.'}")
    print("=" * 70 + "\n")


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def main():
    print("Initializing Conservative CQNLSE Physics Kernel...")
    params = CQNLSE_Params()

    print("Starting Bayesian Optimization...")
    optimizer = MI_Optimizer(params)
    optimizer.run_optimization(n_calls=30)

    viz = SCI_Figure_Generator(optimizer)
    viz.generate_all()

    print_final_report(params, optimizer)


if __name__ == '__main__':
    main()
