import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from scipy.interpolate import griddata, Rbf
from scipy.stats import gaussian_kde
from scipy.linalg import eigvalsh
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import warnings
import time
import sys
from contextlib import contextmanager
from skopt import gp_minimize
from skopt.space import Space
from hyperopt import fmin, tpe, hp, Trials
from cma import CMAEvolutionStrategy

warnings.filterwarnings('ignore')


# ======================== 兼容层 & 工具函数 ========================
@contextmanager
def suppress_stdout():
    old_stdout = sys.stdout
    sys.stdout = open('nul' if sys.platform == 'win32' else '/dev/null', 'w')
    try:
        yield
    finally:
        sys.stdout.close()
        sys.stdout = old_stdout


def set_all_seeds(seed=42):
    np.random.seed(seed)


def calculate_convergence_step(scores, threshold_ratio=0.95):
    if len(scores) == 0 or max(scores) == 0:
        return np.inf
    max_score = max(scores)
    threshold = max_score * threshold_ratio
    for i, s in enumerate(scores):
        if s >= threshold:
            return i + 1
    return len(scores)


def format_number(x, decimals=2):
    return round(x, decimals) if x != np.inf else "inf"


def print_separator(char="=", length=80):
    print(char * length)


def print_progress(iter_num, max_iter, best_score, current_score, params, alg_name):
    progress = (iter_num / max_iter) * 100
    param_str = ", ".join([f"{p:6.3f}" for p in params[:6]])
    print(f"[{alg_name.upper()}] Iter {iter_num:3d}/{max_iter} | Progress: {progress:5.1f}% | "
          f"Current Score: {current_score:8.4f} | Best Score: {best_score:8.4f} | "
          f"Params (A,f,phi,alpha,beta2,sigma): ({param_str})")


# ======================== 核心优化类（6维 CQ-NLSE 保守系统） ========================
class CQNLSEOptimizer6D:
    """
    Optimizer for the conservative cubic-quintic NLSE (paper equation):

        i d_z psi - (beta2/2) d_tau^2 psi + gamma|psi|^2 psi + alpha|psi|^4 psi = 0

    which rearranges to:

        i d_z psi = (beta2/2) d_tau^2 psi - gamma|psi|^2 psi - alpha|psi|^4 psi

    Parameters optimized (6D):
        0: A_mod   - modulation amplitude
        1: f_mod   - modulation frequency (peak-gain wavenumber)
        2: phi0    - initial phase
        3: alpha   - quintic nonlinearity coefficient
        4: beta2   - group-velocity dispersion (< 0 for anomalous)
        5: sigma   - Gaussian envelope width

    Fixed parameters:
        gamma = 1.0   (cubic Kerr nonlinearity)

    This is a CONSERVATIVE system: no dissipation, no frequency offset,
    no gain. Power is conserved to machine precision at every step.
    """

    def __init__(self, tau_range=(-100, 100), ntau=512, z_max=30.0,
                 gamma=1.0, seed=42):
        self.tau_range = tau_range
        self.ntau = ntau
        self.z_max = z_max
        self.gamma = gamma          # fixed cubic nonlinearity
        self.seed = seed
        set_all_seeds(seed)

        self.tau = np.linspace(tau_range[0], tau_range[1], ntau, endpoint=False)
        self.dtau = self.tau[1] - self.tau[0]
        self.dz = 0.05              # propagation step
        self.z_steps = int(z_max / self.dz)
        self.z = np.linspace(0, z_max, self.z_steps)
        self.omega = 2 * np.pi * fftfreq(ntau, self.dtau)

        # ── 6D parameter bounds ─────────────────────────────────
        # Matches the parameter ranges used in the paper
        self.bounds = [
            (0.2,  2.5),            # 0: A_mod  modulation amplitude
            (0.05, 0.7),            # 1: f_mod  modulation frequency
            (0.0,  2 * np.pi),      # 2: phi0   initial phase
            (0.0,  0.5),            # 3: alpha  quintic coefficient
            (-2.8, -0.3),           # 4: beta2  dispersion (negative = anomalous)
            (5.0,  30.0),           # 5: sigma  Gaussian envelope width
        ]
        self.cmaes_bounds = [
            [b[0] for b in self.bounds],
            [b[1] for b in self.bounds]
        ]
        self.ndim = 6

        self.alg_run_time = {}
        self.alg_stop_reason = {}
        self.detailed_history = {}

        # BO uncertainty storage (same structure as original)
        self.bo_uncertainty = {
            'gp_model': None,
            'mu_grid': None,
            'sigma_grid': None,
            'Xi': None,             # A_mod grid
            'Yi': None,             # f_mod grid
            'fixed_params': {},
            'ei_history': [],
            'kernel_lengthscale': None,
            'posterior_at_samples': [],
            'high_dim_samples': [],
            'pca_transform': None,
            'tsne_transform': None
        }

        # Power conservation history (replaces energy_loss_history)
        self.power_error_history = {
            'bo': [], 'cmaes': [], 'tpe': [], 'random': []
        }

    # ──────────────────────────────────────────────────────────
    # SSFM core — implements paper equation (Sulem convention)
    # ──────────────────────────────────────────────────────────
    def _make_disp_op(self, beta2, dz):
        """
        Half-step linear (dispersion) operator for paper Eq.(1):

            i d_z psi = (beta2/2) d_tau^2 psi   [linear sub-problem]

        In Fourier space (d_tau^2 -> -omega^2):
            i d_z psi_hat = -(beta2/2) omega^2 psi_hat
            -> psi_hat(z+dz) = psi_hat(z) * exp(+i (beta2/2) omega^2 dz)

        Half-step: exp(+i (beta2/2) omega^2 (dz/2))

        NOTE: + sign (Sulem convention), opposite to Agrawal convention.
        Verified: only this sign gives MI growth for beta2 < 0, gamma > 0.
        """
        return np.exp(+1j * (beta2 / 2) * self.omega ** 2 * (dz / 2))

    def _nlse_step(self, psi, disp_half, dz, alpha):
        """
        Strang symmetric split-step for paper equation.
        Scheme: L(dz/2) -> N(dz) -> L(dz/2)

        Nonlinear sub-problem:
            i d_z psi = -gamma|psi|^2 psi - alpha|psi|^4 psi
            -> psi(z+dz) = psi(z) * exp(+i (gamma|psi|^2 + alpha|psi|^4) dz)

        Power conserved to machine precision (both operators strictly unitary).
        """
        psi = ifft(fft(psi) * disp_half)
        I = np.abs(psi) ** 2
        psi = psi * np.exp(1j * (self.gamma * I + alpha * I ** 2) * dz)
        psi = ifft(fft(psi) * disp_half)
        return psi

    # ──────────────────────────────────────────────────────────
    # Physical quantities
    # ──────────────────────────────────────────────────────────
    def _compute_C(self, A0, alpha, beta2):
        """
        Effective MI gain coefficient:
            C = |beta2| * (gamma * A0^2 + 2 * alpha * A0^4)
        """
        return abs(beta2) * (self.gamma * A0 ** 2 + 2 * alpha * A0 ** 4)

    def _compute_soliton_order(self, sigma, beta2, A_mod, alpha):
        """
        Soliton order N = sqrt(L_D / L_NL)
            L_D  = sigma^2 / (2 |beta2|)    dispersion length
            L_NL = 1 / (gamma * A_mod^2)    nonlinear length (cubic dominant)
        """
        L_D  = sigma ** 2 / (2 * abs(beta2) + 1e-12)
        L_NL = 1.0 / (self.gamma * A_mod ** 2 + 1e-12)
        return float(np.sqrt(L_D / L_NL)), L_D, L_NL

    # ──────────────────────────────────────────────────────────
    # Forward simulation
    # ──────────────────────────────────────────────────────────
    def simulate_evolution(self, params):
        """
        Simulate the conservative CQ-NLSE forward in z.

        Initial condition (same parameterization as original paper):
            psi(tau, 0) = exp(-tau^2 / (2 sigma^2))
                          * [1 + A_mod * cos(f_mod * tau + phi0)]
        """
        A_mod, f_mod, phi0, alpha, beta2, sigma = params

        # Initial condition
        psi0 = (np.exp(-self.tau ** 2 / (2 * sigma ** 2))
                * (1 + A_mod * np.cos(f_mod * self.tau + phi0)))
        psi0 = psi0.astype(complex)

        P0 = self.dtau * float(np.sum(np.abs(psi0) ** 2))   # Parseval power

        # Pre-compute dispersion operator (reused every step)
        disp_half = self._make_disp_op(beta2, self.dz)

        evolution = np.zeros((self.ntau, self.z_steps), dtype=np.complex128)
        evolution[:, 0] = psi0
        psi = psi0.copy()

        for i in range(1, self.z_steps):
            psi = self._nlse_step(psi, disp_half, self.dz, alpha)
            evolution[:, i] = psi

        amp = np.abs(evolution)
        max_amp = float(np.max(amp))

        # Spectra
        spectrum_initial = np.abs(fftshift(fft(evolution[:, 0])))
        peak_z_idx = int(np.argmax(np.max(amp, axis=0)))
        spectrum_peak = np.abs(fftshift(fft(evolution[:, peak_z_idx])))

        # Power conservation (key diagnostic for conservative system)
        P_final = self.dtau * float(np.sum(np.abs(psi) ** 2))
        power_error = abs(P_final - P0) / P0

        # Localization metrics
        mean_power = float(np.mean(np.sum(amp ** 2 * self.dtau, axis=0)))
        localization = max_amp / (mean_power + 1e-9)
        crest_ratio  = max_amp / (float(np.percentile(amp, 25)) + 1e-9)

        # MI gain and soliton order
        C = self._compute_C(A_mod * np.exp(-0), alpha, beta2)  # at envelope peak ~A_mod
        soliton_order, L_D, L_NL = self._compute_soliton_order(sigma, beta2, A_mod, alpha)

        return {
            'evolution': evolution,
            'spectrum_initial': spectrum_initial,
            'spectrum_peak': spectrum_peak,
            'psi0': psi0,
            'max_amp': max_amp,
            'params': params,
            'metrics': {
                'localization': localization,
                'crest_ratio': crest_ratio,
                'power_error': power_error,     # replaces mass_error
                'soliton_order': soliton_order,
                'L_D': L_D,
                'L_NL': L_NL,
                'C': C,
                'q_peak': float(np.sqrt(2 * C) / abs(beta2)) if C > 0 else 0.0,
                'lambda_max': float(C / abs(beta2)),
            }
        }

    # ──────────────────────────────────────────────────────────
    # Objective function
    # ──────────────────────────────────────────────────────────
    def evaluate(self, params):
        """
        Objective:  J = A_peak * localization * P_mass * S_N

        A_peak     : peak amplitude (maximize rogue wave intensity)
        localization: crest factor (spatial concentration)
        P_mass     : power conservation penalty (hard threshold 1e-8,
                     much tighter than original since system is conservative)
        S_N        : soft soliton-order constraint (Gaussian around N_target=1.1)
        """
        try:
            res = self.simulate_evolution(params)

            A_peak       = res['max_amp']
            localization = res['metrics']['localization']
            power_error  = res['metrics']['power_error']
            N            = res['metrics']['soliton_order']

            # Power conservation (conservative system should be near machine eps)
            # Use soft penalty so near-violations don't hard-zero the score
            P_mass = np.exp(-power_error / 1e-6) if power_error < 1e-3 else 0.1

            # Soft soliton-order constraint (same form as original)
            N_target = 1.1
            sigma_N  = 0.1
            N_min, N_max = 0.8, 1.5
            if N_min <= N <= N_max:
                S_N = np.exp(-2 * (N - N_target) ** 2 / sigma_N ** 2)
            elif N < N_min:
                S_N = float(np.clip(1.0 - 2.0 * (N_min - N), 0.1, 1.0))
            else:
                S_N = float(np.clip(1.0 - 0.5 * (N - N_max), 0.1, 1.0))

            return float(A_peak * localization * P_mass * S_N)
        except Exception:
            return 0.0

    # ──────────────────────────────────────────────────────────
    # Optimization algorithms (identical structure to original)
    # ──────────────────────────────────────────────────────────
    def bo_search(self, max_iter=100):
        start_time = time.time()
        history, scores = [], []
        best_score, best_params = 0.0, [0.0] * self.ndim
        self.detailed_history['bo'] = []
        self.power_error_history['bo'] = []

        print_separator("-", 60)
        print(f"Starting 6D Bayesian Optimization (max_iter={max_iter})")
        print(f"  Equation: i d_z psi - (beta2/2) d_tau^2 psi + gamma|psi|^2 psi + alpha|psi|^4 psi = 0")
        print(f"  Bounds: A_mod{self.bounds[0]}, f_mod{self.bounds[1]}, phi0{self.bounds[2]}, "
              f"alpha{self.bounds[3]}, beta2{self.bounds[4]}, sigma{self.bounds[5]}")

        try:
            def objective(params):
                return -self.evaluate(params)

            res = gp_minimize(
                objective,
                dimensions=self.bounds,
                n_calls=max_iter,
                random_state=self.seed,
                n_initial_points=20,
                verbose=False,
                n_restarts_optimizer=5,
                acq_func='EI'
            )

            # Extract GP model and uncertainty for (A_mod, f_mod) slice
            if hasattr(res, 'models') and len(res.models) > 0:
                gp_model = res.models[-1]
                self.bo_uncertainty['gp_model'] = gp_model

                best_6d = res.x
                fixed_params = {
                    'phi0':  best_6d[2],
                    'alpha': best_6d[3],
                    'beta2': best_6d[4],
                    'sigma': best_6d[5],
                }
                self.bo_uncertainty['fixed_params'] = fixed_params

                xi = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
                yi = np.linspace(self.bounds[1][0], self.bounds[1][1], 100)
                Xi, Yi = np.meshgrid(xi, yi)
                self.bo_uncertainty['Xi'] = Xi
                self.bo_uncertainty['Yi'] = Yi

                grid_pts = np.array([
                    [x, y, fixed_params['phi0'], fixed_params['alpha'],
                     fixed_params['beta2'], fixed_params['sigma']]
                    for x in xi for y in yi
                ])
                try:
                    mu, sig = gp_model.predict(grid_pts, return_std=True)
                    self.bo_uncertainty['mu_grid']    = mu.reshape(Xi.shape)
                    self.bo_uncertainty['sigma_grid'] = sig.reshape(Xi.shape)
                    if hasattr(gp_model, 'kernel_'):
                        k = gp_model.kernel_
                        if hasattr(k, 'length_scale'):
                            self.bo_uncertainty['kernel_lengthscale'] = k.length_scale
                        elif hasattr(k, 'k2') and hasattr(k.k2, 'length_scale'):
                            self.bo_uncertainty['kernel_lengthscale'] = k.k2.length_scale
                    print(f"  [OK] GP uncertainty extracted for (A_mod, f_mod) slice")
                    print(f"  [OK] Fixed: phi0={fixed_params['phi0']:.2f}, "
                          f"alpha={fixed_params['alpha']:.3f}, beta2={fixed_params['beta2']:.2f}, "
                          f"sigma={fixed_params['sigma']:.2f}")
                except Exception as e:
                    print(f"  [WARNING] Could not extract GP posterior: {e}")

            self.bo_uncertainty['high_dim_samples'] = res.x_iters

            for i, params in enumerate(res.x_iters):
                score = -res.func_vals[i]
                sim_res = self.simulate_evolution(params)
                self.power_error_history['bo'].append(sim_res['metrics']['power_error'])

                if score > best_score:
                    best_score = score
                    best_params = list(params)

                history.append({**sim_res, 'score': score})
                scores.append(score)

                if self.bo_uncertainty['gp_model'] is not None and i >= 20:
                    try:
                        mu_s, sig_s = self.bo_uncertainty['gp_model'].predict(
                            [params], return_std=True)
                        self.bo_uncertainty['posterior_at_samples'].append({
                            'iter': i + 1, 'params': params,
                            'mu': mu_s[0], 'sigma': sig_s[0],
                            'actual_score': score
                        })
                    except Exception:
                        pass

                self.detailed_history['bo'].append({
                    'iter': i + 1, 'params': params, 'score': score,
                    'max_amp': sim_res['max_amp'],
                    'localization': sim_res['metrics']['localization'],
                    'crest_ratio': sim_res['metrics']['crest_ratio'],
                    'power_error': sim_res['metrics']['power_error'],
                    'soliton_order': sim_res['metrics']['soliton_order'],
                    'C': sim_res['metrics']['C'],
                })

                if (i + 1) % 10 == 0 or i == max_iter - 1:
                    print_progress(i + 1, max_iter, best_score, score, params, 'bo')
                    print(f"      -> Power Error: {sim_res['metrics']['power_error']:.2e}  "
                          f"C={sim_res['metrics']['C']:.3f}  N={sim_res['metrics']['soliton_order']:.2f}")

            self.alg_stop_reason['bo'] = "6D BO completed normally"

        except Exception as e:
            print(f"[WARNING] BO error: {e}, falling back to random search")
            res_r = self.random_search(max_iter)
            history, scores = res_r['history'], res_r['scores']
            best_score, best_params = res_r['best_score'], res_r['best_params']
            self.alg_stop_reason['bo'] = "BO fallback to random search"

        self.alg_run_time['bo'] = time.time() - start_time

        print(f"BO Completed | Best Score: {best_score:.4f} | "
              f"A={best_params[0]:.3f}, f={best_params[1]:.3f}, phi={best_params[2]:.3f}, "
              f"alpha={best_params[3]:.3f}, beta2={best_params[4]:.3f}, sigma={best_params[5]:.2f}")
        if self.power_error_history['bo']:
            print(f"  -> Mean Power Error: {np.mean(self.power_error_history['bo']):.2e} (machine eps expected)")
        if self.bo_uncertainty['sigma_grid'] is not None:
            print(f"  -> GP Uncertainty: mean_sigma={np.mean(self.bo_uncertainty['sigma_grid']):.3f}, "
                  f"max_sigma={np.max(self.bo_uncertainty['sigma_grid']):.3f}")

        return {'history': history, 'scores': scores,
                'best_score': best_score, 'best_params': best_params}

    def cmaes_search(self, max_iter=100):
        start_time = time.time()
        history, scores = [], []
        best_score, best_params = 0.0, [0.0] * self.ndim
        self.detailed_history['cmaes'] = []
        self.power_error_history['cmaes'] = []

        print_separator("-", 60)
        print(f"Starting 6D CMA-ES (max_iter={max_iter})")

        try:
            # Physically motivated initial point
            x0 = [1.0, 0.3, np.pi, 0.05, -1.0, 10.0]
            sigma0 = 0.15

            with suppress_stdout():
                es = CMAEvolutionStrategy(
                    x0, sigma0,
                    {'bounds': self.cmaes_bounds, 'seed': self.seed, 'verbose': -9,
                     'tolfun': 1e-8, 'tolx': 1e-8,
                     'BoundaryHandler': 'BoundTransform',
                     'CMA_active': True,
                     'popsize': 16}
                )

            for i in range(max_iter):
                if es.stop():
                    self.alg_stop_reason['cmaes'] = "Converged"
                    break
                solutions = es.ask()
                scores_batch = [-self.evaluate(x) for x in solutions]
                es.tell(solutions, scores_batch)

                xb = es.result.xbest
                fb = -es.result.fbest
                sim = self.simulate_evolution(xb)
                self.power_error_history['cmaes'].append(sim['metrics']['power_error'])

                if fb > best_score:
                    best_score = fb
                    best_params = list(xb)

                history.append({**sim, 'score': fb})
                scores.append(fb)

                if (i + 1) % 10 == 0 or i == max_iter - 1:
                    sigma_es = es.sigma
                    eig_vals = eigvalsh(es.C)
                    eig_vals = np.maximum(eig_vals, 1e-10)
                    cond_C = np.max(eig_vals) / np.min(eig_vals)
                    print_progress(i + 1, max_iter, best_score, fb, xb, 'cmaes')
                    print(f"      CMA-ES: sigma={sigma_es:.4f}, cond(C)={cond_C:.1e}")
                    print(f"      -> Power Error: {sim['metrics']['power_error']:.2e}  "
                          f"C={sim['metrics']['C']:.3f}  N={sim['metrics']['soliton_order']:.2f}")

            if not es.stop():
                self.alg_stop_reason['cmaes'] = "Reached max iterations"

        except Exception as e:
            print(f"[WARNING] CMA-ES error: {e}, falling back to random search")
            r = self.random_search(max_iter)
            history, scores = r['history'], r['scores']
            best_score, best_params = r['best_score'], r['best_params']
            self.alg_stop_reason['cmaes'] = "CMA-ES fallback"

        self.alg_run_time['cmaes'] = time.time() - start_time
        print(f"CMA-ES Completed | Best Score: {best_score:.4f} | "
              f"A={best_params[0]:.3f}, f={best_params[1]:.3f}, phi={best_params[2]:.3f}, "
              f"alpha={best_params[3]:.3f}, beta2={best_params[4]:.3f}, sigma={best_params[5]:.2f}")
        if self.power_error_history['cmaes']:
            print(f"  -> Mean Power Error: {np.mean(self.power_error_history['cmaes']):.2e}")
        return {'history': history, 'scores': scores,
                'best_score': best_score, 'best_params': best_params}

    def tpe_search(self, max_iter=100):
        start_time = time.time()
        history, scores = [], []
        best_score, best_params = 0.0, [0.0] * self.ndim
        self.detailed_history['tpe'] = []
        self.power_error_history['tpe'] = []

        print_separator("-", 60)
        print(f"Starting 6D TPE (max_iter={max_iter})")

        try:
            space = {
                'A_mod':  hp.uniform('A_mod',  self.bounds[0][0], self.bounds[0][1]),
                'f_mod':  hp.uniform('f_mod',  self.bounds[1][0], self.bounds[1][1]),
                'phi0':   hp.uniform('phi0',   self.bounds[2][0], self.bounds[2][1]),
                'alpha':  hp.uniform('alpha',  self.bounds[3][0], self.bounds[3][1]),
                'beta2':  hp.uniform('beta2',  self.bounds[4][0], self.bounds[4][1]),
                'sigma':  hp.uniform('sigma',  self.bounds[5][0], self.bounds[5][1]),
            }

            def obj(p):
                return -self.evaluate([
                    p['A_mod'], p['f_mod'], p['phi0'],
                    p['alpha'], p['beta2'], p['sigma']
                ])

            tr = Trials()
            fmin(fn=obj, space=space, algo=tpe.suggest,
                 max_evals=max_iter, trials=tr, show_progressbar=False)

            for i, t in enumerate(tr.trials):
                p = t['misc']['vals']
                params = [
                    p['A_mod'][0], p['f_mod'][0], p['phi0'][0],
                    p['alpha'][0], p['beta2'][0], p['sigma'][0]
                ]
                score = -t['result']['loss']
                sim = self.simulate_evolution(params)
                self.power_error_history['tpe'].append(sim['metrics']['power_error'])

                if score > best_score:
                    best_score = score
                    best_params = list(params)

                history.append({**sim, 'score': score})
                scores.append(score)

                if (i + 1) % 10 == 0 or i == max_iter - 1:
                    print_progress(i + 1, max_iter, best_score, score, params, 'tpe')
                    print(f"      -> Power Error: {sim['metrics']['power_error']:.2e}  "
                          f"C={sim['metrics']['C']:.3f}  N={sim['metrics']['soliton_order']:.2f}")

            self.alg_stop_reason['tpe'] = "Reached max iterations"

        except Exception as e:
            print(f"[WARNING] TPE error: {e}, falling back to random search")
            r = self.random_search(max_iter)
            history, scores = r['history'], r['scores']
            best_score, best_params = r['best_score'], r['best_params']
            self.alg_stop_reason['tpe'] = "TPE fallback"

        self.alg_run_time['tpe'] = time.time() - start_time
        print(f"TPE Completed | Best Score: {best_score:.4f} | "
              f"A={best_params[0]:.3f}, f={best_params[1]:.3f}, phi={best_params[2]:.3f}, "
              f"alpha={best_params[3]:.3f}, beta2={best_params[4]:.3f}, sigma={best_params[5]:.2f}")
        if self.power_error_history['tpe']:
            print(f"  -> Mean Power Error: {np.mean(self.power_error_history['tpe']):.2e}")
        return {'history': history, 'scores': scores,
                'best_score': best_score, 'best_params': best_params}

    def random_search(self, max_iter=100):
        start_time = time.time()
        history, scores = [], []
        best_score, best_params = 0.0, [0.0] * self.ndim
        self.detailed_history['random'] = []
        self.power_error_history['random'] = []

        print_separator("-", 60)
        print(f"Starting 6D Random Search (max_iter={max_iter})")

        for i in range(max_iter):
            params = [np.random.uniform(b[0], b[1]) for b in self.bounds]
            score = self.evaluate(params)
            res = self.simulate_evolution(params)
            self.power_error_history['random'].append(res['metrics']['power_error'])

            if score > best_score:
                best_score = score
                best_params = list(params)

            history.append({**res, 'score': score})
            scores.append(score)

            if (i + 1) % 10 == 0 or i == max_iter - 1:
                print_progress(i + 1, max_iter, best_score, score, params, 'random')
                print(f"      -> Power Error: {res['metrics']['power_error']:.2e}  "
                      f"C={res['metrics']['C']:.3f}  N={res['metrics']['soliton_order']:.2f}")

        self.alg_run_time['random'] = time.time() - start_time
        self.alg_stop_reason['random'] = "Reached max iterations"
        print(f"Random Completed | Best Score: {best_score:.4f} | "
              f"A={best_params[0]:.3f}, f={best_params[1]:.3f}, phi={best_params[2]:.3f}, "
              f"alpha={best_params[3]:.3f}, beta2={best_params[4]:.3f}, sigma={best_params[5]:.2f}")
        if self.power_error_history['random']:
            print(f"  -> Mean Power Error: {np.mean(self.power_error_history['random']):.2e}")
        return {'history': history, 'scores': scores,
                'best_score': best_score, 'best_params': best_params}

    # ──────────────────────────────────────────────────────────
    # Power conservation analysis (doc13 journal style)
    # ──────────────────────────────────────────────────────────
    def plot_power_conservation_analysis(self, results,
                                          base_path='figures_6D/6D_Power_Conservation'):
        """
        Journal-quality power conservation figure (doc13 style).
        Conservative system: power error should be at machine epsilon.
        """
        algorithms = ['bo', 'cmaes', 'tpe', 'random']
        alg_labels = ['BO', 'CMA-ES', 'TPE', 'Random']

        fig, axes = plt.subplots(2, 2, figsize=(11, 9), layout="constrained")
        ((ax1, ax2), (ax3, ax4)) = axes

        # (a) Mean power error per algorithm
        avg_err = [np.mean(self.power_error_history[alg]) for alg in algorithms]
        x_pos = np.arange(len(algorithms))
        bars = ax1.bar(x_pos, avg_err, color=[PALETTE[alg] for alg in algorithms],
                       alpha=0.85, edgecolor='black', linewidth=1.2, width=0.6)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(alg_labels)
        ax1.set_ylabel(r'Mean Power Error $|\Delta P/P_0|$')
        ax1.set_title(r'(a) Power Conservation Robustness', loc='left')
        ax1.set_yscale('log')
        ax1.grid(axis='y', ls='--', alpha=0.3)
        for bar, val in zip(bars, avg_err):
            ax1.text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() * 1.5,
                     f'{val:.1e}', ha='center', va='bottom',
                     fontsize=9, fontweight='bold')

        # (b) Score vs power error
        for i, alg in enumerate(algorithms):
            sc = results[alg]['scores']
            pe = self.power_error_history[alg]
            n = min(len(sc), len(pe))
            ax2.scatter(pe[:n], sc[:n], color=PALETTE[alg],
                        label=alg_labels[i], alpha=0.6, s=40,
                        edgecolors='white', linewidths=0.5)
        ax2.set_xlabel(r'$|\Delta P/P_0|$')
        ax2.set_ylabel('Objective Score')
        ax2.set_xscale('log')
        ax2.set_title(r'(b) Score vs Power Error', loc='left')
        ax2.legend(loc='lower right')

        # (c) MI gain C trajectory
        for alg in algorithms:
            C_vals = [h['metrics']['C'] for h in results[alg]['history']]
            ax3.plot(range(len(C_vals)), C_vals,
                     color=PALETTE[alg], lw=2, alpha=0.85,
                     label=alg_labels[algorithms.index(alg)])
        ax3.set_xlabel('Iteration')
        ax3.set_ylabel(r'MI Gain $C$')
        ax3.set_title(r'(c) MI Gain Evaluation Trajectory', loc='left')
        ax3.legend(loc='upper right')

        # (d) Score convergence (step plot)
        for alg in algorithms:
            rolling_best = np.maximum.accumulate(results[alg]['scores'])
            ax4.step(range(len(rolling_best)), rolling_best, where='post',
                     color=PALETTE[alg], lw=2.5, alpha=0.9,
                     label=alg_labels[algorithms.index(alg)])
            ax4.scatter(len(rolling_best) - 1, rolling_best[-1],
                        color=PALETTE[alg], s=100, zorder=5,
                        edgecolors='white', linewidths=1.5)
        ax4.set_xlabel('Iteration')
        ax4.set_ylabel('Best Discovered Score')
        ax4.set_title(r'(d) Optimization Convergence', loc='left')
        ax4.legend(loc='lower right')

        for fmt in ['png', 'pdf']:
            path = f"{base_path}.{fmt}"
            plt.savefig(path)
            print(f"[SAVED] {path}")
        plt.close()


    # ──────────────────────────────────────────────────────────
    # Dimension reduction (identical to original)
    # ──────────────────────────────────────────────────────────
    def _perform_dimension_reduction(self, results):
        print("\nPerforming 6D -> 2D Dimension Reduction (PCA + t-SNE)...")
        all_samples = {}
        for alg in ['bo', 'cmaes', 'tpe', 'random']:
            if alg == 'bo' and self.bo_uncertainty['high_dim_samples']:
                samples = np.array(self.bo_uncertainty['high_dim_samples'])
            else:
                samples = np.array([h['params'] for h in results[alg]['history']])
            all_samples[alg] = samples

            if len(samples) > 10:
                pca = PCA(n_components=2, random_state=self.seed)
                pca_r = pca.fit_transform(samples)
                all_samples[f'{alg}_pca'] = pca_r

                perplexity = min(30, max(5, len(samples) - 1))
                tsne = TSNE(n_components=2, random_state=self.seed, perplexity=perplexity)
                tsne_r = tsne.fit_transform(samples)
                all_samples[f'{alg}_tsne'] = tsne_r

                print(f"  [OK] {alg.upper()}: 6D->2D "
                      f"(PCA explained var: {np.sum(pca.explained_variance_ratio_):.3f})")
        self.dim_reduction_results = all_samples

    # ──────────────────────────────────────────────────────────
    # Main runner
    # ──────────────────────────────────────────────────────────
    def run_all(self, max_iter=100):
        print_separator()
        print(f"6D CQ-NLSE Optimization (seed={self.seed}, max_iter={max_iter})")
        print(f"Equation: i d_z psi - (beta2/2) d_tau^2 psi + gamma|psi|^2 psi + alpha|psi|^4 psi = 0")
        print(f"gamma={self.gamma} (fixed),  conservative system (no dissipation)")
        print_separator()

        results = {
            'bo':     self.bo_search(max_iter),
            'cmaes':  self.cmaes_search(max_iter),
            'tpe':    self.tpe_search(max_iter),
            'random': self.random_search(max_iter),
        }

        # Physical interpretation of best BO parameters
        print_separator()
        print("[PHYSICS] 6D CQ-NLSE PARAMETER INTERPRETATION")
        print_separator()

        bp = results['bo']['best_params']
        best_idx = np.argmax(results['bo']['scores'])
        metrics = results['bo']['history'][best_idx]['metrics']

        print("Optimal 6D Parameters (BO best):")
        print(f"  A_mod  = {bp[0]:.3f}   modulation amplitude")
        print(f"  f_mod  = {bp[1]:.3f}   modulation frequency (= q_peak seed)")
        print(f"  phi0   = {bp[2]:.3f}   initial phase")
        print(f"  alpha  = {bp[3]:.4f}  quintic nonlinearity coefficient")
        print(f"  beta2  = {bp[4]:.3f}   group-velocity dispersion (anomalous: beta2<0)")
        print(f"  sigma  = {bp[5]:.2f}   Gaussian envelope width")
        print(f"  gamma  = {self.gamma}    cubic nonlinearity (fixed)")

        print("\nDerived MI quantities:")
        print(f"  C (MI gain)      = {metrics['C']:.4f}")
        print(f"  q_peak           = {metrics['q_peak']:.4f}")
        print(f"  lambda_max       = {metrics['lambda_max']:.4f}")
        print(f"  L_D (dispersion) = {metrics['L_D']:.4f}")
        print(f"  L_NL (nonlinear) = {metrics['L_NL']:.4f}")
        print(f"  N (soliton order)= {metrics['soliton_order']:.3f}")
        print(f"  Power error      = {metrics['power_error']:.2e}  (machine eps expected)")

        print("\nAlgorithm Ranking:")
        ranking = sorted(
            [('BO', results['bo']['best_score']),
             ('CMA-ES', results['cmaes']['best_score']),
             ('TPE', results['tpe']['best_score']),
             ('Random', results['random']['best_score'])],
            key=lambda x: x[1], reverse=True
        )
        for i, (alg, sc) in enumerate(ranking, 1):
            print(f"  {i}. {alg}: {sc:.4f}")

        print("\nComputational Time:")
        for alg in ['bo', 'cmaes', 'tpe', 'random']:
            print(f"  {alg.upper():6}: {self.alg_run_time.get(alg, 0):.2f}s")

        self._perform_dimension_reduction(results)
        self.plot_power_conservation_analysis(results)

        return results



# ======================== Journal-quality plotting configuration ========================
import matplotlib as mpl

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "axes.titleweight": "bold",
    "axes.labelweight": "bold",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "axes.linewidth": 1.5,
    "figure.dpi": 300,
    "savefig.dpi": 600,
    "axes.grid": False,
    "legend.frameon": False,
    "legend.fontsize": 10,
})

# Colorblind-safe high-contrast palette
PALETTE = {
    'bo':           '#003f5c',
    'cmaes':        '#d45087',
    'tpe':          '#ffa600',
    'random':       '#7a5195',
    'highlight':    '#ff0000',
    'bg_gradient':  ['#000004', '#2c115f', '#721f81', '#b63679', '#f1605d', '#feaf77', '#fcfdbf']
}
ROGUE_CMAP = LinearSegmentedColormap.from_list("nature_magma", PALETTE['bg_gradient'])
k_axis = None


def plot_rogue_monster_dynamics(optimizer, results, base_path='6D_NLSE_Dynamics'):
    if not results['bo']['history']:
        return

    best_idx = np.argmax(results['bo']['scores'])
    best_run = results['bo']['history'][best_idx]
    evolution = best_run['evolution']
    amp = np.abs(evolution)

    z = np.linspace(0, optimizer.z_max, amp.shape[1])
    tau = optimizer.tau
    global k_axis
    k_axis = fftshift(fftfreq(len(tau), optimizer.dtau)) * 2 * np.pi

    max_amp_z = np.max(amp, axis=0)
    peak_tau_idx, peak_z_idx = np.unravel_index(np.argmax(amp), amp.shape)

    fig = plt.figure(figsize=(12, 8), layout="constrained")
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 0.8, 1], height_ratios=[1, 0.4])

    # (a) 2D spatiotemporal evolution (pcolormesh for physical accuracy)
    ax_main = fig.add_subplot(gs[0, 0])
    T, Z = np.meshgrid(tau, z)
    im = ax_main.pcolormesh(Z, T, amp.T, cmap=ROGUE_CMAP, shading='gouraud')
    ax_main.scatter(z[peak_z_idx], tau[peak_tau_idx], s=120,
                    facecolors='none', edgecolors='#00ffcc', lw=2, marker='o')
    ax_main.annotate(r'Peak Amplitude',
                     xy=(z[peak_z_idx], tau[peak_tau_idx]),
                     xytext=(z[peak_z_idx] + 2, tau[peak_tau_idx] + 20),
                     color='#00ffcc', fontweight='bold',
                     arrowprops=dict(arrowstyle="->", color='#00ffcc', lw=1.5))
    ax_main.set_ylabel(r'Retarded Time $\tau$')
    ax_main.set_xlabel(r'Propagation Distance $z$')
    ax_main.set_title('(a) Spatiotemporal Evolution', loc='left')
    fig.colorbar(im, ax=ax_main, label=r'Amplitude $|\psi|$', pad=0.02)

    # (b) Spatial profiles
    ax_prof = fig.add_subplot(gs[0, 1])
    ax_prof.plot(amp[:, 0], tau, color='gray', lw=2, ls='--', label='Initial')
    ax_prof.plot(amp[:, peak_z_idx], tau, color=PALETTE['highlight'], lw=2.5, label='Peak')
    ax_prof.fill_betweenx(tau, amp[:, peak_z_idx], 0,
                          color=PALETTE['highlight'], alpha=0.15)
    ax_prof.set_xlabel(r'$|\psi|$')
    ax_prof.set_yticklabels([])
    ax_prof.set_title('(b) Profiles', loc='left')
    ax_prof.legend(loc='upper right')

    # (c) Spectral broadening
    ax_spec = fig.add_subplot(gs[1, 0])
    ax_spec.plot(k_axis, best_run['spectrum_initial'],
                 color='gray', lw=2, ls='--', label='Initial')
    ax_spec.plot(k_axis, best_run['spectrum_peak'],
                 color=PALETTE['bo'], lw=2.5, label='Peak')
    ax_spec.fill_between(k_axis, best_run['spectrum_peak'], 0,
                         color=PALETTE['bo'], alpha=0.15)
    ax_spec.set_xlabel(r'Wavenumber $k$')
    ax_spec.set_ylabel(r'$|\tilde{\psi}|$')
    ax_spec.set_xlim(-5, 5)
    ax_spec.set_title('(c) Spectral Broadening', loc='left')
    ax_spec.legend()

    # (d) Peak amplitude vs z
    ax_amp = fig.add_subplot(gs[1, 1])
    ax_amp.plot(z, max_amp_z, color=PALETTE['bo'], lw=2.5)
    ax_amp.fill_between(z, max_amp_z, 0, color=PALETTE['bo'], alpha=0.15)
    ax_amp.set_xlabel(r'Distance $z$')
    ax_amp.set_ylabel(r'$\max|\psi|$')
    ax_amp.set_title(r'(d) Peak vs $z$', loc='left')

    # (e) Parameter table
    ax_text = fig.add_subplot(gs[:, 2])
    ax_text.axis('off')
    params  = best_run['params']
    metrics = best_run['metrics']
    table_text = (
        r"$\mathbf{Optimized\ Parameters}$" + "\n" +
        u"\u2500" * 25 + "\n" +
        rf"$A_{{mod}}$ (Amplitude)  = {params[0]:.3f}" + "\n" +
        rf"$f_{{mod}}$ (Frequency)   = {params[1]:.3f}" + "\n" +
        rf"$\phi_0$ (Phase)        = {params[2]:.3f}" + "\n" +
        rf"$\alpha$ (Quintic)     = {params[3]:.4f}" + "\n" +
        rf"$\beta_2$ (Dispersion) = {params[4]:.3f}" + "\n" +
        rf"$\sigma$ (Width)        = {params[5]:.2f}" + "\n\n" +
        r"$\mathbf{Physical\ Properties}$" + "\n" +
        u"\u2500" * 25 + "\n" +
        rf"MI Gain $C$        = {metrics['C']:.3f}" + "\n" +
        rf"Soliton Order $N$ = {metrics['soliton_order']:.2f}" + "\n" +
        rf"Crest Ratio      = {metrics['crest_ratio']:.1f}" + "\n" +
        rf"$\Delta P / P_0$         = {metrics['power_error']:.1e}"
    )
    ax_text.text(0.1, 0.5, table_text, fontsize=12, va='center', ha='left',
                 linespacing=1.8,
                 bbox=dict(boxstyle="round4,pad=0.8", fc="#f8f9fa",
                           ec="#dee2e6", lw=1.5))

    for fmt in ['png', 'pdf']:
        path = f"figures_6D/{base_path}.{fmt}"
        plt.savefig(path)
        print(f"[SAVED] {path}")
    plt.close()


def plot_landscape_comparison(optimizer, results, base_path='6D_Landscape_Comparison'):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8), layout="constrained")
    algs   = ['bo', 'cmaes', 'tpe', 'random']
    titles = ['Bayesian Opt', 'CMA-ES', 'TPE', 'Random Search']

    scatter_last = None
    for idx, alg in enumerate(algs):
        hist = results[alg]['history']
        if not hist:
            continue

        x = np.array([h['params'][0] for h in hist])
        y = np.array([h['params'][1] for h in hist])
        z = np.array([h['score']      for h in hist])

        # Upper row: 2D parameter landscape (tricontourf avoids RBF artefacts)
        ax_top = axes[0, idx]
        if len(x) > 10:
            try:
                ax_top.tricontourf(x, y, z, levels=30, cmap='viridis', alpha=0.85)
                ax_top.tricontour(x, y, z, levels=10, colors='white',
                                  alpha=0.3, linewidths=0.5)
            except Exception:
                pass

        scatter_last = ax_top.scatter(x, y, c=z, cmap='viridis',
                                      edgecolor='black', s=40,
                                      linewidths=0.8, alpha=0.9, zorder=5)
        best_idx = np.argmax(z)
        ax_top.scatter(x[best_idx], y[best_idx], s=250,
                       facecolors='none', edgecolors=PALETTE['highlight'],
                       lw=3, marker='*', zorder=10)
        ax_top.set_title(f'{titles[idx]}\n(A_mod vs f_mod)')
        ax_top.set_xlabel(r'$A_{\rm mod}$')
        if idx == 0:
            ax_top.set_ylabel(r'$f_{\rm mod}$')
        ax_top.set_xlim(optimizer.bounds[0])
        ax_top.set_ylim(optimizer.bounds[1])

        # Lower row: t-SNE projection
        ax_bot = axes[1, idx]
        if f'{alg}_tsne' in optimizer.dim_reduction_results:
            tsne_data = optimizer.dim_reduction_results[f'{alg}_tsne']
            sc_bot = ax_bot.scatter(tsne_data[:, 0], tsne_data[:, 1], c=z,
                                    cmap='viridis', edgecolor='white',
                                    s=60, alpha=0.9)
            ax_bot.scatter(tsne_data[best_idx, 0], tsne_data[best_idx, 1],
                           s=250, facecolors='none',
                           edgecolors=PALETTE['highlight'], lw=3, marker='*')
            ax_bot.set_title('t-SNE Projection')
            ax_bot.set_xlabel('t-SNE 1')
            if idx == 0:
                ax_bot.set_ylabel('t-SNE 2')

    if scatter_last is not None:
        fig.colorbar(scatter_last, ax=axes.ravel().tolist(),
                     label='Objective Score', shrink=0.8, aspect=30, pad=0.02)

    for fmt in ['png', 'pdf']:
        path = f"figures_6D/{base_path}.{fmt}"
        plt.savefig(path)
        print(f"[SAVED] {path}")
    plt.close()


def plot_raincloud_statistics(optimizer, results, base_path='6D_Raincloud_Stats'):
    """
    True Raincloud Plot:
    left jittered scatter + right half-violin + inner boxplot
    """
    algs   = ['bo', 'cmaes', 'tpe', 'random']
    labels = ['BO', 'CMA-ES', 'TPE', 'Random']
    data   = [results[a]['scores'] for a in algs]
    colors = [PALETTE[a] for a in algs]

    fig, ax = plt.subplots(figsize=(9, 6), layout="constrained")

    # Half-violin
    parts = ax.violinplot(data, positions=np.arange(len(algs)),
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('white')
        pc.set_alpha(0.7)
        # Clip left half to create half-violin
        verts = pc.get_paths()[0].vertices
        verts[:, 0] = np.clip(verts[:, 0], i, np.inf)

    # Jittered scatter (rain drops)
    for i, (d, c) in enumerate(zip(data, colors)):
        xj = np.random.normal(i - 0.15, 0.04, size=len(d))
        ax.scatter(xj, d, alpha=0.6, s=20, color=c, edgecolors='none', zorder=2)

    # Boxplot overlay
    bp = ax.boxplot(data, positions=np.arange(len(algs)), widths=0.1,
                    patch_artist=True, showfliers=False, zorder=3)
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor('white')
        patch.set_edgecolor(color)
        patch.set_linewidth(1.5)
    for median in bp['medians']:
        median.set(color='black', linewidth=2)

    ax.set_xticks(np.arange(len(algs)))
    ax.set_xticklabels(labels, fontweight='bold', fontsize=12)
    ax.set_ylabel('Optimization Objective Score')
    ax.set_title('Performance Distribution (Raincloud Plot)', loc='left', pad=15)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_linewidth(1.5)
    ax.spines['left'].set_linewidth(1.5)
    ax.grid(axis='y', linestyle='--', alpha=0.4)

    for fmt in ['png', 'pdf']:
        path = f"figures_6D/{base_path}.{fmt}"
        plt.savefig(path)
        print(f"[SAVED] {path}")
    plt.close()


def plot_bo_uncertainty_analysis(optimizer, results, base_path='6D_BO_Uncertainty'):
    if optimizer.bo_uncertainty['sigma_grid'] is None:
        print("[WARNING] No GP uncertainty data available")
        return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.5), layout="constrained")
    ax_mu, ax_sigma, ax_ev = axes

    Xi    = optimizer.bo_uncertainty['Xi']
    Yi    = optimizer.bo_uncertainty['Yi']
    Mu    = optimizer.bo_uncertainty['mu_grid']
    Sigma = optimizer.bo_uncertainty['sigma_grid']

    bo_hist = results['bo']['history']
    x_s = np.array([h['params'][0] for h in bo_hist])
    y_s = np.array([h['params'][1] for h in bo_hist])

    # (a) GP posterior mean
    im_mu = ax_mu.contourf(Xi, Yi, -Mu, levels=50, cmap='magma', alpha=0.95)
    ax_mu.contour(Xi, Yi, -Mu, levels=15, colors='white', alpha=0.2, linewidths=0.5)
    ax_mu.scatter(x_s[:20], y_s[:20], c='white', edgecolor='black',
                  marker='s', s=30, label='Init', zorder=10)
    ax_mu.scatter(x_s[20:], y_s[20:], c='#00ffcc', edgecolor='black',
                  s=50, label='BO', zorder=10)
    ax_mu.set_title(r'(a) GP Posterior Mean ($\mu$)')
    ax_mu.set_xlabel(r'$A_{\rm mod}$')
    ax_mu.set_ylabel(r'$f_{\rm mod}$')
    ax_mu.legend(loc='lower right')
    fig.colorbar(im_mu, ax=ax_mu, label='Predicted Score')

    # (b) Uncertainty
    im_sig = ax_sigma.contourf(Xi, Yi, Sigma, levels=50, cmap='YlOrRd', alpha=0.95)
    ax_sigma.contour(Xi, Yi, Sigma, levels=10, colors='darkred',
                     alpha=0.3, linewidths=0.5)
    ax_sigma.plot(x_s[20:], y_s[20:], color='black', lw=1, alpha=0.4,
                  ls='--', zorder=5)
    ax_sigma.scatter(x_s[20:], y_s[20:], c='white', edgecolor='black',
                     s=40, zorder=10)
    ax_sigma.set_title(r'(b) Model Uncertainty ($\sigma$)')
    ax_sigma.set_xlabel(r'$A_{\rm mod}$')
    ax_sigma.set_ylabel(r'$f_{\rm mod}$')
    fig.colorbar(im_sig, ax=ax_sigma, label='Standard Deviation')

    # (c) Prediction vs reality with dual y-axis
    post = optimizer.bo_uncertainty['posterior_at_samples']
    if post:
        iters  = [p['iter']         for p in post]
        mu_s   = [-p['mu']          for p in post]
        sig_s  = [p['sigma']        for p in post]
        actual = [p['actual_score'] for p in post]

        ax_ev2 = ax_ev.twinx()
        l1, = ax_ev.plot(iters, mu_s, color=PALETTE['bo'], lw=2.5,
                         marker='o', label=r'Prediction ($\mu$)')
        l2, = ax_ev.plot(iters, actual, color=PALETTE['highlight'], lw=2,
                         marker='*', markersize=10, label='True Score')
        ax_ev.fill_between(iters,
                           np.array(mu_s) - np.array(sig_s),
                           np.array(mu_s) + np.array(sig_s),
                           color=PALETTE['bo'], alpha=0.15)
        l3, = ax_ev2.plot(iters, sig_s, color='#d45087', lw=2, ls='--',
                          marker='^', label=r'Uncertainty ($\sigma$)')
        ax_ev.set_xlabel('BO Iteration')
        ax_ev.set_ylabel('Objective Score', color=PALETTE['bo'])
        ax_ev2.set_ylabel('Uncertainty (Std. Dev.)', color='#d45087')
        ax_ev.set_title('(c) Exploitation vs Exploration')
        lines = [l1, l2, l3]
        ax_ev.legend(lines, [l.get_label() for l in lines], loc='upper left')
    else:
        ax_ev.text(0.5, 0.5, "No posterior data", ha='center', va='center',
                   transform=ax_ev.transAxes)

    for fmt in ['png', 'pdf']:
        path = f"figures_6D/{base_path}.{fmt}"
        plt.savefig(path)
        print(f"[SAVED] {path}")
    plt.close()


# plot_power_conservation_analysis is a method of CQNLSEOptimizer6D (defined above in the class)
# and is called by run_all().  The standalone functions below match the names used in __main__.

if __name__ == "__main__":
    import os
    os.makedirs("figures_6D", exist_ok=True)

    opt = CQNLSEOptimizer6D(
        tau_range=(-100, 100),
        ntau=512,
        z_max=30.0,
        gamma=1.0,
        seed=42
    )

    results = opt.run_all(max_iter=100)

    plot_rogue_monster_dynamics(opt, results,
                                base_path="6D_NLSE_Optimization_Results")
    plot_landscape_comparison(opt, results,
                              base_path="6D_Landscape_Comparison")
    plot_raincloud_statistics(opt, results,
                              base_path="6D_Raincloud_Statistics")
    plot_bo_uncertainty_analysis(opt, results,
                                 base_path="6D_BO_Uncertainty")

    print_separator()
    print("[DONE] 6D CQ-NLSE Optimization Complete!")
    print_separator()
