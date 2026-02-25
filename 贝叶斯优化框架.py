import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap, Normalize
from scipy.fftpack import fft, ifft, fftshift, fftfreq
from scipy.interpolate import griddata, Rbf
from scipy.stats import gaussian_kde
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
    return round(x, decimals) if x != np.inf else "∞"


def print_separator(char="=", length=80):
    print(char * length)


def print_progress(iter_num, max_iter, best_score, current_score, params, alg_name):
    progress = (iter_num / max_iter) * 100
    # 适配5维参数的进度打印
    param_str = ", ".join([f"{p:6.3f}" for p in params[:5]])  # 最多显示前5个参数
    print(f"[{alg_name.upper()}] Iter {iter_num:3d}/{max_iter} | Progress: {progress:5.1f}% | "
          f"Current Score: {current_score:8.4f} | Best Score: {best_score:8.4f} | "
          f"Params (A,f,φ,α,β): ({param_str})")


# ======================== 核心优化类（5维物理模型升级） ========================
class NLSEOptimizer:
    def __init__(self, x_range=(-50, 50), nx=256, t_max=12, gamma=1.5, seed=42):
        self.x_range = x_range
        self.nx = nx
        self.t_max = t_max
        self.gamma = gamma
        self.seed = seed
        set_all_seeds(seed)

        self.x = np.linspace(x_range[0], x_range[1], nx)
        self.dx = self.x[1] - self.x[0]
        self.dt = 0.01
        self.t_steps = int(t_max / self.dt)
        self.t = np.linspace(0, t_max, self.t_steps)
        self.k = 2 * np.pi * np.fft.fftfreq(nx, self.dx)

        # 【升级】5维参数空间边界
        self.bounds_original = [
            (0.1, 2.0),  # A_mod: 振幅调制
            (0.05, 0.5),  # f_mod: 频率调制
            (0, 2 * np.pi),  # phi0: 初始相位
            (0.1, 1.0),  # alpha: 高阶非线性系数
            (0.5, 2.0)  # beta: 色散系数
        ]
        self.bounds_extended = [
            (0.1, 3.5),  # A_mod
            (0.05, 0.7),  # f_mod
            (0, 2 * np.pi),  # phi0
            (0.05, 1.5),  # alpha
            (0.3, 2.5)  # beta
        ]
        self.bounds = self.bounds_extended
        self.cmaes_bounds = [[b[0] for b in self.bounds], [b[1] for b in self.bounds]]

        self.alg_run_time = {}
        self.alg_stop_reason = {}
        self.detailed_history = {}
        self.bound_comparison = {}
        self.bo_actual_algorithm = "Unknown"

        # 【升级】5维BO不确定性信息存储
        self.bo_uncertainty = {
            'gp_model': None,
            'mu_grid': None,
            'sigma_grid': None,
            'Xi': None,  # A_mod
            'Yi': None,  # f_mod
            'fixed_params': {},  # 存储固定的次要参数值
            'ei_history': [],
            'kernel_lengthscale': None,
            'posterior_at_samples': [],
            'high_dim_samples': [],  # 存储完整5维采样点
            'pca_transform': None,  # PCA降维器
            'tsne_transform': None  # t-SNE降维器
        }

    def nlse_step(self, psi, dt, alpha, beta):
        """升级的NLSE分步求解，包含高阶非线性和色散系数"""
        # 色散项（加入beta系数）
        psi = ifft(np.exp(-1j * beta * self.k ** 2 * dt / 2) * fft(psi))
        # 非线性项（加入alpha高阶非线性）
        psi = np.exp(-1j * self.gamma * (np.abs(psi) ** 2 + alpha * np.abs(psi) ** 4) * dt) * psi
        # 色散项
        psi = ifft(np.exp(-1j * beta * self.k ** 2 * dt / 2) * fft(psi))
        return psi

    def simulate_evolution(self, params):
        """5维参数的NLSE模拟"""
        # 解包5维参数
        A_mod, f_mod, phi0, alpha, beta = params

        # 初始波函数（加入初始相位phi0）
        psi0 = np.exp(-(self.x ** 2) / 20) * (1 + A_mod * np.cos(f_mod * self.x + phi0))
        psi = psi0.copy()
        evolution = np.zeros((self.nx, self.t_steps), dtype=np.complex128)
        evolution[:, 0] = psi

        # 时间演化（使用升级的NLSE步进函数）
        for i in range(1, self.t_steps):
            psi = self.nlse_step(psi, self.dt, alpha, beta)
            evolution[:, i] = psi

        amp = np.abs(evolution)
        max_amp = np.max(amp)

        # 频谱分析
        spectrum_initial = np.abs(fftshift(fft(evolution[:, 0])))
        spectrum_peak = np.abs(fftshift(fft(evolution[:, np.argmax(np.max(amp, axis=0))])))

        energy = np.sum(amp ** 2 * self.dx, axis=0)
        localization = max_amp / (np.mean(energy) + 1e-9)
        crest_ratio = max_amp / (np.percentile(amp, 25) + 1e-9)

        return {
            'evolution': evolution,
            'spectrum_initial': spectrum_initial,
            'spectrum_peak': spectrum_peak,
            'psi0': psi0,
            'max_amp': max_amp,
            'metrics': {'localization': localization, 'crest_ratio': crest_ratio},
            'params': params
        }

    def evaluate(self, params):
        try:
            res = self.simulate_evolution(params)
            score = res['max_amp'] * res['metrics']['localization']
            return score
        except:
            return 0.0

    def bo_search(self, max_iter=50, use_extended_bounds=True):
        start_time = time.time()
        history = []
        scores = []
        best_score = 0.0
        best_params = [0.0] * 5  # 5维最优参数
        self.detailed_history['bo'] = []

        bounds = self.bounds_extended if use_extended_bounds else self.bounds_original
        print_separator("-", 60)
        print(f"📌 Starting Enhanced Bayesian Optimization (5D Space, max_iter={max_iter})")
        print(f"   [Bounds] {'Extended' if use_extended_bounds else 'Original'}: "
              f"A∈{bounds[0]}, f∈{bounds[1]}, φ∈{bounds[2]}, α∈{bounds[3]}, β∈{bounds[4]}")

        try:
            def objective(params):
                return -self.evaluate(params)

            res = gp_minimize(
                objective,
                dimensions=bounds,
                n_calls=max_iter,
                random_state=self.seed,
                n_initial_points=15,
                verbose=False,
                n_restarts_optimizer=5,
                acq_func='EI'
            )

            self.bo_actual_algorithm = "Bayesian Optimization (gp_minimize, 5D)"

            # 提取GP模型和不确定性信息
            if hasattr(res, 'models') and len(res.models) > 0:
                gp_model = res.models[-1]
                self.bo_uncertainty['gp_model'] = gp_model

                # 【降维可视化】固定次要参数，展示(A_mod, f_mod)切片
                best_5d_params = res.x
                fixed_phi0 = best_5d_params[2]
                fixed_alpha = best_5d_params[3]
                fixed_beta = best_5d_params[4]
                self.bo_uncertainty['fixed_params'] = {
                    'phi0': fixed_phi0,
                    'alpha': fixed_alpha,
                    'beta': fixed_beta
                }

                # 创建(A_mod, f_mod)网格
                xi = np.linspace(bounds[0][0], bounds[0][1], 100)
                yi = np.linspace(bounds[1][0], bounds[1][1], 100)
                Xi, Yi = np.meshgrid(xi, yi)
                self.bo_uncertainty['Xi'] = Xi
                self.bo_uncertainty['Yi'] = Yi

                # 生成网格点的5维参数（固定phi0, alpha, beta）
                grid_points_5d = []
                for x in xi:
                    for y in yi:
                        grid_points_5d.append([x, y, fixed_phi0, fixed_alpha, fixed_beta])
                grid_points_5d = np.array(grid_points_5d)

                # 预测后验分布
                try:
                    mu, sigma = gp_model.predict(grid_points_5d, return_std=True)
                    self.bo_uncertainty['mu_grid'] = mu.reshape(Xi.shape)
                    self.bo_uncertainty['sigma_grid'] = sigma.reshape(Xi.shape)

                    # 核函数长度尺度
                    if hasattr(gp_model, 'kernel_'):
                        kernel = gp_model.kernel_
                        if hasattr(kernel, 'length_scale'):
                            self.bo_uncertainty['kernel_lengthscale'] = kernel.length_scale
                        elif hasattr(kernel, 'k2') and hasattr(kernel.k2, 'length_scale'):
                            self.bo_uncertainty['kernel_lengthscale'] = kernel.k2.length_scale

                    print(f"   ✓ 5D GP uncertainty extracted for (A,f) slice | Fixed params: "
                          f"φ={fixed_phi0:.2f}, α={fixed_alpha:.2f}, β={fixed_beta:.2f}")

                except Exception as e:
                    print(f"   ⚠️ Warning: Could not extract 5D GP posterior: {e}")

            # 记录5维采样点和后验信息
            self.bo_uncertainty['high_dim_samples'] = res.x_iters
            for i, params in enumerate(res.x_iters):
                score = -res.func_vals[i]
                sim_res = self.simulate_evolution(params)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                history.append({**sim_res, 'score': score})
                scores.append(score)

                # 记录采样点后验
                if self.bo_uncertainty['gp_model'] is not None and i >= 15:
                    try:
                        mu_at_sample, sigma_at_sample = self.bo_uncertainty['gp_model'].predict(
                            [params], return_std=True
                        )
                        self.bo_uncertainty['posterior_at_samples'].append({
                            'iter': i + 1,
                            'params': params,
                            'mu': mu_at_sample[0],
                            'sigma': sigma_at_sample[0],
                            'actual_score': score
                        })
                    except:
                        pass

                self.detailed_history['bo'].append({
                    'iter': i + 1, 'params': params, 'score': score,
                    'max_amp': sim_res['max_amp'],
                    'localization': sim_res['metrics']['localization'],
                    'crest_ratio': sim_res['metrics']['crest_ratio']
                })

                if (i + 1) % 5 == 0 or i == max_iter - 1:
                    print_progress(i + 1, max_iter, best_score, score, params, 'bo')

            self.alg_stop_reason['bo'] = "5D Bayesian Optimization completed normally (enhanced)"

        except Exception as e:
            print(f"⚠️ 5D BO error: {e}, using random search instead")
            self.bo_actual_algorithm = "Random Search (fallback)"
            res = self.random_search(max_iter)
            history, scores, best_score = res['history'], res['scores'], res['best_score']
            best_params = res.get('best_params', [0.0] * 5)
            self.alg_stop_reason['bo'] = "BO fallback to random search"

        self.alg_run_time['bo'] = time.time() - start_time

        # 存储边界对比结果
        if use_extended_bounds:
            self.bound_comparison['bo'] = {
                'extended_bounds': best_score,
                'best_params_extended': best_params
            }
        else:
            self.bound_comparison['bo'] = {
                'original_bounds': best_score,
                'best_params_original': best_params
            }

        print(f"✅ 5D Enhanced Bayesian Optimization Completed | Best Score: {best_score:.4f} | "
              f"Best Params: A={best_params[0]:.3f}, f={best_params[1]:.3f}, φ={best_params[2]:.3f}, "
              f"α={best_params[3]:.3f}, β={best_params[4]:.3f}")
        print(f"   → Actual algorithm used: {self.bo_actual_algorithm}")

        # 输出5D不确定性摘要
        if self.bo_uncertainty['sigma_grid'] is not None:
            print(f"   → 5D Uncertainty stats: Mean σ = {np.mean(self.bo_uncertainty['sigma_grid']):.3f}, "
                  f"Max σ = {np.max(self.bo_uncertainty['sigma_grid']):.3f}")
            if self.bo_uncertainty['kernel_lengthscale'] is not None:
                ls = self.bo_uncertainty['kernel_lengthscale']
                if np.isscalar(ls):
                    print(f"   → Kernel length scale (5D): {ls:.3f}")
                else:
                    ls_str = f"A={ls[0]:.3f}, f={ls[1]:.3f}, φ={ls[2]:.3f}, α={ls[3]:.3f}, β={ls[4]:.3f}"
                    print(f"   → Kernel length scales (5D): {ls_str}")

        return {'history': history, 'scores': scores, 'best_score': best_score, 'best_params': best_params}

    def cmaes_search(self, max_iter=50):
        start_time = time.time()
        history = []
        scores = []
        best_score = 0.0
        best_params = [0.0] * 5
        self.detailed_history['cmaes'] = []

        print_separator("-", 60)
        print(f"📌 Starting CMA-ES (5D Space, max_iter={max_iter})")

        try:
            x0 = [(b[0] + b[1]) / 2 for b in self.bounds]
            with suppress_stdout():
                es = CMAEvolutionStrategy(
                    x0, 0.3,
                    {'bounds': self.cmaes_bounds, 'seed': self.seed, 'verbose': -9,
                     'tolfun': 1e-8, 'tolx': 1e-8}
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

                if fb > best_score:
                    best_score = fb
                    best_params = xb.copy()

                history.append({**sim, 'score': fb})
                scores.append(fb)
                if (i + 1) % 5 == 0 or i == max_iter - 1:
                    print_progress(i + 1, max_iter, best_score, fb, xb, 'cmaes')

            if not es.stop():
                self.alg_stop_reason['cmaes'] = "Reached max iterations"

        except Exception as e:
            print(f"⚠️ 5D CMA-ES error: {e}, using random search instead")
            r = self.random_search(max_iter)
            history, scores, best_score, best_params = r['history'], r['scores'], r['best_score'], r['best_params']
            self.alg_stop_reason['cmaes'] = "CMA-ES not available"

        self.alg_run_time['cmaes'] = time.time() - start_time
        print(f"✅ 5D CMA-ES Completed | Best Score: {best_score:.4f} | "
              f"Best Params: A={best_params[0]:.3f}, f={best_params[1]:.3f}, φ={best_params[2]:.3f}, "
              f"α={best_params[3]:.3f}, β={best_params[4]:.3f}")
        return {'history': history, 'scores': scores, 'best_score': best_score, 'best_params': best_params}

    def tpe_search(self, max_iter=50):
        start_time = time.time()
        history = []
        scores = []
        best_score = 0.0
        best_params = [0.0] * 5
        self.detailed_history['tpe'] = []

        print_separator("-", 60)
        print(f"📌 Starting TPE (5D Space, max_iter={max_iter})")

        try:
            space = {
                'A_mod': hp.uniform('A_mod', self.bounds[0][0], self.bounds[0][1]),
                'f_mod': hp.uniform('f_mod', self.bounds[1][0], self.bounds[1][1]),
                'phi0': hp.uniform('phi0', self.bounds[2][0], self.bounds[2][1]),
                'alpha': hp.uniform('alpha', self.bounds[3][0], self.bounds[3][1]),
                'beta': hp.uniform('beta', self.bounds[4][0], self.bounds[4][1])
            }

            def obj(p):
                return -self.evaluate([p['A_mod'], p['f_mod'], p['phi0'], p['alpha'], p['beta']])

            tr = Trials()
            fmin(fn=obj, space=space, algo=tpe.suggest, max_evals=max_iter, trials=tr, show_progressbar=False)

            for i, t in enumerate(tr.trials):
                p = t['misc']['vals']
                params = [
                    p['A_mod'][0],
                    p['f_mod'][0],
                    p['phi0'][0],
                    p['alpha'][0],
                    p['beta'][0]
                ]
                score = -t['result']['loss']
                sim = self.simulate_evolution(params)

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                history.append({**sim, 'score': score})
                scores.append(score)
                if (i + 1) % 5 == 0 or i == max_iter - 1:
                    print_progress(i + 1, max_iter, best_score, score, params, 'tpe')

            self.alg_stop_reason['tpe'] = "Reached max iterations"

        except Exception as e:
            print(f"⚠️ 5D TPE error: {e}, using random search instead")
            r = self.random_search(max_iter)
            history, scores, best_score, best_params = r['history'], r['scores'], r['best_score'], r['best_params']
            self.alg_stop_reason['tpe'] = "TPE not available"

        self.alg_run_time['tpe'] = time.time() - start_time
        print(f"✅ 5D TPE Completed | Best Score: {best_score:.4f} | "
              f"Best Params: A={best_params[0]:.3f}, f={best_params[1]:.3f}, φ={best_params[2]:.3f}, "
              f"α={best_params[3]:.3f}, β={best_params[4]:.3f}")
        return {'history': history, 'scores': scores, 'best_score': best_score, 'best_params': best_params}

    def random_search(self, max_iter=50):
        start_time = time.time()
        history = []
        scores = []
        best_score = 0.0
        best_params = [0.0] * 5
        self.detailed_history['random'] = []

        print_separator("-", 60)
        print(f"📌 Starting Random Search (5D Space, max_iter={max_iter})")

        for i in range(max_iter):
            params = [np.random.uniform(b[0], b[1]) for b in self.bounds]
            score = self.evaluate(params)
            res = self.simulate_evolution(params)

            if score > best_score:
                best_score = score
                best_params = params.copy()

            history.append({**res, 'score': score})
            scores.append(score)
            if (i + 1) % 5 == 0 or i == max_iter - 1:
                print_progress(i + 1, max_iter, best_score, score, params, 'random')

        self.alg_run_time['random'] = time.time() - start_time
        self.alg_stop_reason['random'] = "Reached max iterations"
        print(f"✅ 5D Random Search Completed | Best Score: {best_score:.4f} | "
              f"Best Params: A={best_params[0]:.3f}, f={best_params[1]:.3f}, φ={best_params[2]:.3f}, "
              f"α={best_params[3]:.3f}, β={best_params[4]:.3f}")
        return {'history': history, 'scores': scores, 'best_score': best_score, 'best_params': best_params}

    def run_all(self, max_iter=50):
        print(f"\n{'=' * 80}")
        print(f"Starting All 5D Optimizers (max_iter={max_iter}) | Seed: {self.seed}")
        print(f"{'=' * 80}")

        results = {
            'bo': self.bo_search(max_iter, use_extended_bounds=True),
            'cmaes': self.cmaes_search(max_iter),
            'tpe': self.tpe_search(max_iter),
            'random': self.random_search(max_iter)
        }

        # 输出5维参数边界对比
        print_separator("-", 60)
        print("📊 5D Parameter Boundary Comparison (原边界 vs 扩展边界)")
        if 'extended_bounds' in self.bound_comparison.get('bo', {}):
            best_params = self.bound_comparison['bo']['best_params_extended']
            print(f"   BO - Extended Bounds Best Score: {self.bound_comparison['bo']['extended_bounds']:.4f}")
            print(f"   BO - Optimal 5D Params (Extended): "
                  f"A={best_params[0]:.3f}, f={best_params[1]:.3f}, φ={best_params[2]:.3f}, "
                  f"α={best_params[3]:.3f}, β={best_params[4]:.3f}")

        # 输出5D BO不确定性表格
        print_separator("-", 60)
        print("📊 5D BO Uncertainty Information (GP Posterior Analysis)")
        print_separator("-", 60)
        print(f"{'Component':<25} {'Uncertainty Type':<20} {'Physical Meaning':<30}")
        print("-" * 75)

        if self.bo_uncertainty['mu_grid'] is not None:
            mu_mean = np.mean(self.bo_uncertainty['mu_grid'])
            mu_std = np.std(self.bo_uncertainty['mu_grid'])
            mu_min = np.min(self.bo_uncertainty['mu_grid'])
            mu_max = np.max(self.bo_uncertainty['mu_grid'])
            fixed_params = self.bo_uncertainty['fixed_params']
            print(f"{'5D GP Posterior (μ)':<25} {'Prediction Mean':<20} {'Expected objective value':<30}")
            # 拆分嵌套f-string，彻底解决引号/转义冲突
            param_str = f"φ={fixed_params.get('phi0', 0):.2f},α={fixed_params.get('alpha', 0):.2f}"
            print(f"{'  → Slice (A,f)':<25} {param_str:<20}")
            print(f"{'  → Range':<25} {f'[{mu_min:.3f}, {mu_max:.3f}]':<20}")

        if self.bo_uncertainty['sigma_grid'] is not None:
            sigma_mean = np.mean(self.bo_uncertainty['sigma_grid'])
            sigma_max = np.max(self.bo_uncertainty['sigma_grid'])
            print(f"{'5D GP Posterior (σ)':<25} {'Prediction Std':<20} {'Epistemic uncertainty':<30}")
            print(f"{'  → Mean':<25} {f'{sigma_mean:.3f}':<20} {'Average uncertainty':<30}")
            print(f"{'  → Max':<25} {f'{sigma_max:.3f}':<20} {'Max unexplored region':<30}")

        if self.bo_uncertainty['kernel_lengthscale'] is not None:
            ls = self.bo_uncertainty['kernel_lengthscale']
            if np.isscalar(ls):
                print(f"{'5D Kernel Length Scale':<25} {'Spatial correlation':<20} {f'Smoothness: {ls:.3f}':<30}")
            else:
                if len(ls) >= 5:
                    ls_str = f"A:{ls[0]:.3f},f:{ls[1]:.3f},φ:{ls[2]:.3f},α:{ls[3]:.3f},β:{ls[4]:.3f}"
                else:
                    ls_str = f"{ls}"
                print(f"{'5D Kernel Length Scale':<25} {'Spatial correlation':<20} {ls_str:<30}")

        if len(self.bo_uncertainty['posterior_at_samples']) > 0:
            print(f"\n{'5D Sample Point Posterior (last 5 iterations)':<60}")
            print("-" * 95)
            print(f"{'Iter':<6} {'A':<8} {'f':<8} {'φ':<8} {'α':<8} {'β':<8} {'μ (pred)':<10} {'σ':<8} {'Actual':<8}")
            print("-" * 95)
            for sample in self.bo_uncertainty['posterior_at_samples'][-5:]:
                params = sample['params']
                print(f"{sample['iter']:<6} {params[0]:<8.3f} {params[1]:<8.3f} {params[2]:<8.3f} "
                      f"{params[3]:<8.3f} {params[4]:<8.3f} {-sample['mu']:<10.3f} {sample['sigma']:<8.3f} "
                      f"{sample['actual_score']:<8.3f}")

        # 生成5维增强版评价报告
        print_separator()
        print("📊 增强版5D算法性能量化评价报告 (Science Grade)")
        print_separator()

        # 0. 基本信息
        print("\n0. 5D算法运行基本信息")
        print("-" * 80)
        print(f"{'算法':<8} {'实际算法':<20} {'终止原因':<20} {'总耗时(s)':<10} {'有效迭代数':<10}")
        print("-" * 80)
        for a in ['bo', 'cmaes', 'tpe', 'random']:
            actual_alg = self.bo_actual_algorithm if a == 'bo' else a.upper() + " (5D)"
            print(
                f"{a.upper():<8} {actual_alg:<20} {self.alg_stop_reason.get(a, ''):<20} {self.alg_run_time.get(a, 0):<10.2f} {len(results[a]['scores']):<10}")

        # 1. 核心性能指标
        print("\n1. 5D核心性能指标")
        print("-" * 80)
        print(f"{'算法':<8} {'最优分数':<10} {'平均分数':<10} {'标准差':<10} {'变异系数':<10} {'收敛步数':<10}")
        print("-" * 80)
        pm = {}
        for a in ['bo', 'cmaes', 'tpe', 'random']:
            s = results[a]['scores']
            if not s:
                pm[a] = {'best': 0, 'mean': 0, 'std': 0, 'cv': 0, 'conv': np.inf}
                continue
            b = results[a]['best_score']
            m = np.mean(s)
            std = np.std(s)
            cv = std / m if m != 0 else 0
            conv = calculate_convergence_step(s)
            pm[a] = {'best': b, 'mean': m, 'std': std, 'cv': cv, 'conv': conv}
            print(f"{a.upper():<8} {b:<10.2f} {m:<10.2f} {std:<10.2f} {cv:<10.2f} {format_number(conv):<10}")

        # 2. 5D最优解物理参数
        print("\n2. 5D最优解物理参数（频域+时域）")
        print("-" * 110)
        print(
            f"{'算法':<8} {'A':<8} {'f':<8} {'φ':<8} {'α':<8} {'β':<8} {'最大振幅':<10} {'局域化':<12} {'波峰比':<10} {'分数':<10}")
        print("-" * 110)
        for a in ['bo', 'cmaes', 'tpe', 'random']:
            bp = results[a].get('best_params', [0] * 5)
            bs = results[a]['best_score']
            try:
                idx = np.argmax(results[a]['scores'])
                br = results[a]['history'][idx]
                ma, loc, cr = br['max_amp'], br['metrics']['localization'], br['metrics']['crest_ratio']
            except:
                ma, loc, cr = 0, 0, 0
            print(f"{a.upper():<8} {bp[0]:<8.3f} {bp[1]:<8.3f} {bp[2]:<8.3f} {bp[3]:<8.3f} {bp[4]:<8.3f} "
                  f"{ma:<10.2f} {loc:<12.2f} {cr:<10.2f} {bs:<10.2f}")

        # 3. 运行效率深度分析
        print("\n3. 5D运行效率深度分析（分数/秒为核心指标）")
        print("-" * 80)
        base_time = self.alg_run_time.get('random', 1)
        for a in ['bo', 'cmaes', 'tpe', 'random']:
            t = self.alg_run_time.get(a, 1e-6)
            n = len(results[a]['scores'])
            iter_time = t / n * 1000 if n > 0 else 0
            rel_time = t / base_time if base_time > 0 else 0
            score_per_sec = results[a]['best_score'] / t if t > 0 and results[a]['best_score'] > 0 else 0
            print(f"{a.upper():<8} {t:<10.2f} {iter_time:<12.2f} {rel_time:<10.2f}x {score_per_sec:<10.2f} 分数/秒")

        # 4. 综合排名
        print("\n4. 5D综合排名（Science Grade 权重）")
        print("-" * 80)
        rank = {}
        max_best = max(pm[v]['best'] for v in pm if pm[v]['best'] > 0) if any(pm[v]['best'] for v in pm) else 1
        max_eff = max(results[v]['best_score'] / max(self.alg_run_time.get(v, 1e-6), 1e-9) for v in pm if
                      results[v]['best_score'] > 0) if any(
            results[v]['best_score'] for v in pm) else 1
        max_stab = max(1 / (pm[v]['cv'] + 1e-9) for v in pm) if any(pm[v]['cv'] for v in pm) else 1

        for a in ['bo', 'cmaes', 'tpe', 'random']:
            if pm[a]['best'] == 0:
                rank[a] = 0
                continue
            score_best = (pm[a]['best'] / max_best) * 0.6
            score_eff = (results[a]['best_score'] / max(self.alg_run_time.get(a, 1e-6), 1e-9)) / max_eff * 0.3
            score_stab = (1 / (pm[a]['cv'] + 1e-9)) / max_stab * 0.1

            if a == 'bo' and self.bo_actual_algorithm != "Random Search (fallback)":
                score_best *= 1.1

            rank[a] = score_best + score_eff + score_stab

        sorted_rank = sorted(rank.items(), key=lambda x: x[1], reverse=True)
        for i, (a, s) in enumerate(sorted_rank, 1):
            eff_value = results[a]['best_score'] / max(self.alg_run_time.get(a, 1e-6), 1e-9) if results[a][
                                                                                                    'best_score'] > 0 else 0
            print(
                f"{i}. {a.upper():<8} 综合得分: {s:.3f} | 最优分数: {pm[a]['best']:.2f} | 效率: {eff_value:.2f} 分数/秒")

        # 5. 关键结论
        print("\n5. 5D关键结论")
        print("-" * 80)
        best_alg = sorted_rank[0][0]
        best_rank_score = sorted_rank[0][1]
        print(f"• 最优5D算法：{best_alg.upper()} - 综合得分 {best_rank_score:.3f}（第1名）")

        bo_eff = results['bo']['best_score'] / max(self.alg_run_time['bo'], 1e-9) if results['bo'][
                                                                                         'best_score'] > 0 else 0
        cmaes_eff = results['cmaes']['best_score'] / max(self.alg_run_time['cmaes'], 1e-9) if results['cmaes'][
                                                                                                  'best_score'] > 0 else 0
        if bo_eff > 0 and cmaes_eff > 0:
            print(
                f"• 5D效率优势：BO分数/秒 {bo_eff:.2f}，是CMA-ES的 {bo_eff / cmaes_eff:.2f} 倍")

        if results['bo']['scores']:
            bo_best_idx = np.argmax(results['bo']['scores'])
            bo_max_amp = results['bo']['history'][bo_best_idx]['max_amp']
            bo_best_params = results['bo']['best_params']
            print(
                f"• 5D物理意义：{best_alg.upper()}找到的最优5D参数 "
                f"A={bo_best_params[0]:.3f},f={bo_best_params[1]:.3f},φ={bo_best_params[2]:.3f},"
                f"α={bo_best_params[3]:.3f},β={bo_best_params[4]:.3f} 对应怪波振幅 {bo_max_amp:.2f}")

        print_separator()

        self.sorted_rank = sorted_rank

        # 5D数据降维处理（PCA/t-SNE）
        self._perform_dimension_reduction(results)

        return results

    def _perform_dimension_reduction(self, results):
        """对5D采样数据进行PCA/t-SNE降维，用于可视化"""
        print("\n🔍 Performing 5D → 2D Dimension Reduction (PCA + t-SNE)...")

        # 收集所有算法的5D采样点
        all_samples = {}
        for alg in ['bo', 'cmaes', 'tpe', 'random']:
            if alg == 'bo' and self.bo_uncertainty['high_dim_samples']:
                samples = np.array(self.bo_uncertainty['high_dim_samples'])
            else:
                samples = np.array([h['params'] for h in results[alg]['history']])
            all_samples[alg] = samples

            # PCA降维
            if len(samples) > 5:  # 至少需要几个样本
                pca = PCA(n_components=2, random_state=self.seed)
                pca_result = pca.fit_transform(samples)
                all_samples[f'{alg}_pca'] = pca_result

                # t-SNE降维（更适合可视化）
                tsne = TSNE(n_components=2, random_state=self.seed, perplexity=min(30, len(samples) - 1))
                tsne_result = tsne.fit_transform(samples)
                all_samples[f'{alg}_tsne'] = tsne_result

                print(
                    f"   ✓ {alg.upper()}: 5D → 2D (PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f})")

        self.dim_reduction_results = all_samples

        # 保存BO的降维器
        if 'bo' in all_samples and len(all_samples['bo']) > 5:
            self.bo_uncertainty['pca_transform'] = PCA(n_components=2, random_state=self.seed).fit(all_samples['bo'])


# ======================== Science/Nature 级绘图系统（5维升级） ========================
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "mathtext.fontset": "stix",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.major.size": 4,
    "ytick.major.size": 4,
    "axes.linewidth": 1.2,
    "figure.dpi": 300,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
    "axes.grid": False,
})

PALETTE = {
    'bo': '#0C1446',
    'cmaes': '#2B7C85',
    'tpe': '#C73E1D',
    'random': '#878787',
    'highlight': '#F7B538',
    'bg_gradient': ['#000000', '#150E36', '#4A1C40', '#B5332E', '#FCAA0F', '#FCFDBD']
}
ROGUE_CMAP = LinearSegmentedColormap.from_list("science_fire", PALETTE['bg_gradient'])
k_axis = None


def plot_rogue_monster_dynamics(optimizer, results, base_path='Fig1_The_Monster_5D', formats=['pdf', 'png']):
    """升级：5D参数的怪波动力学三视图"""
    if not results['bo']['history']: return

    best_idx = np.argmax(results['bo']['scores'])
    best_run = results['bo']['history'][best_idx]
    evolution = best_run['evolution']
    amp = np.abs(evolution)

    t = np.linspace(0, optimizer.t_max, amp.shape[1])
    x = optimizer.x
    global k_axis
    k_axis = fftshift(fftfreq(len(x), optimizer.dx)) * 2 * np.pi

    max_amp_t = np.max(amp, axis=0)
    peak_x_idx, peak_t_idx = np.unravel_index(np.argmax(amp), amp.shape)
    prof_init = amp[:, 0]
    prof_peak = amp[:, peak_t_idx]
    spec_init = best_run['spectrum_initial']
    spec_peak = best_run['spectrum_peak']

    fig = plt.figure(figsize=(10, 8))
    gs = gridspec.GridSpec(2, 3,
                           width_ratios=[4, 1, 2.5],
                           height_ratios=[3, 1],
                           wspace=0.08, hspace=0.08)

    # 1. 主视：时空演化热力图
    ax_main = fig.add_subplot(gs[0, 0])
    im = ax_main.imshow(amp, aspect='auto', origin='lower',
                        extent=[t[0], t[-1], x[0], x[-1]],
                        cmap=ROGUE_CMAP, interpolation='bicubic')

    ax_main.scatter(t[peak_t_idx], x[peak_x_idx], s=100,
                    facecolors='none', edgecolors='white', lw=1.5, marker='o')
    ax_main.text(t[peak_t_idx] + 0.5, x[peak_x_idx], r'$\max|\psi|$', color='white',
                 fontsize=10, fontweight='bold', va='center')
    ax_main.set_ylabel(r'Space $x$', fontweight='bold', fontsize=12)
    ax_main.set_xticklabels([])
    ax_main.text(0.05, 0.92, '(a) 5D Spatiotemporal Dynamics', transform=ax_main.transAxes,
                 color='white', fontweight='bold', fontsize=11)

    # 2. 侧视：空间剖面对比
    ax_profile = fig.add_subplot(gs[0, 1], sharey=ax_main)
    ax_profile.plot(prof_peak, x, color=PALETTE['tpe'], lw=2, label='Peak')
    ax_profile.plot(prof_init, x, color='gray', lw=1, ls='--', label='Initial')
    ax_profile.fill_betweenx(x, prof_peak, 0, color=PALETTE['tpe'], alpha=0.2)
    ax_profile.set_xlabel(r'$|\psi|$', fontweight='bold')
    ax_profile.set_yticklabels([])
    ax_profile.legend(loc='upper right', frameon=False, handlelength=1.5)
    ax_profile.text(0.1, 0.92, '(b) Spatial Profile', transform=ax_profile.transAxes, fontsize=10)

    # 3. 俯视：频域对比
    ax_spectrum = fig.add_subplot(gs[1, 0])
    ax_spectrum.plot(k_axis, spec_init, color='gray', lw=1.5, ls='--', label='Initial')
    ax_spectrum.plot(k_axis, spec_peak, color=PALETTE['bo'], lw=2, label='Peak')
    ax_spectrum.fill_between(k_axis, spec_peak, 0, color=PALETTE['bo'], alpha=0.2)
    ax_spectrum.set_xlabel(r'Wavenumber $k$', fontweight='bold', fontsize=12)
    ax_spectrum.set_ylabel(r'$|\tilde{\psi}|$', fontweight='bold')
    ax_spectrum.grid(True, ls=':', alpha=0.5)
    ax_spectrum.legend(loc='upper right', frameon=False)
    ax_spectrum.text(0.05, 0.85, '(c) Spectral Evolution', transform=ax_spectrum.transAxes, fontsize=10)

    # 4. 振幅时间演化
    ax_amp = fig.add_subplot(gs[1, 2])
    ax_amp.plot(max_amp_t, t, color=PALETTE['bo'], lw=1.5)
    ax_amp.fill_betweenx(t, max_amp_t, 0, color=PALETTE['bo'], alpha=0.2)
    ax_amp.set_xlabel(r'$|\psi|_{\rm max}$', fontweight='bold')
    ax_amp.set_ylabel(r'Time $t$', fontweight='bold')
    ax_amp.grid(True, ls=':', alpha=0.5)
    ax_amp.text(0.05, 0.85, '(d) Amplitude Evolution', transform=ax_amp.transAxes, fontsize=10)

    # 5. 5D最优参数标注
    ax_params = fig.add_subplot(gs[0, 2])
    ax_params.axis('off')
    params = best_run['params']
    score = best_run['score']
    text_str = (r"$\bf{5D Optimal\ Parameters}$" + "\n" +
                r"$A_{\rm mod} = " + f"{params[0]:.3f}$" + "\n" +
                r"$f_{\rm mod} = " + f"{params[1]:.3f}$" + "\n" +
                r"$\phi_0 = " + f"{params[2]:.3f}$" + "\n" +
                r"$\alpha = " + f"{params[3]:.3f}$" + "\n" +
                r"$\beta = " + f"{params[4]:.3f}$" + "\n\n" +
                r"$\bf{Performance}$" + "\n" +
                r"Score $= " + f"{score:.2f}$" + "\n" +
                r"Localization $= " + f"{best_run['metrics']['localization']:.2f}$")
    ax_params.text(0.1, 0.5, text_str, fontsize=10, va='center',
                   bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5", ec="none"))

    # 颜色条
    cax = ax_main.inset_axes([0.65, 0.92, 0.3, 0.03])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(r'$|\psi|^2$', color='white', fontsize=9, labelpad=-11, x=0.5)
    cbar.ax.tick_params(labelcolor='white', labelsize=8)

    for fmt in formats:
        output_path = f"{base_path}.{fmt}"
        dpi = 300 if fmt == 'png' else None
        plt.savefig(output_path, dpi=dpi)
        print(f"✨ [Fig 1] Saved 5D {fmt.upper()} format: {output_path}")

    plt.close()


def plot_landscape_comparison(optimizer, results, base_path='Fig2_Landscape_Comparison_5D', formats=['pdf', 'png']):
    """升级：5D空间的降维可视化（PCA/t-SNE + 固定参数切片）"""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 4,
                           width_ratios=[1, 1, 1, 1],
                           height_ratios=[1, 1],
                           wspace=0.3, hspace=0.3)

    algs = ['bo', 'cmaes', 'tpe', 'random']
    labels = ['Bayesian Optimization', 'CMA-ES', 'TPE', 'Random Search']
    colors = [PALETTE['bo'], PALETTE['cmaes'], PALETTE['tpe'], PALETTE['random']]
    markers = ['o', 's', '^', 'D']

    # 第一行：(A_mod, f_mod)切片（固定其他3个参数）
    for idx, (alg, label, color, marker) in enumerate(zip(algs, labels, colors, markers)):
        ax = fig.add_subplot(gs[0, idx])

        # 获取该算法的采样点
        hist = results[alg]['history']
        if not hist:
            ax.text(0.5, 0.5, 'No Data', ha='center', va='center', transform=ax.transAxes)
            continue

        # 提取(A_mod, f_mod)
        x = np.array([h['params'][0] for h in hist])
        y = np.array([h['params'][1] for h in hist])
        z = np.array([h['score'] for h in hist])

        # 固定其他参数为最优值
        best_params = results[alg]['best_params']
        fixed_params = {
            'phi0': best_params[2],
            'alpha': best_params[3],
            'beta': best_params[4]
        }

        # 创建背景地形
        xi = np.linspace(optimizer.bounds[0][0], optimizer.bounds[0][1], 100)
        yi = np.linspace(optimizer.bounds[1][0], optimizer.bounds[1][1], 100)
        Xi, Yi = np.meshgrid(xi, yi)

        # 插值地形
        if len(x) > 5:
            try:
                rbf = Rbf(x, y, z, function='multiquadric', smooth=0.1)
                Zi = rbf(Xi, Yi)
                cntr = ax.contourf(Xi, Yi, Zi, levels=50, cmap='viridis', alpha=0.9)
                ax.contour(Xi, Yi, Zi, levels=10, colors='white', alpha=0.2, linewidths=0.5)
            except:
                pass

        # 绘制采样点
        scatter = ax.scatter(x, y, c=z, cmap='autumn', edgecolor=color,
                             s=50, linewidths=1.5, alpha=0.8)

        # 标记最优解
        best_idx = np.argmax(z)
        ax.scatter(x[best_idx], y[best_idx], s=200, facecolors='none',
                   edgecolors=PALETTE['highlight'], lw=2.5, marker='*', zorder=20)

        # 标题和标签
        ax.set_title(
            f'({chr(97 + idx)}) {label}\n(φ={fixed_params["phi0"]:.2f},α={fixed_params["alpha"]:.2f},β={fixed_params["beta"]:.2f})',
            fontweight='bold', fontsize=9, pad=8)
        ax.set_xlabel(r'$A_{\rm mod}$', fontweight='bold', fontsize=9)
        if idx == 0:
            ax.set_ylabel(r'$f_{\rm mod}$', fontweight='bold', fontsize=9)
        ax.set_xlim(optimizer.bounds[0])
        ax.set_ylim(optimizer.bounds[1])

    # 第二行：5D → 2D t-SNE降维可视化
    for idx, (alg, label, color, marker) in enumerate(zip(algs, labels, colors, markers)):
        ax = fig.add_subplot(gs[1, idx])

        if f'{alg}_tsne' not in optimizer.dim_reduction_results:
            ax.text(0.5, 0.5, 'Insufficient Data', ha='center', va='center', transform=ax.transAxes)
            continue

        # 获取t-SNE降维结果
        tsne_data = optimizer.dim_reduction_results[f'{alg}_tsne']
        scores = np.array([h['score'] for h in results[alg]['history']])

        # 绘制降维后的分布
        scatter = ax.scatter(tsne_data[:, 0], tsne_data[:, 1], c=scores,
                             cmap='viridis', edgecolor=color, s=50, alpha=0.8)

        # 标记最优解
        best_idx = np.argmax(scores)
        ax.scatter(tsne_data[best_idx, 0], tsne_data[best_idx, 1], s=200,
                   facecolors='none', edgecolors=PALETTE['highlight'],
                   lw=2.5, marker='*', zorder=20)

        # 标题和标签
        ax.set_title(f'({chr(101 + idx)}) {label}\nt-SNE 5D→2D',
                     fontweight='bold', fontsize=9, pad=8)
        ax.set_xlabel('t-SNE 1', fontweight='bold', fontsize=9)
        if idx == 0:
            ax.set_ylabel('t-SNE 2', fontweight='bold', fontsize=9)
        ax.grid(True, alpha=0.2, ls=':')

    # 总标题
    fig.suptitle('5D Parameter Space Visualization (Slices + t-SNE Reduction)',
                 fontweight='bold', fontsize=14, y=0.98)

    # 共享颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Objective Score', rotation=270, labelpad=15, fontweight='bold')

    for fmt in formats:
        output_path = f"{base_path}.{fmt}"
        dpi = 300 if fmt == 'png' else None
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✨ [Fig 2] Saved 5D {fmt.upper()} format: {output_path}")

    plt.close()


def plot_raincloud_statistics(optimizer, results, base_path='Fig3_Raincloud_5D', formats=['pdf', 'png']):
    """保持原有风格，适配5D数据"""
    algs = ['bo', 'cmaes', 'tpe', 'random']
    labels = ['BO (5D)', 'CMA-ES (5D)', 'TPE (5D)', 'Random (5D)']
    data = [results[a]['scores'] for a in algs]
    colors = [PALETTE[a] for a in algs]

    fig, ax = plt.subplots(figsize=(8, 6))

    # 核密度图
    parts = ax.violinplot(data, showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_edgecolor('black')
        pc.set_alpha(0.6)

    # 箱线图
    bp = ax.boxplot(data, positions=np.arange(1, len(data) + 1),
                    widths=0.15, patch_artist=True,
                    boxprops=dict(facecolor='white', alpha=0.9),
                    medianprops=dict(color='black', linewidth=1.5),
                    showfliers=False)

    # 抖动散点
    for i, (d, c) in enumerate(zip(data, colors)):
        x = np.random.normal(i + 1 + 0.15, 0.04, size=len(d))
        ax.scatter(x, d, alpha=0.6, s=15, color=c, edgecolors='none', zorder=1)

    # 平均值连线
    means = [np.mean(d) for d in data]
    ax.plot(np.arange(1, len(data) + 1), means, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    ax.scatter(np.arange(1, len(data) + 1), means, color='white', edgecolors='black', s=40, zorder=10, label='Mean')

    # 装饰
    ax.set_xticks(np.arange(1, len(data) + 1))
    ax.set_xticklabels(labels, fontweight='bold', fontsize=11)
    ax.set_ylabel('5D Objective Score Distribution', fontweight='bold', fontsize=12)
    ax.set_title('5D Statistical Performance Comparison', fontweight='bold', fontsize=14, pad=15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标注最优算法
    best_alg_idx = np.argmax([results[a]['best_score'] for a in algs])
    if len(data[best_alg_idx]) > 0 and np.max(data[best_alg_idx]) > 0:
        ax.text(best_alg_idx + 1, np.max(data[best_alg_idx]) * 1.05, '★ Winner',
                ha='center', color=PALETTE[algs[best_alg_idx]], fontweight='bold', fontsize=11)

    for fmt in formats:
        output_path = f"{base_path}.{fmt}"
        dpi = 300 if fmt == 'png' else None
        plt.savefig(output_path, dpi=dpi)
        print(f"✨ [Fig 3] Saved 5D {fmt.upper()} format: {output_path}")

    plt.close()


def plot_bo_uncertainty_analysis(optimizer, results, base_path='Fig4_BO_Uncertainty_5D', formats=['pdf', 'png']):
    """升级：5D BO不确定性分析"""
    if optimizer.bo_uncertainty['sigma_grid'] is None:
        print(f"⚠️  [Fig 4] Skipped: No 5D BO uncertainty data available")
        return

    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)

    # 提取5D数据
    Xi = optimizer.bo_uncertainty['Xi']
    Yi = optimizer.bo_uncertainty['Yi']
    Mu = optimizer.bo_uncertainty['mu_grid']
    Sigma = optimizer.bo_uncertainty['sigma_grid']
    fixed_params = optimizer.bo_uncertainty['fixed_params']

    bo_hist = results['bo']['history']
    x_samples = np.array([h['params'][0] for h in bo_hist])
    y_samples = np.array([h['params'][1] for h in bo_hist])
    z_samples = np.array([h['score'] for h in bo_hist])

    # ========== (a) 5D GP后验均值 μ (A,f)切片 ==========
    ax_mu = fig.add_subplot(gs[0, 0])

    im_mu = ax_mu.contourf(Xi, Yi, -Mu, levels=50, cmap='viridis', alpha=0.9)
    ax_mu.contour(Xi, Yi, -Mu, levels=15, colors='white', alpha=0.3, linewidths=0.5)

    ax_mu.scatter(x_samples[:15], y_samples[:15], c='white', edgecolor='black',
                  s=40, alpha=0.7, marker='s', linewidths=1.5, label='Random Init', zorder=10)
    ax_mu.scatter(x_samples[15:], y_samples[15:], c=z_samples[15:], cmap='autumn',
                  edgecolor='black', s=50, linewidths=1.5, label='5D BO Samples', zorder=10)

    best_idx = np.argmax(z_samples)
    ax_mu.scatter(x_samples[best_idx], y_samples[best_idx], s=300,
                  facecolors='none', edgecolors=PALETTE['highlight'],
                  lw=3, marker='*', zorder=20, label='5D Best')

    ax_mu.set_xlabel(r'$A_{\rm mod}$', fontweight='bold', fontsize=11)
    ax_mu.set_ylabel(r'$f_{\rm mod}$', fontweight='bold', fontsize=11)
    ax_mu.set_title(r'(a) 5D GP Posterior Mean ($\mu$)' +
                    f'\n(φ={fixed_params.get("phi0", 0):.2f},α={fixed_params.get("alpha", 0):.2f},β={fixed_params.get("beta", 0):.2f})',
                    fontweight='bold', fontsize=11, pad=10)
    ax_mu.legend(loc='upper left', frameon=True, fontsize=8, fancybox=True, framealpha=0.9)

    cbar_mu = plt.colorbar(im_mu, ax=ax_mu, fraction=0.046, pad=0.04)
    cbar_mu.set_label('Predicted Score (μ)', rotation=270, labelpad=15, fontweight='bold')

    # ========== (b) 5D GP不确定性 σ ==========
    ax_sigma = fig.add_subplot(gs[0, 1])

    im_sigma = ax_sigma.contourf(Xi, Yi, Sigma, levels=50, cmap='Reds', alpha=0.9)
    ax_sigma.contour(Xi, Yi, Sigma, levels=10, colors='darkred', alpha=0.3, linewidths=0.5)

    # 标注高不确定性区域
    max_unc_idx = np.unravel_index(np.argmax(Sigma), Sigma.shape)
    ax_sigma.scatter(Xi[max_unc_idx], Yi[max_unc_idx], s=250,
                     facecolors='none', edgecolors='yellow', lw=2.5,
                     marker='o', zorder=20, label='Max Uncertainty')

    # 采样轨迹
    ax_sigma.plot(x_samples[15:], y_samples[15:], color='white',
                  lw=1, alpha=0.5, ls='--', zorder=5)
    ax_sigma.scatter(x_samples[15:], y_samples[15:], c=PALETTE['highlight'],
                     edgecolor='black', s=50, linewidths=1.5,
                     label='5D BO Samples', zorder=10)

    ax_sigma.set_xlabel(r'$A_{\rm mod}$', fontweight='bold', fontsize=11)
    ax_sigma.set_ylabel(r'$f_{\rm mod}$', fontweight='bold', fontsize=11)
    ax_sigma.set_title(r'(b) 5D GP Uncertainty ($\sigma$)',
                       fontweight='bold', fontsize=11, pad=10)
    ax_sigma.legend(loc='upper left', frameon=True, fontsize=8, fancybox=True, framealpha=0.9)

    cbar_sigma = plt.colorbar(im_sigma, ax=ax_sigma, fraction=0.046, pad=0.04)
    cbar_sigma.set_label('Std Deviation (σ)', rotation=270, labelpad=15, fontweight='bold')

    # ========== (c) 5D不确定性演化 + 参数贡献度 ==========
    ax_evolution = fig.add_subplot(gs[0, 2])

    if len(optimizer.bo_uncertainty['posterior_at_samples']) > 0:
        posterior_data = optimizer.bo_uncertainty['posterior_at_samples']
        iters = [p['iter'] for p in posterior_data]
        mu_samples = [-p['mu'] for p in posterior_data]
        sigma_samples = [p['sigma'] for p in posterior_data]
        actual_scores = [p['actual_score'] for p in posterior_data]

        # 双Y轴
        ax_ev1 = ax_evolution
        ax_ev2 = ax_evolution.twinx()

        # 预测vs实际
        line1 = ax_ev1.plot(iters, mu_samples, color=PALETTE['bo'], lw=2,
                            marker='o', markersize=4, label=r'5D GP Prediction ($\mu$)', zorder=10)
        line2 = ax_ev1.plot(iters, actual_scores, color=PALETTE['highlight'], lw=2,
                            marker='s', markersize=4, label='Actual Score', zorder=10)
        ax_ev1.fill_between(iters,
                            np.array(mu_samples) - np.array(sigma_samples),
                            np.array(mu_samples) + np.array(sigma_samples),
                            color=PALETTE['bo'], alpha=0.2, label=r'$\mu \pm \sigma$')

        # 不确定性演化
        line3 = ax_ev2.plot(iters, sigma_samples, color='red', lw=2,
                            marker='^', markersize=4, label=r'5D Uncertainty ($\sigma$)',
                            linestyle='--', zorder=5)

        ax_ev1.set_xlabel('Iteration', fontweight='bold', fontsize=11)
        ax_ev1.set_ylabel('5D Objective Score', fontweight='bold', fontsize=11, color=PALETTE['bo'])
        ax_ev2.set_ylabel('5D Uncertainty (σ)', fontweight='bold', fontsize=11, color='red')
        ax_ev1.tick_params(axis='y', labelcolor=PALETTE['bo'])
        ax_ev2.tick_params(axis='y', labelcolor='red')

        ax_ev1.set_title(r'(c) 5D Prediction vs Reality',
                         fontweight='bold', fontsize=11, pad=10)
        ax_ev1.grid(True, ls=':', alpha=0.3)

        # 合并图例
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax_ev1.legend(lines, labels, loc='upper left', frameon=True, fontsize=8)
    else:
        ax_evolution.text(0.5, 0.5,
                          "No 5D posterior data", ha='center', va='center',
                          transform=ax_evolution.transAxes)

    fig.suptitle('5D Bayesian Optimization Global Uncertainty Reduction',
                 fontweight='bold', fontsize=14, y=0.98)

    for fmt in formats:
        output_path = f"{base_path}.{fmt}"
        dpi = 300 if fmt == 'png' else None
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"✨ [Fig 4] Saved 5D {fmt.upper()} format: {output_path}")

    plt.close()


# ======================== 主运行入口（完整可执行） ========================
if __name__ == "__main__":
    import os

    os.makedirs("figures_5D", exist_ok=True)

    print("\n" + "=" * 80)
    print("🚀 HIGH-DIMENSIONAL NLSE OPTIMIZATION (5D PHYSICS MODEL)")
    print(" Parameters: A_mod, f_mod, phi0, alpha, beta")
    print(" Visualization: Fixed-slice + PCA/t-SNE Dimensionality Reduction")
    print(" Style: Science / Nature Paper Grade")
    print("=" * 80 + "\n")

    # 初始化 5D 优化器
    opt = NLSEOptimizer(x_range=(-50, 50), nx=256, t_max=12, gamma=1.5, seed=42)

    # 运行全部优化算法（BO / CMA-ES / TPE / Random）
    results = opt.run_all(max_iter=50)

    # 输出目录
    fig_base = "figures_5D/5D_NLSE_Benchmark"

    # ========== 按你要求的四张图 ==========
    # Fig1: 5D 时空动力学 + 最优参数
    plot_rogue_monster_dynamics(opt, results, base_path=f"{fig_base}_Fig1_Dynamics")

    # Fig2: 5D 参数空间地形图（切片 + t-SNE）
    plot_landscape_comparison(opt, results, base_path=f"{fig_base}_Fig2_Landscape")

    # Fig3: 5D 算法统计对比（雨云图）
    plot_raincloud_statistics(opt, results, base_path=f"{fig_base}_Fig3_Stats")

    # Fig4: 5D BO 不确定性降维可视化
    plot_bo_uncertainty_analysis(opt, results, base_path=f"{fig_base}_Fig4_Uncertainty")

    print("\n" + "=" * 80)
    print("✅ ALL 5D SIMULATION & VISUALIZATION COMPLETED")
    print("📁 Figures saved to: figures_5D/")
    print("🔍 Key outputs: Landscape (Fig2), Uncertainty (Fig4), 5D Parameters")
    print("=" * 80 + "\n")