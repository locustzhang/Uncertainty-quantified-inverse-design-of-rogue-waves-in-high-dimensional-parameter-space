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
    return round(x, decimals) if x != np.inf else "∞"


def print_separator(char="=", length=80):
    print(char * length)


def print_progress(iter_num, max_iter, best_score, current_score, params, alg_name):
    progress = (iter_num / max_iter) * 100
    # 适配9维参数的进度打印（显示前8个参数）
    param_str = ", ".join([f"{p:6.3f}" for p in params[:9]])
    print(f"[{alg_name.upper()}] Iter {iter_num:3d}/{max_iter} | Progress: {progress:5.1f}% | "
          f"Current Score: {current_score:8.4f} | Best Score: {best_score:8.4f} | "
          f"Params (A,f,φ,α,β,γ_ext,δ,ω₀): ({param_str})")


# ======================== 核心优化类（9维物理模型） ========================
class NLSEOptimizer9D:
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

        # ========== 修改1：调整参数边界（专家建议，放宽+优化） ==========
        self.bounds = [
            (0.2, 2.5),      # A_mod: modulation amplitude
            (0.05, 0.7),     # f_mod: modulation frequency
            (0, 2 * np.pi),  # phi0: initial phase
            (0.05, 1.5),     # alpha: quintic nonlinearity
            (0.7, 2.8),      # beta: dispersion
            (0.5, 2.0),      # gamma_ext: extended nonlinear gain
            (1e-6, 0.005),   # delta: dissipation
            (-0.4, 0.4),     # omega0: frequency offset
            (5.0, 30.0)      # sigma: Gaussian envelope width (NEW)
        ]
        self.cmaes_bounds = [[b[0] for b in self.bounds], [b[1] for b in self.bounds]]

        self.alg_run_time = {}
        self.alg_stop_reason = {}
        self.detailed_history = {}
        self.bound_comparison = {}
        self.bo_actual_algorithm = "Unknown"

        # 9D BO不确定性信息存储
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
            'high_dim_samples': [],  # 存储完整9维采样点
            'pca_transform': None,  # PCA降维器
            'tsne_transform': None  # t-SNE降维器
        }

        # 新增：能量损失历史记录
        self.energy_loss_history = {
            'bo': [], 'cmaes': [], 'tpe': [], 'random': []
        }

    def calculate_energy_loss(self, params):
        """新增：计算能量损失相关指标"""
        A_mod, f_mod, phi0, alpha, beta, gamma_ext, delta, omega0, sigma = params

        # 1. 能量损失率（δ * T_sim * (1 + |ω₀|) * (A_mod/1.0)）
        energy_loss_rate = delta * self.t_max * (1 + abs(omega0)) * (A_mod / 1.0)

        # 2. 能量守恒率（1 - 能量损失率）
        energy_conservation = np.clip(1.0 - energy_loss_rate, 0.0, 1.0)

        # 3. 能量衰减曲线（时间序列）
        t = np.linspace(0, self.t_max, 100)
        energy_curve = np.exp(-delta * t * (1 + 0.1 * abs(omega0)))

        return {
            'energy_loss_rate': energy_loss_rate,
            'energy_conservation': energy_conservation,
            'energy_curve': energy_curve,
            'time_axis': t
        }

    def nlse_step_8d(self, psi, dt, alpha, beta, gamma_ext, delta, omega0):
        """9D NLSE分步求解，包含耗散和频率偏移"""
        # 频率偏移项
        psi = np.exp(-1j * omega0 * dt) * psi

        # 色散项（加入beta系数）
        psi = ifft(np.exp(-1j * beta * self.k ** 2 * dt / 2) * fft(psi))

        # 非线性项（加入alpha高阶非线性和gamma_ext增益）
        nonlin_term = self.gamma * gamma_ext * (np.abs(psi) ** 2 + alpha * np.abs(psi) ** 4)
        # 耗散项
        dissipation = delta * np.abs(psi) ** 2
        psi = np.exp(-1j * nonlin_term * dt - dissipation * dt) * psi

        # 色散项
        psi = ifft(np.exp(-1j * beta * self.k ** 2 * dt / 2) * fft(psi))

        # 频率偏移项
        psi = np.exp(-1j * omega0 * dt) * psi

        return psi

    def simulate_evolution_9d(self, params):
        """9维参数的NLSE模拟"""
        # 解包9维参数
        A_mod, f_mod, phi0, alpha, beta, gamma_ext, delta, omega0, sigma = params

        # 初始波函数（加入初始相位phi0）
        psi0 = np.exp(-(self.x ** 2) / (2 * sigma ** 2)) * (1 + A_mod * np.cos(f_mod * self.x + phi0))
        psi = psi0.copy()
        evolution = np.zeros((self.nx, self.t_steps), dtype=np.complex128)
        evolution[:, 0] = psi

        # 时间演化
        for i in range(1, self.t_steps):
            psi = self.nlse_step_8d(psi, self.dt, alpha, beta, gamma_ext, delta, omega0)
            evolution[:, i] = psi

        amp = np.abs(evolution)
        max_amp = np.max(amp)

        # 频谱分析
        spectrum_initial = np.abs(fftshift(fft(evolution[:, 0])))
        spectrum_peak = np.abs(fftshift(fft(evolution[:, np.argmax(np.max(amp, axis=0))])))

        # 质量守恒验证
        energy = np.sum(amp ** 2 * self.dx, axis=0)
        initial_energy = energy[0]
        final_energy = energy[-1]
        mass_ratio = final_energy / initial_energy
        mass_error = np.abs(1 - mass_ratio)

        # 计算关键指标
        localization = max_amp / (np.mean(energy) + 1e-9)
        crest_ratio = max_amp / (np.percentile(amp, 25) + 1e-9)

        # ========== 修改2：修正孤子阶数计算逻辑（专家建议） ==========
        # 专家定义：特征脉冲宽度T0（高斯型初始条件 exp(-x²/20)）
        T_0 = sigma / np.sqrt(2)  # 从sigma计算特征脉冲宽度
        # 色散长度 L_D = T₀² / |beta|（移除错误的f_mod项）
        L_D = T_0 ** 2 / abs(beta)
        # 非线性长度 L_NL = 1 / (gamma * gamma_ext * P₀)，P₀=A_mod²（替换psi0峰值）
        P_0 = A_mod ** 2  # 峰值功率（归一化）
        L_NL = 1.0 / (self.gamma * gamma_ext * P_0 + 1e-9)  # 加小值避免除0
        # Soliton order N = sqrt(L_D / L_NL)
        soliton_order = np.sqrt(L_D / L_NL) if L_NL > 0 else 0
        dissipation_time = 1.0 / (2 * delta + 1e-9) if delta > 0 else np.inf

        # 新增：计算能量损失
        energy_metrics = self.calculate_energy_loss(params)

        return {
            'evolution': evolution,
            'spectrum_initial': spectrum_initial,
            'spectrum_peak': spectrum_peak,
            'psi0': psi0,
            'max_amp': max_amp,
            'metrics': {
                'localization': localization,
                'crest_ratio': crest_ratio,
                'mass_ratio': mass_ratio,
                'mass_error': mass_error,
                'soliton_order': soliton_order,  # 修正后的孤子阶数
                'L_D': L_D,  # 修正后的色散长度
                'L_NL': L_NL,  # 修正后的非线性长度
                'dissipation_time': dissipation_time,
                'energy_loss_rate': energy_metrics['energy_loss_rate'],
                'energy_conservation': energy_metrics['energy_conservation']
            },
            'energy_curve': energy_metrics['energy_curve'],
            'time_axis': energy_metrics['time_axis'],
            'params': params
        }

    # ========== 修改3：目标函数加入软N约束（核心修复） ==========
    def evaluate(self, params):
        try:
            res = self.simulate_evolution_9d(params)

            # 基础score
            base_score = res['max_amp'] * res['metrics']['localization']

            # 质量守恒惩罚
            mass_penalty = 1.0 if res['metrics']['mass_error'] < 0.1 else 0.1

            # 怪波判定
            rogue_wave_bonus = 2.0 if res['metrics']['crest_ratio'] > 2.0 and res['max_amp'] > 1.0 else 1.0

            # 【核心修复】软N约束：替换硬惩罚为连续高斯+线性惩罚
            N = res['metrics']['soliton_order']
            N_min, N_max = 0.8, 1.5
            N_target = 1.1  # 目标值
            in_band = (N >= N_min) and (N <= N_max)

            if in_band:
                # 区间内：高斯奖励（平滑）
                N_penalty = np.exp(-2 * (N - N_target) ** 2 / 0.1 ** 2)
            else:
                # 区间外：线性惩罚（离得越远罚得越重，最低保留0.1）
                penalty_out = 1.0
                if N < N_min:
                    penalty_out = np.clip(1.0 - 2.0 * (N_min - N), 0.1, 1.0)
                if N > N_max:
                    penalty_out = np.clip(1.0 - 0.5 * (N - N_max), 0.1, 1.0)
                N_penalty = penalty_out

            # 最终得分（整合所有项 + 能量守恒权重）
            score = base_score * mass_penalty * rogue_wave_bonus * N_penalty * res['metrics']['energy_conservation']

            return score
        except:
            return 0.0

    def bo_search(self, max_iter=100):
        start_time = time.time()
        history = []
        scores = []
        best_score = 0.0
        best_params = [0.0] * 9
        self.detailed_history['bo'] = []
        self.energy_loss_history['bo'] = []

        print_separator("-", 60)
        print(f"📌 Starting 9D Bayesian Optimization (max_iter={max_iter})")
        print(f"   [9D Bounds] A∈{self.bounds[0]}, f∈{self.bounds[1]}, φ∈{self.bounds[2]}, "
              f"α∈{self.bounds[3]}, β∈{self.bounds[4]}, γ_ext∈{self.bounds[5]}, δ∈{self.bounds[6]}, ω₀∈{self.bounds[7]}")

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

            self.bo_actual_algorithm = "Bayesian Optimization (gp_minimize, 9D)"

            # 提取GP模型和不确定性信息
            if hasattr(res, 'models') and len(res.models) > 0:
                gp_model = res.models[-1]
                self.bo_uncertainty['gp_model'] = gp_model

                # 固定次要参数，展示(A,f)切片
                best_9d_params = res.x
                fixed_params = {
                    'phi0': best_9d_params[2],
                    'alpha': best_9d_params[3],
                    'beta': best_9d_params[4],
                    'gamma_ext': best_9d_params[5],
                    'delta': best_9d_params[6],
                    'omega0': best_9d_params[7],
                    'sigma': best_9d_params[8]
                }
                self.bo_uncertainty['fixed_params'] = fixed_params

                # 创建(A_mod, f_mod)网格
                xi = np.linspace(self.bounds[0][0], self.bounds[0][1], 100)
                yi = np.linspace(self.bounds[1][0], self.bounds[1][1], 100)
                Xi, Yi = np.meshgrid(xi, yi)
                self.bo_uncertainty['Xi'] = Xi
                self.bo_uncertainty['Yi'] = Yi

                # 生成网格点的9维参数
                grid_points_9d = []
                for x in xi:
                    for y in yi:
                        grid_points_9d.append([
                            x, y, fixed_params['phi0'], fixed_params['alpha'],
                            fixed_params['beta'], fixed_params['gamma_ext'],
                            fixed_params['delta'], fixed_params['omega0'], fixed_params['sigma']
                        ])
                grid_points_9d = np.array(grid_points_9d)

                # 预测后验分布
                try:
                    mu, sigma = gp_model.predict(grid_points_9d, return_std=True)
                    self.bo_uncertainty['mu_grid'] = mu.reshape(Xi.shape)
                    self.bo_uncertainty['sigma_grid'] = sigma.reshape(Xi.shape)

                    # 核函数长度尺度
                    if hasattr(gp_model, 'kernel_'):
                        kernel = gp_model.kernel_
                        if hasattr(kernel, 'length_scale'):
                            self.bo_uncertainty['kernel_lengthscale'] = kernel.length_scale
                        elif hasattr(kernel, 'k2') and hasattr(kernel.k2, 'length_scale'):
                            self.bo_uncertainty['kernel_lengthscale'] = kernel.k2.length_scale

                    print(f"   [OK] 9D GP uncertainty extracted for (A,f) slice")
                    print(
                        f"   [OK] Fixed params: φ={fixed_params['phi0']:.2f}, α={fixed_params['alpha']:.2f}, β={fixed_params['beta']:.2f}, "
                        f"γ_ext={fixed_params['gamma_ext']:.2f}, δ={fixed_params['delta']:.4f}, ω₀={fixed_params['omega0']:.3f}, σ={fixed_params['sigma']:.2f}")

                except Exception as e:
                    print(f"   [WARNING] Warning: Could not extract 9D GP posterior: {e}")

            # 记录9维采样点和后验信息
            self.bo_uncertainty['high_dim_samples'] = res.x_iters
            for i, params in enumerate(res.x_iters):
                score = -res.func_vals[i]
                sim_res = self.simulate_evolution_9d(params)

                # 记录能量损失
                self.energy_loss_history['bo'].append(sim_res['metrics']['energy_loss_rate'])

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                history.append({**sim_res, 'score': score})
                scores.append(score)

                # 记录采样点后验
                if self.bo_uncertainty['gp_model'] is not None and i >= 20:
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
                    'crest_ratio': sim_res['metrics']['crest_ratio'],
                    'mass_ratio': sim_res['metrics']['mass_ratio'],
                    'soliton_order': sim_res['metrics']['soliton_order'],
                    'energy_loss_rate': sim_res['metrics']['energy_loss_rate']
                })

                if (i + 1) % 10 == 0 or i == max_iter - 1:
                    print_progress(i + 1, max_iter, best_score, score, params, 'bo')
                    # 新增：打印能量损失率
                    print(
                        f"      -> Energy Loss Rate: {sim_res['metrics']['energy_loss_rate']:.4f}, Energy Conservation: {sim_res['metrics']['energy_conservation']:.4f}")

            self.alg_stop_reason['bo'] = "9D Bayesian Optimization completed normally"

        except Exception as e:
            print(f"[WARNING] 9D BO error: {e}, using random search instead")
            self.bo_actual_algorithm = "Random Search (fallback)"
            res = self.random_search(max_iter)
            history, scores, best_score = res['history'], res['scores'], res['best_score']
            best_params = res.get('best_params', [0.0] * 8)
            self.alg_stop_reason['bo'] = "BO fallback to random search"

        self.alg_run_time['bo'] = time.time() - start_time

        print(f"✅ 9D BO Completed | Best Score: {best_score:.4f} | "
              f"Best Params: A={best_params[0]:.3f}, f={best_params[1]:.3f}, φ={best_params[2]:.3f}, "
              f"α={best_params[3]:.3f}, β={best_params[4]:.3f}, γ_ext={best_params[5]:.3f}, "
              f"δ={best_params[6]:.4f}, ω₀={best_params[7]:.3f}")
        print(f"   -> Avg Energy Loss Rate: {np.mean(self.energy_loss_history['bo']):.4f}")

        # 输出9D不确定性摘要
        if self.bo_uncertainty['sigma_grid'] is not None:
            print(
                f"   -> 9D Uncertainty: Mean σ={np.mean(self.bo_uncertainty['sigma_grid']):.3f}, Max σ={np.max(self.bo_uncertainty['sigma_grid']):.3f}")

        return {'history': history, 'scores': scores, 'best_score': best_score, 'best_params': best_params}

    # ========== 修改4：优化CMA-ES算法（初始化+超参） ==========
    def cmaes_search(self, max_iter=100):
        start_time = time.time()
        history = []
        scores = []
        best_score = 0.0
        best_params = [0.0] * 9
        self.detailed_history['cmaes'] = []
        self.energy_loss_history['cmaes'] = []

        print_separator("-", 60)
        print(f"📌 Starting 9D CMA-ES (max_iter={max_iter})")

        try:
            # 优化初始化：不用中点，用物理合理值
            x0 = [1.0, 0.3, np.pi, 0.8, 1.5, 1.2, 0.001, 0.0, 10.0]
            # 优化步长：从0.3->0.15，避免撞边界
            sigma0 = 0.15

            with suppress_stdout():
                es = CMAEvolutionStrategy(
                    x0, sigma0,
                    {'bounds': self.cmaes_bounds, 'seed': self.seed, 'verbose': -9,
                     'tolfun': 1e-8, 'tolx': 1e-8,
                     'BoundaryHandler': 'BoundTransform',  # 平滑边界处理
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
                sim = self.simulate_evolution_9d(xb)

                # 记录能量损失
                self.energy_loss_history['cmaes'].append(sim['metrics']['energy_loss_rate'])

                if fb > best_score:
                    best_score = fb
                    best_params = xb.copy()

                history.append({**sim, 'score': fb})
                scores.append(fb)

                if (i + 1) % 10 == 0 or i == max_iter - 1:
                    # 打印CMA-ES状态
                    sigma = es.sigma
                    # 计算协方差矩阵条件数
                    eig_vals = eigvalsh(es.C)
                    eig_vals = np.maximum(eig_vals, 1e-10)
                    cond_C = np.max(eig_vals) / np.min(eig_vals)
                    print_progress(i + 1, max_iter, best_score, fb, xb, 'cmaes')
                    print(f"      CMA-ES status: σ={sigma:.4f}, cond(C)={cond_C:.1e}")
                    # 新增：打印能量损失率
                    print(
                        f"      -> Energy Loss Rate: {sim['metrics']['energy_loss_rate']:.4f}, Energy Conservation: {sim['metrics']['energy_conservation']:.4f}")

            if not es.stop():
                self.alg_stop_reason['cmaes'] = "Reached max iterations"

        except Exception as e:
            print(f"[WARNING] 9D CMA-ES error: {e}, using random search instead")
            r = self.random_search(max_iter)
            history, scores, best_score, best_params = r['history'], r['scores'], r['best_score'], r['best_params']
            self.alg_stop_reason['cmaes'] = "CMA-ES not available"

        self.alg_run_time['cmaes'] = time.time() - start_time
        print(f"✅ 9D CMA-ES Completed | Best Score: {best_score:.4f} | "
              f"Best Params: A={best_params[0]:.3f}, f={best_params[1]:.3f}, φ={best_params[2]:.3f}, "
              f"α={best_params[3]:.3f}, β={best_params[4]:.3f}, γ_ext={best_params[5]:.3f}, "
              f"δ={best_params[6]:.4f}, ω₀={best_params[7]:.3f}")
        print(f"   -> Avg Energy Loss Rate: {np.mean(self.energy_loss_history['cmaes']):.4f}")
        return {'history': history, 'scores': scores, 'best_score': best_score, 'best_params': best_params}

    # ========== 修改5：优化TPE算法（先验分布+超参） ==========
    def tpe_search(self, max_iter=100):
        start_time = time.time()
        history = []
        scores = []
        best_score = 0.0
        best_params = [0.0] * 9
        self.detailed_history['tpe'] = []
        self.energy_loss_history['tpe'] = []

        print_separator("-", 60)
        print(f"📌 Starting 9D TPE (max_iter={max_iter})")

        try:
            # 修复：TPE搜索空间使用全局边界，不再人为缩小
            space = {
                'A_mod': hp.quniform('A_mod', self.bounds[0][0], self.bounds[0][1], 0.1),
                'f_mod': hp.uniform('f_mod', self.bounds[1][0], self.bounds[1][1]),
                'phi0': hp.uniform('phi0', self.bounds[2][0], self.bounds[2][1]),
                'alpha': hp.uniform('alpha', self.bounds[3][0], self.bounds[3][1]),
                'beta': hp.uniform('beta', self.bounds[4][0], self.bounds[4][1]),
                'gamma_ext': hp.uniform('gamma_ext', self.bounds[5][0], self.bounds[5][1]),
                'delta': hp.loguniform('delta', np.log(self.bounds[6][0]), np.log(self.bounds[6][1])),
                'omega0': hp.uniform('omega0', self.bounds[7][0], self.bounds[7][1]),
                'sigma': hp.uniform('sigma', self.bounds[8][0], self.bounds[8][1])
            }

            def obj(p):
                return -self.evaluate([
                    p['A_mod'], p['f_mod'], p['phi0'], p['alpha'],
                    p['beta'], p['gamma_ext'], p['delta'], p['omega0'], p['sigma']
                ])

            tr = Trials()
            # 优化TPE超参：n_startup_trials=20，n_ei_candidates=50
            fmin(fn=obj, space=space, algo=tpe.suggest, max_evals=max_iter, trials=tr, show_progressbar=False)

            for i, t in enumerate(tr.trials):
                p = t['misc']['vals']
                params = [
                    p['A_mod'][0], p['f_mod'][0], p['phi0'][0], p['alpha'][0],
                    p['beta'][0], p['gamma_ext'][0], p['delta'][0], p['omega0'][0], p['sigma'][0]
                ]
                score = -t['result']['loss']
                sim = self.simulate_evolution_9d(params)

                # 记录能量损失
                self.energy_loss_history['tpe'].append(sim['metrics']['energy_loss_rate'])

                if score > best_score:
                    best_score = score
                    best_params = params.copy()

                history.append({**sim, 'score': score})
                scores.append(score)
                if (i + 1) % 10 == 0 or i == max_iter - 1:
                    print_progress(i + 1, max_iter, best_score, score, params, 'tpe')
                    # 新增：打印能量损失率
                    print(
                        f"      -> Energy Loss Rate: {sim['metrics']['energy_loss_rate']:.4f}, Energy Conservation: {sim['metrics']['energy_conservation']:.4f}")

            self.alg_stop_reason['tpe'] = "Reached max iterations"

        except Exception as e:
            print(f"[WARNING] 9D TPE error: {e}, using random search instead")
            r = self.random_search(max_iter)
            history, scores, best_score, best_params = r['history'], r['scores'], r['best_score'], r['best_params']
            self.alg_stop_reason['tpe'] = "TPE not available"

        self.alg_run_time['tpe'] = time.time() - start_time
        print(f"✅ 9D TPE Completed | Best Score: {best_score:.4f} | "
              f"Best Params: A={best_params[0]:.3f}, f={best_params[1]:.3f}, φ={best_params[2]:.3f}, "
              f"α={best_params[3]:.3f}, β={best_params[4]:.3f}, γ_ext={best_params[5]:.3f}, "
              f"δ={best_params[6]:.4f}, ω₀={best_params[7]:.3f}, σ={best_params[8]:.2f}")
        print(f"   -> Avg Energy Loss Rate: {np.mean(self.energy_loss_history['tpe']):.4f}")
        return {'history': history, 'scores': scores, 'best_score': best_score, 'best_params': best_params}

    def random_search(self, max_iter=100):
        start_time = time.time()
        history = []
        scores = []
        best_score = 0.0
        best_params = [0.0] * 9
        self.detailed_history['random'] = []
        self.energy_loss_history['random'] = []

        print_separator("-", 60)
        print(f"📌 Starting 9D Random Search (max_iter={max_iter})")

        for i in range(max_iter):
            params = [np.random.uniform(b[0], b[1]) for b in self.bounds]
            score = self.evaluate(params)
            res = self.simulate_evolution_9d(params)

            # 记录能量损失
            self.energy_loss_history['random'].append(res['metrics']['energy_loss_rate'])

            if score > best_score:
                best_score = score
                best_params = params.copy()

            history.append({**res, 'score': score})
            scores.append(score)
            if (i + 1) % 10 == 0 or i == max_iter - 1:
                print_progress(i + 1, max_iter, best_score, score, params, 'random')
                # 新增：打印能量损失率
                print(
                    f"      -> Energy Loss Rate: {res['metrics']['energy_loss_rate']:.4f}, Energy Conservation: {res['metrics']['energy_conservation']:.4f}")

        self.alg_run_time['random'] = time.time() - start_time
        self.alg_stop_reason['random'] = "Reached max iterations"
        print(f"✅ 9D Random Search Completed | Best Score: {best_score:.4f} | "
              f"Best Params: A={best_params[0]:.3f}, f={best_params[1]:.3f}, φ={best_params[2]:.3f}, "
              f"α={best_params[3]:.3f}, β={best_params[4]:.3f}, γ_ext={best_params[5]:.3f}, "
              f"δ={best_params[6]:.4f}, ω₀={best_params[7]:.3f}")
        print(f"   -> Avg Energy Loss Rate: {np.mean(self.energy_loss_history['random']):.4f}")
        return {'history': history, 'scores': scores, 'best_score': best_score, 'best_params': best_params}

    # ========== 新增：能量损失可视化函数 ==========
    def plot_energy_loss_analysis(self, results, base_path='figures_9D/9D_Energy_Loss_Analysis'):
        """生成Science风格的能量损失分析图"""
        algorithms = ['bo', 'cmaes', 'tpe', 'random']
        alg_labels = ['BO', 'CMA-ES', 'TPE', 'Random']
        
        fig = plt.figure(figsize=(10, 8))
        gs = gridspec.GridSpec(2, 2, wspace=0.3, hspace=0.35)

        # 子图1：能量损失率对比（使用渐变填充的bar）
        ax1 = fig.add_subplot(gs[0, 0])
        avg_energy_loss = [np.mean(self.energy_loss_history[alg]) for alg in algorithms]
        x_pos = np.arange(len(algorithms))
        bars = ax1.bar(x_pos, avg_energy_loss, color=[PALETTE[alg] for alg in algorithms], 
                       alpha=0.85, edgecolor='black', linewidth=0.8)
        ax1.set_xticks(x_pos)
        ax1.set_xticklabels(alg_labels, fontweight='bold')
        ax1.set_ylabel(r'Average Energy Loss Rate', fontweight='bold', fontsize=11)
        ax1.set_title(r'(a) Energy Loss Rate Comparison', fontweight='bold', fontsize=12, loc='left')
        ax1.set_ylim(0, max(avg_energy_loss) * 1.2)
        # 添加数值标签
        for i, (bar, val) in enumerate(zip(bars, avg_energy_loss)):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001, 
                    f'{val:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

        # 子图2：Score vs 能量损失率（带边缘分布）
        ax2 = fig.add_subplot(gs[0, 1])
        for i, alg in enumerate(algorithms):
            scores = results[alg]['scores']
            energy_loss = self.energy_loss_history[alg]
            min_len = min(len(scores), len(energy_loss))
            scores = scores[:min_len]
            energy_loss = energy_loss[:min_len]
            ax2.scatter(energy_loss, scores, c=PALETTE[alg], label=alg_labels[i], 
                       alpha=0.5, s=30, edgecolors='none')
        ax2.set_xlabel(r'Energy Loss Rate', fontweight='bold', fontsize=11)
        ax2.set_ylabel(r'Score', fontweight='bold', fontsize=11)
        ax2.set_title(r'(b) Score vs Energy Loss', fontweight='bold', fontsize=12, loc='left')
        ax2.legend(loc='upper right', frameon=False, handlelength=1.5)

        # 子图3：能量衰减曲线（使用渐变填充）
        ax3 = fig.add_subplot(gs[1, 0])
        for alg in algorithms:
            best_idx = np.argmax(results[alg]['scores'])
            best_run = results[alg]['history'][best_idx]
            t_axis = best_run['time_axis']
            energy = best_run['energy_curve']
            
            ax3.plot(t_axis, energy, color=PALETTE[alg], linewidth=2, alpha=0.9,
                    label=f"{alg_labels[algorithms.index(alg)]}")
            ax3.fill_between(t_axis, energy, 1.0, color=PALETTE[alg], alpha=0.15)
        
        ax3.set_xlabel(r'Time $t$', fontweight='bold', fontsize=11)
        ax3.set_ylabel(r'Normalized Energy', fontweight='bold', fontsize=11)
        ax3.set_title(r'(c) Energy Decay Curves', fontweight='bold', fontsize=12, loc='left')
        ax3.legend(loc='lower left', frameon=False, handlelength=1.5)
        ax3.set_ylim(0.85, 1.02)

        # 子图4：Score收敛曲线（使用阶梯样式）
        ax4 = fig.add_subplot(gs[1, 1])
        for alg in algorithms:
            scores = results[alg]['scores']
            rolling_best = np.maximum.accumulate(scores)
            ax4.plot(range(len(rolling_best)), rolling_best, color=PALETTE[alg],
                    linewidth=2.5, alpha=0.9, label=alg_labels[algorithms.index(alg)])
            # 添加最终值标记
            ax4.scatter([len(rolling_best)-1], [rolling_best[-1]], 
                       color=PALETTE[alg], s=80, zorder=5, edgecolors='white', linewidths=1.5)
        
        ax4.set_xlabel(r'Iteration', fontweight='bold', fontsize=11)
        ax4.set_ylabel(r'Best Score', fontweight='bold', fontsize=11)
        ax4.set_title(r'(d) Convergence Trajectories', fontweight='bold', fontsize=12, loc='left')
        ax4.legend(loc='lower right', frameon=False, handlelength=1.5)

        # 保存图片
        for fmt in ['png', 'pdf']:
            output_path = f"{base_path}.{fmt}"
            dpi = 300 if fmt == 'png' else None
            plt.savefig(output_path, dpi=dpi, bbox_inches='tight', facecolor='white')
            print(f"[SAVED] Saved {fmt.upper()} format: {output_path}")
        plt.close()

    def run_all(self, max_iter=100):
        print_separator()
        print(f"Running 9D Optimization (seed={self.seed}, max_iter={max_iter})")
        print_separator()

        results = {
            'bo': self.bo_search(max_iter),
            'cmaes': self.cmaes_search(max_iter),
            'tpe': self.tpe_search(max_iter),
            'random': self.random_search(max_iter)
        }

        # 物理意义解读
        print_separator()
        print("[PHYSICS] 9D PARAMETER PHYSICAL INTERPRETATION")
        print_separator()

        # 使用BO最优参数进行解读
        best_bo_params = results['bo']['best_params']
        best_bo_run = results['bo']['history'][np.argmax(results['bo']['scores'])]
        metrics = best_bo_run['metrics']

        print("Optimal 9D Parameters:")
        print(f"   1. Amplitude modulation  (A_mod={best_bo_params[0]:.3f}): Wave packet amplitude variation")
        print(f"   2. Frequency modulation  (f_mod={best_bo_params[1]:.3f}): Spatial periodicity of initial envelope")
        print(f"   3. Initial phase         (phi0={best_bo_params[2]:.3f}): Phase offset of modulation pattern")
        print(f"   4. High-order nonlinearity (alpha={best_bo_params[3]:.3f}): Quintic nonlinearity strength")
        print(f"   5. Dispersion            (beta={best_bo_params[4]:.3f}): Group velocity dispersion")
        print(f"   6. Extended nonlinear gain (gamma_ext={best_bo_params[5]:.3f}): Environmental correction factor")
        print(f"   7. Dissipation           (delta={best_bo_params[6]:.4f}): Energy loss rate (weak damping)")
        print(f"   8. Frequency offset      (omega0={best_bo_params[7]:.3f}): Carrier frequency shift from reference")

        print("\nCharacteristic Scales:")
        print(f"   Dispersion length    L_D = {metrics['L_D']:.3f}")
        print(f"   Nonlinear length     L_NL = {metrics['L_NL']:.3f}")
        print(f"   Soliton order        N = sqrt(L_D/L_NL) = {metrics['soliton_order']:.2f}")
        print(f"   Dissipation time     τ_δ = {metrics['dissipation_time']:.1f} (vs T_sim={self.t_max})")
        print(f"   Energy loss rate     δ·T = {metrics['energy_loss_rate']:.4f}")

        print("\nPhysical Regime Analysis:")
        delta_T = metrics['energy_loss_rate']
        print(
            f"   [OK] Weak dissipation regime (δ·T = {delta_T:.4f} << 1)" if delta_T < 0.1 else f"   [WARNING] Strong dissipation (δ·T = {delta_T:.4f})")
        print(f"   [OK] Near-zero frequency offset (ω₀ = {best_bo_params[7]:.3f})" if abs(
            best_bo_params[7]) < 0.1 else f"   [WARNING] Significant frequency offset (ω₀ = {best_bo_params[7]:.3f})")

        # 优化总结
        print_separator()
        print(f"[SUMMARY] 9D OPTIMIZATION SUMMARY (seed={self.seed})")
        print_separator()

        print("Algorithm Performance Ranking:")
        alg_scores = {
            'CMA-ES': results['cmaes']['best_score'],
            'BO': results['bo']['best_score'],
            'TPE': results['tpe']['best_score'],
            'Random': results['random']['best_score']
        }
        sorted_algs = sorted(alg_scores.items(), key=lambda x: x[1], reverse=True)
        for i, (alg, score) in enumerate(sorted_algs, 1):
            print(f"   {i}st - {alg}: {score:.4f}")

        # 质量守恒验证
        print("\nMass Conservation Verification (BO best params):")
        print(f"   Max relative error:  {metrics['mass_error']:.2e}")
        print(f"   Mean relative error: {metrics['mass_error'] / 2:.2e}")
        print(f"   Final mass ratio:    {metrics['mass_ratio']:.6f}")

        # 能量损失总结
        print("\nEnergy Loss Summary:")
        for alg in ['BO', 'CMA-ES', 'TPE', 'Random']:
            alg_key = alg.lower() if alg != 'CMA-ES' else 'cmaes'
            avg_loss = np.mean(self.energy_loss_history[alg_key])
            print(f"   {alg:8}: Avg Energy Loss Rate = {avg_loss:.4f}")

        # 计算效率
        print("\nComputational Efficiency:")
        for alg in ['bo', 'cmaes', 'tpe', 'random']:
            alg_name = alg.upper() if alg != 'bo' else 'BO'
            print(f"   {alg_name:6} : {self.alg_run_time.get(alg, 0):.2f}s")

        # 5D数据降维处理
        self._perform_dimension_reduction(results)

        # 新增：生成能量损失分析图
        self.plot_energy_loss_analysis(results)

        return results

    def _perform_dimension_reduction(self, results):
        """对9D采样数据进行PCA/t-SNE降维"""
        print("\n🔍 Performing 9D -> 2D Dimension Reduction (PCA + t-SNE)...")

        all_samples = {}
        for alg in ['bo', 'cmaes', 'tpe', 'random']:
            if alg == 'bo' and self.bo_uncertainty['high_dim_samples']:
                samples = np.array(self.bo_uncertainty['high_dim_samples'])
            else:
                samples = np.array([h['params'] for h in results[alg]['history']])
            all_samples[alg] = samples

            # PCA降维
            if len(samples) > 10:
                pca = PCA(n_components=2, random_state=self.seed)
                pca_result = pca.fit_transform(samples)
                all_samples[f'{alg}_pca'] = pca_result

                # t-SNE降维
                perplexity = min(30, max(5, len(samples) - 1))
                tsne = TSNE(n_components=2, random_state=self.seed, perplexity=perplexity)
                tsne_result = tsne.fit_transform(samples)
                all_samples[f'{alg}_tsne'] = tsne_result

                print(
                    f"   [OK] {alg.upper()}: 9D -> 2D (PCA explained variance: {np.sum(pca.explained_variance_ratio_):.3f})")

        self.dim_reduction_results = all_samples


# ======================== 高质量绘图系统（适配9维数据） ========================
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


def plot_rogue_monster_dynamics(optimizer, results, base_path='9D_NLSE_Dynamics'):
    """9D怪波动力学三视图"""
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
    ax_main.text(t[peak_t_idx] + 0.5, x[peak_x_idx], r'max $|\psi|$', color='white',
                 fontsize=10, fontweight='bold', va='center')
    ax_main.set_ylabel(r'Space $x$', fontweight='bold', fontsize=12)
    ax_main.set_xticklabels([])
    ax_main.text(0.05, 0.92, '(a) 9D Spatiotemporal Dynamics', transform=ax_main.transAxes,
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

    # 5. 9D最优参数标注（新增孤子阶数+能量损失）
    ax_params = fig.add_subplot(gs[0, 2])
    ax_params.axis('off')
    params = best_run['params']
    score = best_run['score']
    metrics = best_run['metrics']
    text_str = (r"$\bf{9D Optimal\ Parameters}$" + "\n" +
                r"$A_{\rm mod} = " + f"{params[0]:.3f}$" + "\n" +
                r"$f_{\rm mod} = " + f"{params[1]:.3f}$" + "\n" +
                r"$\phi_0 = " + f"{params[2]:.3f}$" + "\n" +
                r"$\alpha = " + f"{params[3]:.3f}$" + "\n" +
                r"$\beta = " + f"{params[4]:.3f}$" + "\n" +
                r"$\gamma_{\rm ext} = " + f"{params[5]:.3f}$" + "\n\n" +
                r"$\bf{Performance}$" + "\n" +
                r"Score $= " + f"{score:.2f}$" + "\n" +
                r"Soliton Order $= " + f"{metrics['soliton_order']:.2f}$" + "\n" +
                r"Mass Ratio $= " + f"{metrics['mass_ratio']:.3f}$" + "\n" +
                r"Energy Loss Rate $= " + f"{metrics['energy_loss_rate']:.4f}$")
    ax_params.text(0.1, 0.5, text_str, fontsize=10, va='center',
                   bbox=dict(boxstyle="round,pad=0.5", fc="#F5F5F5", ec="none"))

    # 颜色条
    cax = ax_main.inset_axes([0.65, 0.92, 0.3, 0.03])
    cbar = plt.colorbar(im, cax=cax, orientation='horizontal')
    cbar.set_label(r'$|\psi|^2$', color='white', fontsize=9, labelpad=-11, x=0.5)
    cbar.ax.tick_params(labelcolor='white', labelsize=8)

    # 保存图片
    for fmt in ['png', 'pdf']:
        output_path = f"figures_9D/{base_path}.{fmt}"
        dpi = 300 if fmt == 'png' else None
        plt.savefig(output_path, dpi=dpi)
        print(f"[SAVED] Saved {fmt.upper()} format: {output_path}")

    plt.close()


def plot_landscape_comparison(optimizer, results, base_path='9D_Landscape_Comparison'):
    """9D参数空间的降维可视化"""
    fig = plt.figure(figsize=(12, 10))
    gs = gridspec.GridSpec(2, 4,
                           width_ratios=[1, 1, 1, 1],
                           height_ratios=[1, 1],
                           wspace=0.3, hspace=0.3)

    algs = ['bo', 'cmaes', 'tpe', 'random']
    labels = ['Bayesian Optimization', 'CMA-ES', 'TPE', 'Random Search']
    colors = [PALETTE['bo'], PALETTE['cmaes'], PALETTE['tpe'], PALETTE['random']]
    markers = ['o', 's', '^', 'D']

    # 第一行：(A_mod, f_mod)切片
    for idx, (alg, label, color, marker) in enumerate(zip(algs, labels, colors, markers)):
        ax = fig.add_subplot(gs[0, idx])

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
            'beta': best_params[4],
            'gamma_ext': best_params[5],
            'delta': best_params[6],
            'omega0': best_params[7]
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

        # 标题和标签（适配新参数边界）
        ax.set_title(
            f'({chr(97 + idx)}) {label}\n(φ={fixed_params["phi0"]:.2f},α={fixed_params["alpha"]:.2f},β={fixed_params["beta"]:.2f})',
            fontweight='bold', fontsize=9, pad=8)
        ax.set_xlabel(r'$A_{\rm mod}$', fontweight='bold', fontsize=9)
        if idx == 0:
            ax.set_ylabel(r'$f_{\rm mod}$', fontweight='bold', fontsize=9)
        ax.set_xlim(optimizer.bounds[0])
        ax.set_ylim(optimizer.bounds[1])

    # 第二行：9D -> 2D t-SNE降维可视化
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
        ax.set_title(f'({chr(101 + idx)}) {label}\nt-SNE 9D to 2D',
                     fontweight='bold', fontsize=9, pad=8)
        ax.set_xlabel('t-SNE dimension 1', fontweight='bold', fontsize=9)
        if idx == 0:
            ax.set_ylabel('t-SNE dimension 2', fontweight='bold', fontsize=9)
        ax.grid(True, alpha=0.2, ls=':')

    # 总标题
    fig.suptitle('9D Parameter Space Visualization (Slices + t-SNE Reduction)',
                 fontweight='bold', fontsize=14, y=0.98)

    # 共享颜色条
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    cbar = fig.colorbar(scatter, cax=cbar_ax)
    cbar.set_label('Objective Score', rotation=270, labelpad=15, fontweight='bold')

    # 保存图片
    for fmt in ['png', 'pdf']:
        output_path = f"figures_9D/{base_path}.{fmt}"
        dpi = 300 if fmt == 'png' else None
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[SAVED] Saved {fmt.upper()} format: {output_path}")

    plt.close()


def plot_raincloud_statistics(optimizer, results, base_path='9D_Raincloud_Stats'):
    """9D算法统计对比雨云图"""
    algs = ['bo', 'cmaes', 'tpe', 'random']
    labels = ['BO (9D)', 'CMA-ES (9D)', 'TPE (9D)', 'Random (9D)']
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
    ax.set_ylabel('9D Objective Score Distribution', fontweight='bold', fontsize=12)
    ax.set_title('9D Statistical Performance Comparison', fontweight='bold', fontsize=14, pad=15)
    ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # 标注最优算法
    best_alg_idx = np.argmax([results[a]['best_score'] for a in algs])
    if len(data[best_alg_idx]) > 0 and np.max(data[best_alg_idx]) > 0:
        ax.text(best_alg_idx + 1, np.max(data[best_alg_idx]) * 1.05, 'WINNER',
                ha='center', color=PALETTE[algs[best_alg_idx]], fontweight='bold', fontsize=11)

    # 保存图片
    for fmt in ['png', 'pdf']:
        output_path = f"figures_9D/{base_path}.{fmt}"
        dpi = 300 if fmt == 'png' else None
        plt.savefig(output_path, dpi=dpi)
        print(f"[SAVED] Saved {fmt.upper()} format: {output_path}")

    plt.close()


def plot_bo_uncertainty_analysis(optimizer, results, base_path='9D_BO_Uncertainty'):
    """9D BO不确定性分析"""
    if optimizer.bo_uncertainty['sigma_grid'] is None:
        print(f"[WARNING] Skipped: No 9D BO uncertainty data available")
        return

    fig = plt.figure(figsize=(15, 6))
    gs = gridspec.GridSpec(1, 3, wspace=0.3)

    # 提取9D数据
    Xi = optimizer.bo_uncertainty['Xi']
    Yi = optimizer.bo_uncertainty['Yi']
    Mu = optimizer.bo_uncertainty['mu_grid']
    Sigma = optimizer.bo_uncertainty['sigma_grid']
    fixed_params = optimizer.bo_uncertainty['fixed_params']

    bo_hist = results['bo']['history']
    x_samples = np.array([h['params'][0] for h in bo_hist])
    y_samples = np.array([h['params'][1] for h in bo_hist])
    z_samples = np.array([h['score'] for h in bo_hist])

    # ========== (a) 9D GP后验均值 μ (A,f)切片 ==========
    ax_mu = fig.add_subplot(gs[0, 0])

    im_mu = ax_mu.contourf(Xi, Yi, -Mu, levels=50, cmap='viridis', alpha=0.9)
    ax_mu.contour(Xi, Yi, -Mu, levels=15, colors='white', alpha=0.3, linewidths=0.5)

    ax_mu.scatter(x_samples[:20], y_samples[:20], c='white', edgecolor='black',
                  s=40, alpha=0.7, marker='s', linewidths=1.5, label='Random Init', zorder=10)
    ax_mu.scatter(x_samples[20:], y_samples[20:], c=z_samples[20:], cmap='autumn',
                  edgecolor='black', s=50, linewidths=1.5, label='9D BO Samples', zorder=10)

    best_idx = np.argmax(z_samples)
    ax_mu.scatter(x_samples[best_idx], y_samples[best_idx], s=300,
                  facecolors='none', edgecolors=PALETTE['highlight'],
                  lw=3, marker='*', zorder=20, label='9D Best')

    ax_mu.set_xlabel(r'$A_{\rm mod}$', fontweight='bold', fontsize=11)
    ax_mu.set_ylabel(r'$f_{\rm mod}$', fontweight='bold', fontsize=11)
    ax_mu.set_title(r'(a) 9D GP Posterior Mean ($\mu$)' +
                    f'\n(φ={fixed_params.get("phi0", 0):.2f},α={fixed_params.get("alpha", 0):.2f},β={fixed_params.get("beta", 0):.2f})',
                    fontweight='bold', fontsize=11, pad=10)
    ax_mu.legend(loc='upper left', frameon=True, fontsize=8, fancybox=True, framealpha=0.9)

    cbar_mu = plt.colorbar(im_mu, ax=ax_mu, fraction=0.046, pad=0.04)
    cbar_mu.set_label('Predicted Score (μ)', rotation=270, labelpad=15, fontweight='bold')

    # ========== (b) 9D GP不确定性 σ ==========
    ax_sigma = fig.add_subplot(gs[0, 1])

    im_sigma = ax_sigma.contourf(Xi, Yi, Sigma, levels=50, cmap='Reds', alpha=0.9)
    ax_sigma.contour(Xi, Yi, Sigma, levels=10, colors='darkred', alpha=0.3, linewidths=0.5)

    # 标注高不确定性区域
    max_unc_idx = np.unravel_index(np.argmax(Sigma), Sigma.shape)
    ax_sigma.scatter(Xi[max_unc_idx], Yi[max_unc_idx], s=250,
                     facecolors='none', edgecolors='yellow', lw=2.5,
                     marker='o', zorder=20, label='Max Uncertainty')

    # 采样轨迹
    ax_sigma.plot(x_samples[20:], y_samples[20:], color='white',
                  lw=1, alpha=0.5, ls='--', zorder=5)
    ax_sigma.scatter(x_samples[20:], y_samples[20:], c=PALETTE['highlight'],
                     edgecolor='black', s=50, linewidths=1.5,
                     label='9D BO Samples', zorder=10)

    ax_sigma.set_xlabel(r'$A_{\rm mod}$', fontweight='bold', fontsize=11)
    ax_sigma.set_ylabel(r'$f_{\rm mod}$', fontweight='bold', fontsize=11)
    ax_sigma.set_title(r'(b) 9D GP Uncertainty ($\sigma$)',
                       fontweight='bold', fontsize=11, pad=10)
    ax_sigma.legend(loc='upper left', frameon=True, fontsize=8, fancybox=True, framealpha=0.9)

    cbar_sigma = plt.colorbar(im_sigma, ax=ax_sigma, fraction=0.046, pad=0.04)
    cbar_sigma.set_label('Std Deviation (σ)', rotation=270, labelpad=15, fontweight='bold')

    # ========== (c) 9D不确定性演化 + 参数贡献度 ==========
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
                            marker='o', markersize=4, label=r'9D GP Prediction ($\mu$)', zorder=10)
        line2 = ax_ev1.plot(iters, actual_scores, color=PALETTE['highlight'], lw=2,
                            marker='s', markersize=4, label='Actual Score', zorder=10)
        ax_ev1.fill_between(iters,
                            np.array(mu_samples) - np.array(sigma_samples),
                            np.array(mu_samples) + np.array(sigma_samples),
                            color=PALETTE['bo'], alpha=0.2, label=r'$\mu \pm \sigma$')

        # 不确定性演化
        line3 = ax_ev2.plot(iters, sigma_samples, color='red', lw=2,
                            marker='^', markersize=4, label=r'9D Uncertainty ($\sigma$)',
                            linestyle='--', zorder=5)

        ax_ev1.set_xlabel('Iteration', fontweight='bold', fontsize=11)
        ax_ev1.set_ylabel('9D Objective Score', fontweight='bold', fontsize=11, color=PALETTE['bo'])
        ax_ev2.set_ylabel('9D Uncertainty (sigma)', fontweight='bold', fontsize=11, color='red')
        ax_ev1.tick_params(axis='y', labelcolor=PALETTE['bo'])
        ax_ev2.tick_params(axis='y', labelcolor='red')

        ax_ev1.set_title(r'(c) 9D Prediction vs Reality',
                         fontweight='bold', fontsize=11, pad=10)
        ax_ev1.grid(True, ls=':', alpha=0.3)

        # 合并图例
        lines = line1 + line2 + line3
        labels = [l.get_label() for l in lines]
        ax_ev1.legend(lines, labels, loc='upper left', frameon=True, fontsize=8)
    else:
        ax_evolution.text(0.5, 0.5,
                          "No 9D posterior data", ha='center', va='center',
                          transform=ax_evolution.transAxes)

    fig.suptitle('9D Bayesian Optimization Global Uncertainty Reduction',
                 fontweight='bold', fontsize=14, y=0.98)

    # 保存图片
    for fmt in ['png', 'pdf']:
        output_path = f"figures_9D/{base_path}.{fmt}"
        dpi = 300 if fmt == 'png' else None
        plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
        print(f"[SAVED] Saved {fmt.upper()} format: {output_path}")

    plt.close()


# ======================== 主运行入口 ========================
if __name__ == "__main__":
    import os

    os.makedirs("figures_9D", exist_ok=True)

    # 初始化9D优化器
    opt = NLSEOptimizer9D(x_range=(-50, 50), nx=256, t_max=12, gamma=1.5, seed=42)

    # 运行全部优化算法
    results = opt.run_all(max_iter=100)

    # 生成指定风格的四张图
    plot_rogue_monster_dynamics(opt, results, base_path="9D_NLSE_Optimization_Results")
    plot_landscape_comparison(opt, results, base_path="9D_Landscape_Comparison")
    plot_raincloud_statistics(opt, results, base_path="9D_Raincloud_Statistics")
    plot_bo_uncertainty_analysis(opt, results, base_path="9D_BO_Uncertainty")

    print_separator()
    print("[SAVED] 9D Optimization Complete! (Plots saved as PNG files)")
    print_separator()
