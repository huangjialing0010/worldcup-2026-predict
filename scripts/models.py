"""
Dixon-Coles 模型 + 优化后的预测模型
排名参数化的攻防强度，用 MLE 拟合
"""
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from model_utils import (
    FIFA_RANKINGS, rank_to_strength, actual_result, poisson_best_score,
    poisson_probabilities,
)


# ============================================================
# Dixon-Coles 模型
# ============================================================


def _tau(x, y, lambda_h, lambda_a, rho):
    """Dixon-Coles 低比分依赖修正项 τ(x,y)"""
    if x == 0 and y == 0:
        return 1 - lambda_h * lambda_a * rho
    elif x == 0 and y == 1:
        return 1 + lambda_h * rho
    elif x == 1 and y == 0:
        return 1 + lambda_a * rho
    elif x == 1 and y == 1:
        return 1 - rho
    else:
        return 1.0


def _dc_loglikelihood(params, matches, rankings):
    """Dixon-Coles 负对数似然（供 minimize 使用）"""
    A, B, C, D, gamma, rho = params

    # 约束检查
    if A <= 0 or B <= 0 or C <= 0 or D <= 0 or gamma <= 0 or rho <= -0.05 or rho >= 0.05:
        return 1e10

    ll = 0
    for home, away, h_goals, a_goals in matches:
        rank_h = rankings.get(home, 50)
        rank_a = rankings.get(away, 50)

        attack_h = A * np.exp(-rank_h / B)
        defense_h = C * np.exp(-rank_h / D)
        attack_a = A * np.exp(-rank_a / B)
        defense_a = C * np.exp(-rank_a / D)

        lambda_h = attack_h * defense_a * gamma
        lambda_a = attack_a * defense_h

        if lambda_h <= 0 or lambda_a <= 0:
            return 1e10

        tau_val = _tau(h_goals, a_goals, lambda_h, lambda_a, rho)
        if tau_val <= 0:
            return 1e10

        prob_h = stats.poisson.pmf(h_goals, lambda_h)
        prob_a = stats.poisson.pmf(a_goals, lambda_a)
        joint = tau_val * prob_h * prob_a

        if joint <= 0:
            return 1e10

        ll += np.log(joint)

    return -ll  # 返回负对数似然


class DixonColes:
    """排名参数化的 Dixon-Coles 模型"""

    def __init__(self):
        self.A = None
        self.B = None
        self.C = None
        self.D = None
        self.gamma = None
        self.rho = None
        self.fitted = False

    def fit(self, matches, rankings=None):
        """用 MLE 拟合模型参数"""
        if rankings is None:
            rankings = FIFA_RANKINGS

        # 初始值：基于泊松模型的合理猜测
        # A ~ 1.5 (进攻基准), B ~ 150 (排名衰减), C ~ 1.0 (防守基准), D ~ 200, gamma ~ 1.1, rho ~ 0
        init = [1.5, 150.0, 1.0, 200.0, 1.10, 0.0]
        bounds = [
            (0.1, 10.0),    # A
            (20.0, 500.0),  # B
            (0.1, 10.0),    # C
            (20.0, 500.0),  # D
            (0.8, 1.8),     # gamma
            (-0.04, 0.04),  # rho
        ]

        result = minimize(
            _dc_loglikelihood, init, args=(matches, rankings),
            method="L-BFGS-B", bounds=bounds,
            options={"maxiter": 5000, "ftol": 1e-12},
        )

        if not result.success:
            print(f"  Dixon-Coles 拟合警告: {result.message}")

        self.A, self.B, self.C, self.D, self.gamma, self.rho = result.x
        self.fitted = True
        return self

    def get_lambdas(self, home, away, rankings=None):
        """计算对阵双方的预期进球 λ"""
        if not self.fitted:
            raise RuntimeError("模型未拟合，请先调用 fit()")
        if rankings is None:
            rankings = FIFA_RANKINGS

        rank_h = rankings.get(home, 50)
        rank_a = rankings.get(away, 50)

        attack_h = self.A * np.exp(-rank_h / self.B)
        defense_h = self.C * np.exp(-rank_h / self.D)
        attack_a = self.A * np.exp(-rank_a / self.B)
        defense_a = self.C * np.exp(-rank_a / self.D)

        lambda_h = attack_h * defense_a * self.gamma
        lambda_a = attack_a * defense_h
        return lambda_h, lambda_a

    def predict_score(self, home, away, rankings=None, max_goals=8):
        """预测最可能比分和结果（结果用聚合概率）"""
        lambda_h, lambda_a = self.get_lambdas(home, away, rankings)
        # 用聚合概率判结果（含 Dixon-Coles τ 修正）
        p_h, p_d, p_a = self.predict_proba(home, away, rankings, max_goals)
        best = max(p_h, p_d, p_a)
        result = "H" if best == p_h else ("D" if best == p_d else "A")
        # 最可能比分
        score = poisson_best_score(lambda_h, lambda_a, max_goals)
        return result, score[1], score[2]

    def predict_proba(self, home, away, rankings=None, max_goals=10):
        """预测 H/D/A 概率（含 Dixon-Coles τ 修正）"""
        lambda_h, lambda_a = self.get_lambdas(home, away, rankings)

        p_h, p_d, p_a = 0.0, 0.0, 0.0
        for i in range(max_goals + 1):
            for j in range(max_goals + 1):
                joint = (
                    _tau(i, j, lambda_h, lambda_a, self.rho)
                    * stats.poisson.pmf(i, lambda_h)
                    * stats.poisson.pmf(j, lambda_a)
                )
                if i > j:
                    p_h += joint
                elif i == j:
                    p_d += joint
                else:
                    p_a += joint

        total = p_h + p_d + p_a
        return p_h / total, p_d / total, p_a / total

    def predict_result(self, home, away, rankings=None):
        p_h, p_d, p_a = self.predict_proba(home, away, rankings)
        best = max(p_h, p_d, p_a)
        if best == p_h:
            return "H"
        elif best == p_d:
            return "D"
        else:
            return "A"

    def get_params(self):
        return {
            "A": round(self.A, 4),
            "B": round(self.B, 1),
            "C": round(self.C, 4),
            "D": round(self.D, 1),
            "gamma": round(self.gamma, 4),
            "rho": round(self.rho, 6),
        }


# ============================================================
# 优化后的泊松模型（使用调优参数）
# ============================================================


class OptimizedPoisson:
    """使用优化参数的泊松模型"""

    def __init__(self, avg_goals=2.85, scale=100, home_advantage=1.15, max_goals=8):
        self.avg_goals = avg_goals
        self.scale = scale
        self.home_advantage = home_advantage
        self.max_goals = max_goals

    def get_lambdas(self, home, away, rankings=None):
        if rankings is None:
            rankings = FIFA_RANKINGS
        strength_h = rank_to_strength(rankings.get(home, 50), self.scale)
        strength_a = rank_to_strength(rankings.get(away, 50), self.scale)
        lambda_h = self.avg_goals * (strength_h * self.home_advantage) / (strength_h + strength_a)
        lambda_a = self.avg_goals * strength_a / (strength_h + strength_a)
        return lambda_h, lambda_a

    def predict_score(self, home, away, rankings=None):
        lambda_h, lambda_a = self.get_lambdas(home, away, rankings)
        # 用聚合概率判结果，单比分找最可能比分
        p_h, p_d, p_a = poisson_probabilities(lambda_h, lambda_a)
        best = max(p_h, p_d, p_a)
        result = "H" if best == p_h else ("D" if best == p_d else "A")
        score = poisson_best_score(lambda_h, lambda_a, self.max_goals)
        return result, score[1], score[2]

    def predict_proba(self, home, away, rankings=None):
        lambda_h, lambda_a = self.get_lambdas(home, away, rankings)
        return poisson_probabilities(lambda_h, lambda_a)

    def predict_result(self, home, away, rankings=None):
        p_h, p_d, p_a = self.predict_proba(home, away, rankings)
        best = max(p_h, p_d, p_a)
        return "H" if best == p_h else ("D" if best == p_d else "A")


# ============================================================
# 优化后的 ELO 模型
# ============================================================


class OptimizedELO:
    """使用优化参数的 ELO 模型（支持中立场 + 可调平局衰减）"""

    def __init__(self, rank_weight=4, home_bias=45, draw_coeff=0.35,
                 draw_decay=727, draw_power=1.4, neutral_venue=False):
        self.rank_weight = rank_weight
        self.home_bias = home_bias
        self.draw_coeff = draw_coeff
        self.draw_decay = draw_decay
        self.draw_power = draw_power
        self.neutral_venue = neutral_venue

    def predict_proba(self, home, away, rankings=None):
        if rankings is None:
            rankings = FIFA_RANKINGS
        elo_diff = (rankings.get(away, 50) - rankings.get(home, 50)) * self.rank_weight + self.home_bias
        p_h = 1.0 / (1 + 10 ** (-elo_diff / 400))
        p_draw = self.draw_coeff * np.exp(-(np.abs(elo_diff) / self.draw_decay) ** self.draw_power)
        p_h -= p_draw / 2
        p_a = 1.0 - p_h - p_draw
        return p_h, p_draw, p_a

    def predict_result(self, home, away, rankings=None):
        p_h, p_d, p_a = self.predict_proba(home, away, rankings)
        best = max(p_h, p_d, p_a)
        if best == p_h:
            return "H"
        elif best == p_d:
            return "D"
        else:
            return "A"


# ============================================================
# 模型加载工具
# ============================================================


def load_best_params():
    """加载优化后的参数"""
    import json
    params_path = Path(__file__).parent.parent / "output" / "best_params.json"
    if params_path.exists():
        return json.loads(params_path.read_text(encoding="utf-8"))
    return None


def create_optimized_poisson(params=None):
    if params is None:
        params = load_best_params()
    if params and "poisson" in params:
        p = params["poisson"]
        return OptimizedPoisson(
            avg_goals=p.get("avg_goals", 2.85),
            scale=p.get("scale", 100),
            home_advantage=p.get("home_advantage", 1.15),
            max_goals=p.get("max_goals", 8),
        )
    return OptimizedPoisson()


def create_optimized_elo(params=None, neutral_venue=False):
    if params is None:
        params = load_best_params()
    if params and "elo" in params:
        e = params["elo"]
        # 中立场使用专门参数
        if neutral_venue and "elo_neutral" in params:
            e = params["elo_neutral"]
        return OptimizedELO(
            rank_weight=e.get("rank_weight", 4),
            home_bias=e.get("home_bias", 45),
            draw_coeff=e.get("draw_coeff", 0.35),
            draw_decay=e.get("draw_decay", 727),
            draw_power=e.get("draw_power", 1.4),
            neutral_venue=neutral_venue,
        )
    return OptimizedELO()


# ============================================================
# 平局感知泊松模型
# ============================================================


class DrawAwarePoisson(OptimizedPoisson):
    """
    在泊松模型基础上增加平局检测：
    当预期进球差小 + 排名接近时，将预测修正为平局
    """

    def __init__(self, avg_goals=2.85, scale=100, home_advantage=1.15, max_goals=8,
                 draw_goal_margin=0.5, draw_rank_closeness=20):
        super().__init__(avg_goals, scale, home_advantage, max_goals)
        self.draw_goal_margin = draw_goal_margin
        self.draw_rank_closeness = draw_rank_closeness

    def predict_result(self, home, away, rankings=None):
        if rankings is None:
            rankings = FIFA_RANKINGS
        lambda_h, lambda_a = self.get_lambdas(home, away, rankings)
        goal_diff = abs(lambda_h - lambda_a)
        rank_diff = abs(rankings.get(home, 50) - rankings.get(away, 50))

        # 平局条件：预期进球差小 且 排名接近
        if goal_diff < self.draw_goal_margin and rank_diff < self.draw_rank_closeness:
            return "D"

        # 用聚合概率判结果
        p_h, p_d, p_a = poisson_probabilities(lambda_h, lambda_a)
        best = max(p_h, p_d, p_a)
        return "H" if best == p_h else ("D" if best == p_d else "A")

    def predict_score(self, home, away, rankings=None):
        if rankings is None:
            rankings = FIFA_RANKINGS
        result = self.predict_result(home, away, rankings)

        # 如果预测平局，用泊松找最可能的平局比分
        if result == "D":
            lambda_h, lambda_a = self.get_lambdas(home, away, rankings)
            best_prob = -1
            best_h, best_a = 0, 0
            for h in range(self.max_goals + 1):
                for a in range(self.max_goals + 1):
                    if h == a:  # 只看平局比分
                        prob = stats.poisson.pmf(h, lambda_h) * stats.poisson.pmf(a, lambda_a)
                        if prob > best_prob:
                            best_prob = prob
                            best_h, best_a = h, a
            return "D", best_h, best_a

        return super().predict_score(home, away, rankings)


def optimize_draw_thresholds(matches, rankings, poisson_params, n_iter=500):
    """优化平局检测的两个阈值"""
    import random as rnd
    rng = rnd.Random(42)

    best_acc = -1
    best_thresholds = {}

    for _ in range(n_iter):
        goal_margin = rng.uniform(0.1, 1.5)
        rank_close = rng.randint(5, 60)

        model = DrawAwarePoisson(
            avg_goals=poisson_params.get("avg_goals", 2.85),
            scale=poisson_params.get("scale", 100),
            home_advantage=poisson_params.get("home_advantage", 1.15),
            max_goals=poisson_params.get("max_goals", 8),
            draw_goal_margin=goal_margin,
            draw_rank_closeness=rank_close,
        )

        correct = 0
        for home, away, h_s, a_s in matches:
            pred = model.predict_result(home, away, rankings)
            if pred == actual_result(h_s, a_s):
                correct += 1
        acc = correct / len(matches) * 100

        if acc > best_acc:
            best_acc = acc
            best_thresholds = {
                "draw_goal_margin": round(goal_margin, 2),
                "draw_rank_closeness": rank_close,
                "accuracy": round(acc, 1),
            }

    return best_thresholds


# 为 backtest.py 兼容提供的函数接口
from pathlib import Path


def predict_poisson_optimized(home, away, rankings, params=None):
    model = create_optimized_poisson(params)
    return model.predict_score(home, away, rankings)


def predict_elo_optimized(home, away, rankings, params=None):
    model = create_optimized_elo(params)
    return model.predict_result(home, away, rankings)
