"""
泊松回归模型：用 ELO + 近期状态 + 交手记录预测进球数
时间序列分割：pre-2026 训练，2026 WC 测试
"""
import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
from pathlib import Path
import json
import warnings
warnings.filterwarnings("ignore")

ROOT = Path(__file__).parent.parent
OUTPUT_DIR = ROOT / "output"

# ============================================================
# 数据加载
# ============================================================

def load_feature_data():
    df = pd.read_csv(OUTPUT_DIR / "features.csv", encoding="utf-8-sig")
    df["date"] = pd.to_datetime(df["date"])
    return df

def split_data(df):
    """时间切分：pre-2026 训练 / 2026 WC 测试"""
    wc2026 = df[(df["date"] >= "2026-06-01") & (df["round"] == "FIFA World Cup")]
    train = df[df["date"] < "2026-01-01"]
    # 加 2026 非 WC 比赛到训练集
    extra = df[(df["date"] >= "2026-01-01") & (df["round"] != "FIFA World Cup")]
    train = pd.concat([train, extra])
    return train, wc2026

def actual_result(hg, ag):
    if hg > ag: return "H"
    elif ag > hg: return "A"
    return "D"

# ============================================================
# 泊松回归模型
# ============================================================

class PoissonRegression:
    """泊松 GLM — 分别拟合 home λ 和 away λ"""

    def __init__(self):
        self.coef_h = None  # for home goals
        self.coef_a = None  # for away goals
        self.feature_names = None

    def _design_matrix(self, df):
        """构建设计矩阵"""
        X = np.column_stack([
            np.ones(len(df)),                        # intercept
            df["elo_h"].values / 100,                # home ELO
            df["elo_a"].values / 100,                # away ELO
            df["h_win5"].values,                     # home recent win rate
            df["a_win5"].values,                     # away recent win rate
            df["h_gf5"].values,                      # home recent goals for
            df["h_ga5"].values,                      # home recent goals against
            df["a_gf5"].values,                      # away recent goals for
            df["a_ga5"].values,                      # away recent goals against
            df["h2h_n"].values.clip(0, 5),           # h2h count (cap at 5)
            df["h2h_h_winrate"].values,              # h2h home win rate
        ])
        self.feature_names = [
            "intercept", "elo_h/100", "elo_a/100",
            "h_win5", "a_win5", "h_gf5", "h_ga5", "a_gf5", "a_ga5",
            "h2h_n", "h2h_h_winrate",
        ]
        return X

    def _poisson_loglik(self, beta, X, y):
        """负对数似然"""
        lam = np.exp(X @ beta)
        lam = np.clip(lam, 0.01, 20.0)
        return -np.sum(stats.poisson.logpmf(y, lam))

    def fit(self, df):
        X = self._design_matrix(df)
        y_h = df["home_goals"].values.astype(int)
        y_a = df["away_goals"].values.astype(int)

        init = np.zeros(X.shape[1])
        init[0] = 0.2  # intercept ~ log(avg goals/2)

        res_h = minimize(self._poisson_loglik, init, args=(X, y_h),
                         method="L-BFGS-B", options={"maxiter": 5000})
        res_a = minimize(self._poisson_loglik, init, args=(X, y_a),
                         method="L-BFGS-B", options={"maxiter": 5000})

        self.coef_h = res_h.x
        self.coef_a = res_a.x
        return self

    def predict_lambdas(self, df):
        X = self._design_matrix(df)
        lam_h = np.exp(np.clip(X @ self.coef_h, -5, 5))
        lam_a = np.exp(np.clip(X @ self.coef_a, -5, 5))
        return lam_h, lam_a

    def predict_proba(self, df_row_or_lam):
        """输入一行 DataFrame 或 (lam_h, lam_a)，返回 (p_h, p_d, p_a)"""
        if isinstance(df_row_or_lam, tuple):
            lam_h, lam_a = df_row_or_lam
        else:
            lh, la = self.predict_lambdas(pd.DataFrame([df_row_or_lam]))
            lam_h, lam_a = lh[0], la[0]

        MAX_G = 10
        g = np.arange(MAX_G + 1)
        pmf_h = stats.poisson.pmf(g, lam_h)
        pmf_a = stats.poisson.pmf(g, lam_a)

        p_h = sum(pmf_h[i] * sum(pmf_a[:i]) for i in range(MAX_G + 1))
        p_d = sum(pmf_h[i] * pmf_a[i] for i in range(MAX_G + 1))
        p_a = 1.0 - p_h - p_d
        return p_h, p_d, p_a

    def predict_result(self, df_row):
        ph, pd, pa = self.predict_proba(df_row)
        best = max(ph, pd, pa)
        return "H" if best == ph else ("D" if best == pd else "A")

    def predict_score(self, df_row):
        lh, la = self.predict_lambdas(pd.DataFrame([df_row]))
        lam_h, lam_a = lh[0], la[0]
        # 最可能比分
        MAX_G = 8
        best_prob, best_h, best_a = -1, 0, 0
        for h in range(MAX_G + 1):
            for a in range(MAX_G + 1):
                prob = stats.poisson.pmf(h, lam_h) * stats.poisson.pmf(a, lam_a)
                if prob > best_prob:
                    best_prob, best_h, best_a = prob, h, a
        result = self.predict_result(df_row)
        return result, best_h, best_a

    def get_params(self):
        return {
            "home_coef": {n: round(c, 4) for n, c in zip(self.feature_names, self.coef_h)},
            "away_coef": {n: round(c, 4) for n, c in zip(self.feature_names, self.coef_a)},
        }


# ============================================================
# 评估
# ============================================================

def evaluate_model(model, df, name=""):
    correct = 0
    exact = 0
    mae = 0
    n = len(df)

    for _, row in df.iterrows():
        result, h_pred, a_pred = model.predict_score(row)
        actual = actual_result(int(row["home_goals"]), int(row["away_goals"]))
        if result == actual:
            correct += 1
        if h_pred == int(row["home_goals"]) and a_pred == int(row["away_goals"]):
            exact += 1
        mae += abs(h_pred - int(row["home_goals"])) + abs(a_pred - int(row["away_goals"]))

    acc = correct / n * 100
    print(f"  {name}: {correct}/{n} = {acc:.1f}%  |  比分命中 {exact}/{n}  |  MAE {mae/n:.2f}")
    return acc

def compare_models(models, df):
    """逐行对比多个模型"""
    print(f"\n{'Match':<35} {'Actual':>8} ", end="")
    for name in models:
        print(f"{name:>12}", end=" ")
    print()

    for _, row in df.iterrows():
        actual = actual_result(int(row["home_goals"]), int(row["away_goals"]))
        print(f"{row['home_team']} vs {row['away_team']:<12} {int(row['home_goals'])}:{int(row['away_goals'])} ({actual}) ", end="")
        for name, model in models.items():
            pred = model.predict_result(row)
            mark = "OK" if pred == actual else "XX"
            print(f"{pred:>10} {mark}", end=" ")
        print()


# ============================================================
# 主流程
# ============================================================

def main():
    print("=" * 60)
    print("  泊松回归模型训练与评估")
    print("=" * 60)

    df = load_feature_data()
    train, test = split_data(df)
    print(f"\n训练集: {len(train)} 场 (pre-2026 + 2026非WC)")
    print(f"测试集: {len(test)} 场 (2026 World Cup)")

    # --- 训练泊松回归 ---
    print("\n[1] 泊松回归 (ELO + 状态 + H2H)")
    pr = PoissonRegression()
    pr.fit(train)

    coefs = pr.get_params()
    for side in ["home_coef", "away_coef"]:
        print(f"  {side}:")
        for name, val in coefs[side].items():
            print(f"    {name}: {val:+.4f}")

    train_acc = evaluate_model(pr, train, "训练集")
    test_acc = evaluate_model(pr, test, "测试集(2026 WC)")

    # --- 基准对比：当前最优模型 ---
    print("\n[2] 基准: 旧泊松模型 (FIFA排名)")
    from model_utils import load_rankings
    from models import create_optimized_poisson, load_best_params

    rankings = load_rankings()
    best_params = load_best_params()
    old_poisson = create_optimized_poisson(best_params)

    class OldModelWrapper:
        def __init__(self, model, rankings):
            self.model = model
            self.rankings = rankings
        def predict_result(self, row):
            return self.model.predict_result(row["home_team"], row["away_team"], self.rankings)
        def predict_score(self, row):
            r, h, a = self.model.predict_score(row["home_team"], row["away_team"], self.rankings)
            return r, h, a

    old_wrapper = OldModelWrapper(old_poisson, rankings)
    old_test_acc = evaluate_model(old_wrapper, test, "旧泊松(测试集)")

    # --- 基准对比：旧ELO模型 ---
    from models import create_optimized_elo
    old_elo = create_optimized_elo(best_params)

    class OldELOWrapper:
        def __init__(self, model, rankings):
            self.model = model
            self.rankings = rankings
        def predict_result(self, row):
            return self.model.predict_result(row["home_team"], row["away_team"], self.rankings)
        def predict_score(self, row):
            return self.predict_result(row), 0, 0

    old_elo_w = OldELOWrapper(old_elo, rankings)
    elo_test_acc = evaluate_model(old_elo_w, test, "旧ELO(测试集)")

    # --- 逐场对比 ---
    print("\n[3] 逐场对比 (2026 WC)")
    models = {
        "新泊松回归": pr,
        "旧泊松(FIFA)": old_wrapper,
        "旧ELO(FIFA)": old_elo_w,
    }
    compare_models(models, test)

    # --- 简版：只用 ELO 的回归 ---
    print("\n[4] 消融: 仅用 ELO 的泊松回归")
    pr_elo_only = PoissonRegression()

    class ELoOnlyPR(PoissonRegression):
        def _design_matrix(self, df):
            X = np.column_stack([
                np.ones(len(df)),
                df["elo_h"].values / 100,
                df["elo_a"].values / 100,
            ])
            self.feature_names = ["intercept", "elo_h/100", "elo_a/100"]
            return X

    pr_elo = ELoOnlyPR()
    pr_elo.fit(train)
    print("  系数:", pr_elo.get_params())
    evaluate_model(pr_elo, train, "ELO-only 训练")
    evaluate_model(pr_elo, test, "ELO-only 测试")

    # --- 保存 ---
    print(f"\n[5] 结果")
    improvement = test_acc - old_test_acc
    print(f"  旧泊松: {old_test_acc:.1f}%")
    print(f"  新泊松回归: {test_acc:.1f}%")
    print(f"  提升: {improvement:+.1f}pp")

    # 保存模型参数
    model_params = {
        "model": "PoissonRegression",
        "features": pr.feature_names,
        "coefficients": coefs,
        "train_accuracy": round(train_acc, 1),
        "test_accuracy": round(test_acc, 1),
        "baseline_accuracy": round(old_test_acc, 1),
        "improvement_pp": round(improvement, 1),
        "n_train": len(train),
        "n_test": len(test),
    }
    out_path = OUTPUT_DIR / "poisson_regression_params.json"
    out_path.write_text(json.dumps(model_params, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  参数保存到 {out_path}")


if __name__ == "__main__":
    main()
