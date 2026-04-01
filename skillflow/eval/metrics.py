"""评估指标计算：通过率、pass@k、pass^k、reward。"""

from __future__ import annotations

import numpy as np


def compute_pass_rate(case_results: list[dict]) -> float:
    """计算单个测试用例的通过率。"""
    if not case_results:
        return 0.0
    passes = sum(1 for r in case_results if r["pass"])
    return passes / len(case_results)


def compute_reward(scores, threshold=0.8):
    """根据多次 trial 得分列表计算 reward。

    公式：
    - top_quality = max(f)  （潜力：最好的一次表现）
    - bottom_quality = min(f) if 全员达标 else 0  （稳健：最差表现）
    - mean_quality = mean(scores)  （整体水平）
    - base_score = 0.3 * top + 0.2 * bottom + 0.5 * mean
    - reward = base_score * (1 - min(var, 0.25) / 0.25)
    """
    scores = np.array(scores, dtype=float)
    f = np.where(scores > threshold, scores, 0)

    top_quality = float(np.max(f))
    bottom_quality = float(np.min(f)) if np.all(scores > threshold) else 0.0
    mean_quality = float(np.mean(scores))
    var_val = float(np.var(scores))

    stability_discount = 1 - min(var_val, 0.25) / 0.25
    base_score = 0.3 * top_quality + 0.2 * bottom_quality + 0.5 * mean_quality
    final_reward = base_score * stability_discount

    return round(final_reward, 4)


def compute_case_metrics(case_results: list[dict], k: int = 3, threshold: float = 0.8) -> dict:
    """计算单个测试用例的详细指标。

    Args:
        case_results: [{"pass": bool, "score": float, ...}, ...]
        k: pass@k 中的 k 值
        threshold: 通过阈值

    Returns:
        包含 pass_at_k, pass_hat_k, pass_rate_mean, pass_rate_var, reward
    """
    pass_rate = compute_pass_rate(case_results)
    scores = [r["score"] for r in case_results]
    scores_arr = np.array(scores, dtype=float)

    mean_score = float(np.mean(scores_arr))
    var_score = float(np.var(scores_arr))
    p_at_k = 1 - (1 - pass_rate) ** k
    p_hat_k = pass_rate ** k
    reward = compute_reward(scores, threshold)

    return {
        "pass_at_k": round(p_at_k, 4),
        "pass_hat_k": round(p_hat_k, 4),
        "pass_rate_mean": round(mean_score, 4),
        "pass_rate_var": round(var_score, 4),
        "reward": reward,
    }


def compute_overall_reward(case_rewards: list[float]) -> float:
    """根据所有用例的 reward 列表计算全局 reward。"""
    return compute_reward(case_rewards)
