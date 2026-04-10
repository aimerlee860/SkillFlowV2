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
    - n = len(scores)（实际 trial 数）
    - mean_score = mean(scores)          （整体质量）
    - pass@n = 1 - (1 - pass_rate)^n     （可达性：至少通过一次的概率）
    - pass^n = pass_rate^n               （可靠性：全部通过的概率）
    - reward = 0.5 × mean_score + 0.2 × pass@n + 0.3 × pass^n
    """
    scores = np.array(scores, dtype=float)
    n = len(scores)
    mean_score = float(np.mean(scores))
    pass_rate = float(np.mean(scores > threshold))
    p_at_k = 1 - (1 - pass_rate) ** n
    p_hat_k = pass_rate ** n

    return round(0.5 * mean_score + 0.2 * p_at_k + 0.3 * p_hat_k, 4)


def compute_case_metrics(case_results: list[dict], threshold: float = 0.8) -> dict:
    """计算单个测试用例的详细指标。

    Args:
        case_results: [{"pass": bool, "score": float, ...}, ...]
        threshold: 通过阈值

    Returns:
        包含 pass_at_k, pass_hat_k, pass_rate_mean, pass_rate_var, reward
    """
    pass_rate = compute_pass_rate(case_results)
    scores = [r["score"] for r in case_results]
    scores_arr = np.array(scores, dtype=float)
    n = len(scores)

    mean_score = float(np.mean(scores_arr))
    var_score = float(np.var(scores_arr))
    p_at_k = 1 - (1 - pass_rate) ** n
    p_hat_k = pass_rate ** n
    reward = compute_reward(scores, threshold)

    return {
        "pass_at_k": round(p_at_k, 4),
        "pass_hat_k": round(p_hat_k, 4),
        "pass_rate_mean": round(mean_score, 4),
        "pass_rate_var": round(var_score, 4),
        "reward": reward,
    }


def compute_overall_reward(case_rewards: list[float]) -> float:
    """根据所有用例的 reward 列表计算全局 reward。

    归一化综合法：均值权重 0.8，稳定性权重 0.2。
    overall_reward = 0.8 × mean + 0.2 × (1 - std)
    """
    if not case_rewards:
        return 0.0
    rewards = np.array(case_rewards, dtype=float)
    mean_val = float(np.mean(rewards))
    std_val = float(np.std(rewards))
    return round(0.8 * mean_val + 0.2 * (1 - std_val), 4)
