"""评估指标计算：通过率、pass@k、pass^k。"""

from __future__ import annotations

from math import comb


def compute_pass_rate(case_results: list[dict]) -> float:
    """计算单个测试用例的通过率。"""
    if not case_results:
        return 0.0
    passes = sum(1 for r in case_results if r["pass"])
    return passes / len(case_results)


def compute_pass_at_k(n: int, c: int, k: int) -> float:
    """计算 pass@k：k 次采样中至少一次通过的概率。

    公式：1 - C(n-c, k) / C(n, k)
    """
    if n < k:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - comb(n - c, k) / comb(n, k)


def compute_pass_hat_k(n: int, c: int, k: int) -> float:
    """计算 pass^k：k 次采样全部通过的概率。

    公式：C(c, k) / C(n, k)
    """
    if c < k or n < k:
        return 0.0
    return comb(c, k) / comb(n, k)


def compute_all_metrics(case_results: list[dict], trials: int) -> dict:
    """计算所有指标。

    Args:
        case_results: [{"test_point": ..., "results": [{"pass": bool, ...}, ...]}]
        trials: 总运行次数

    Returns:
        包含所有指标的 dict
    """
    n = trials
    total_passes = 0
    total_cases = len(case_results)

    for case in case_results:
        c = sum(1 for r in case["results"] if r["pass"])
        total_passes += c

    # 总体通过率
    overall_pass_rate = total_passes / (total_cases * n) if total_cases * n > 0 else 0.0

    # 平均用例通过率
    case_pass_rates = []
    for case in case_results:
        c = sum(1 for r in case["results"] if r["pass"])
        case_pass_rates.append(c / n if n > 0 else 0.0)
    avg_case_pass_rate = sum(case_pass_rates) / len(case_pass_rates) if case_pass_rates else 0.0

    # 对每个用例计算 pass@k 和 pass^k，然后取平均
    k_values = [1, 3, 5] if n >= 5 else list(range(1, n + 1))
    pass_at_k = {}
    pass_hat_k = {}

    for k in k_values:
        if k > n:
            continue
        at_k_values = []
        hat_k_values = []
        for case in case_results:
            c = sum(1 for r in case["results"] if r["pass"])
            at_k_values.append(compute_pass_at_k(n, c, k))
            hat_k_values.append(compute_pass_hat_k(n, c, k))
        pass_at_k[f"k={k}"] = sum(at_k_values) / len(at_k_values) if at_k_values else 0.0
        pass_hat_k[f"k={k}"] = sum(hat_k_values) / len(hat_k_values) if hat_k_values else 0.0

    return {
        "overall_pass_rate": round(overall_pass_rate, 4),
        "avg_case_pass_rate": round(avg_case_pass_rate, 4),
        "pass_at_k": {k: round(v, 4) for k, v in pass_at_k.items()},
        "pass_hat_k": {k: round(v, 4) for k, v in pass_hat_k.items()},
    }
