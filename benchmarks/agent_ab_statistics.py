"""Paired binary power, multiplicity, and interval helpers for agent A/B runs."""

from __future__ import annotations

import hashlib
import math
import random
from functools import lru_cache
from statistics import NormalDist
from typing import Any, Sequence


_NORMAL = NormalDist()


def _wilson_interval(successes: int, count: int, confidence: float) -> tuple[float, float]:
    if count <= 0:
        return (0.0, 1.0)
    z = _NORMAL.inv_cdf(0.5 + confidence / 2.0)
    proportion = successes / count
    denominator = 1.0 + z * z / count
    center = (proportion + z * z / (2.0 * count)) / denominator
    half_width = z * math.sqrt(
        proportion * (1.0 - proportion) / count + z * z / (4.0 * count * count)
    ) / denominator
    return (max(0.0, center - half_width), min(1.0, center + half_width))


def _binomial_cdf(successes: int, count: int, probability: float) -> float:
    """Evaluate a binomial CDF by summing outward from the nearer tail."""
    if successes < 0:
        return 0.0
    if successes >= count:
        return 1.0
    if probability <= 0.0:
        return 1.0
    if probability >= 1.0:
        return 0.0

    def log_mass(value: int) -> float:
        return (
            math.lgamma(count + 1)
            - math.lgamma(value + 1)
            - math.lgamma(count - value + 1)
            + value * math.log(probability)
            + (count - value) * math.log1p(-probability)
        )

    mode = int((count + 1) * probability)
    if successes < mode:
        term = math.exp(log_mass(successes))
        total = term
        for value in range(successes, 0, -1):
            term *= value * (1.0 - probability) / ((count - value + 1) * probability)
            total += term
        return min(1.0, total)

    first_upper = successes + 1
    term = math.exp(log_mass(first_upper))
    upper_tail = term
    for value in range(first_upper, count):
        term *= (count - value) * probability / ((value + 1) * (1.0 - probability))
        upper_tail += term
    return max(0.0, 1.0 - upper_tail)


def clopper_pearson_upper_bound(successes: int, count: int, confidence: float) -> float:
    """Return a one-sided exact binomial upper confidence bound."""
    if (
        isinstance(successes, bool)
        or not isinstance(successes, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not 0 <= successes <= count
    ):
        raise ValueError("binomial successes and count must be valid integers")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if successes == count:
        return 1.0
    alpha = 1.0 - confidence
    if successes == 0:
        return -math.expm1(math.log(alpha) / count)

    lower = successes / count
    upper = 1.0
    for _ in range(64):
        middle = (lower + upper) / 2.0
        if _binomial_cdf(successes, count, middle) > alpha:
            lower = middle
        else:
            upper = middle
    return upper


def clopper_pearson_lower_bound(successes: int, count: int, confidence: float) -> float:
    """Return a one-sided exact binomial lower confidence bound."""
    if (
        isinstance(successes, bool)
        or not isinstance(successes, int)
        or isinstance(count, bool)
        or not isinstance(count, int)
        or count < 1
        or not 0 <= successes <= count
    ):
        raise ValueError("binomial successes and count must be valid integers")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    if successes == 0:
        return 0.0
    alpha = 1.0 - confidence
    target_cdf = 1.0 - alpha
    lower = 0.0
    upper = successes / count
    for _ in range(64):
        middle = (lower + upper) / 2.0
        if _binomial_cdf(successes - 1, count, middle) > target_cdf:
            lower = middle
        else:
            upper = middle
    return lower


def clopper_pearson_interval(
    successes: int,
    count: int,
    confidence: float,
) -> tuple[float, float]:
    """Return a two-sided exact Clopper-Pearson binomial interval."""
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")
    tail_confidence = 1.0 - (1.0 - confidence) / 2.0
    lower = clopper_pearson_lower_bound(successes, count, tail_confidence)
    upper = clopper_pearson_upper_bound(successes, count, tail_confidence)
    return lower, upper


def newcombe_paired_score_interval(
    *,
    both_successes: int,
    mcp_only_successes: int,
    control_only_successes: int,
    count: int,
    confidence: float = 0.95,
) -> dict[str, float | int]:
    """Return Newcombe's method-10 paired score interval for MCP minus control.

    ``both_successes``, ``mcp_only_successes``, and
    ``control_only_successes`` are cells from the matched 2x2 table. The
    concordant-failure cell is derived from ``count``.
    """
    cells = (both_successes, mcp_only_successes, control_only_successes, count)
    if any(isinstance(value, bool) or not isinstance(value, int) for value in cells):
        raise ValueError("paired score interval counts must be integers")
    if count < 1 or min(both_successes, mcp_only_successes, control_only_successes) < 0:
        raise ValueError("paired score interval counts must be nonnegative and count positive")
    if both_successes + mcp_only_successes + control_only_successes > count:
        raise ValueError("paired score interval cells exceed the pair count")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be between zero and one")

    e = both_successes
    f = mcp_only_successes
    g = control_only_successes
    h = count - e - f - g
    mcp_rate = (e + f) / count
    control_rate = (e + g) / count
    estimate = (f - g) / count
    denominator = math.sqrt((e + f) * (g + h) * (e + g) * (f + h))
    if denominator == 0:
        phi = 0.0
    else:
        numerator = e * h - f * g
        if numerator > 0:
            numerator = max(0.0, numerator - count / 2.0)
        phi = numerator / denominator
        phi = max(-1.0, min(1.0, phi))

    alpha = 1.0 - confidence
    mcp_low, mcp_high = _wilson_interval(sum((e, f)), count, confidence)
    control_low, control_high = _wilson_interval(sum((e, g)), count, confidence)
    mcp_down = max(0.0, mcp_rate - mcp_low)
    mcp_up = max(0.0, mcp_high - mcp_rate)
    control_down = max(0.0, control_rate - control_low)
    control_up = max(0.0, control_high - control_rate)
    lower_distance = math.sqrt(max(
        0.0,
        mcp_down * mcp_down - 2.0 * phi * mcp_down * control_up + control_up * control_up,
    ))
    upper_distance = math.sqrt(max(
        0.0,
        mcp_up * mcp_up - 2.0 * phi * mcp_up * control_down + control_down * control_down,
    ))
    return {
        "estimate": estimate,
        "lower": max(-1.0, estimate - lower_distance),
        "upper": min(1.0, estimate + upper_distance),
        "confidence": confidence,
        "count": count,
        "method": "Newcombe-1998-method-10-paired-score",
        "alpha": alpha,
    }


@lru_cache(maxsize=200_000)
def exact_mcnemar_pvalue(mcp_only_successes: int, control_only_successes: int) -> float:
    """Two-sided exact McNemar p-value using the conditional binomial test."""
    if min(mcp_only_successes, control_only_successes) < 0:
        raise ValueError("discordant pair counts must be nonnegative")
    discordant = mcp_only_successes + control_only_successes
    if discordant == 0:
        return 1.0
    lower_tail_count = min(mcp_only_successes, control_only_successes)
    log_probability = (
        math.lgamma(discordant + 1)
        - math.lgamma(lower_tail_count + 1)
        - math.lgamma(discordant - lower_tail_count + 1)
        - discordant * math.log(2.0)
    )
    probability = math.exp(log_probability) if log_probability > -745.0 else 0.0
    lower_tail = probability
    for value in range(lower_tail_count, 0, -1):
        probability *= value / (discordant - value + 1)
        lower_tail += probability
    return min(1.0, 2.0 * lower_tail)


def holm_adjust(p_values: Sequence[float | None]) -> list[float | None]:
    """Return Holm step-down adjusted p-values while preserving input order."""
    present = [
        (index, float(value))
        for index, value in enumerate(p_values)
        if value is not None
    ]
    if any(not math.isfinite(value) or value < 0.0 or value > 1.0 for _, value in present):
        raise ValueError("p-values must be finite values between zero and one")
    present.sort(key=lambda pair: pair[1])
    adjusted: list[float | None] = [None] * len(p_values)
    running_max = 0.0
    count = len(present)
    for rank, (index, value) in enumerate(present):
        running_max = max(running_max, min(1.0, (count - rank) * value))
        adjusted[index] = running_max
    return adjusted


def _holm_rejections(p_values: Sequence[float], alpha: float) -> list[bool]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    rejected = [False] * len(p_values)
    for rank, index in enumerate(order):
        if p_values[index] > alpha / (len(order) - rank):
            break
        rejected[index] = True
    return rejected


def _derive_validation_seed(seed: int, attempt: int) -> int:
    """Create a reproducible, domain-separated validation seed."""
    label = f"agent-ab-power-validation:{seed}:{attempt}".encode("ascii")
    return int.from_bytes(hashlib.sha256(label).digest()[:8], "big")


def simulate_paired_mcnemar_power(
    *,
    repetitions: int,
    effect: float,
    discordance_rates: Sequence[float],
    simulations: int = 5000,
    alpha: float = 0.05,
    seed: int = 20261031,
) -> dict[str, Any]:
    """Simulate matched binary pairs, exact McNemar tests, then Holm correction.

    The paired outcome probabilities are determined by the marginal difference
    ``effect`` and the discordance probability. Each pair is sampled as one of
    the two discordant outcomes or a concordant outcome; independent-arm
    Bernoulli samples are never used.
    """
    if repetitions < 1 or simulations < 1:
        raise ValueError("repetitions and simulations must be positive")
    if not discordance_rates:
        raise ValueError("at least one task discordance rate is required")
    if not -1.0 <= effect <= 1.0:
        raise ValueError("effect must be between -1 and one")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be between zero and one")
    for rate in discordance_rates:
        if not 0.0 <= rate <= 1.0 or rate + 1e-12 < abs(effect):
            raise ValueError("discordance rates must be at least the absolute effect and at most one")

    rng = random.Random(seed)
    rejected_counts = [0] * len(discordance_rates)
    any_rejected_count = 0
    for _ in range(simulations):
        p_values = []
        for discordance in discordance_rates:
            mcp_only_probability = (discordance + effect) / 2.0
            discordant_probability = discordance
            mcp_only = 0
            control_only = 0
            for _ in range(repetitions):
                draw = rng.random()
                if draw < mcp_only_probability:
                    mcp_only += 1
                elif draw < discordant_probability:
                    control_only += 1
            p_values.append(exact_mcnemar_pvalue(mcp_only, control_only))
        rejections = _holm_rejections(p_values, alpha)
        any_rejected_count += int(any(rejections))
        for index, rejected in enumerate(rejections):
            rejected_counts[index] += int(rejected)

    powers = [count / simulations for count in rejected_counts]
    intervals = [_wilson_interval(count, simulations, 0.95) for count in rejected_counts]
    return {
        "repetitions": repetitions,
        "effect": effect,
        "discordance_rates": list(discordance_rates),
        "simulations": simulations,
        "seed": seed,
        "alpha": alpha,
        "power_by_task": powers,
        "power_lower_95_by_task": [interval[0] for interval in intervals],
        "power_upper_95_by_task": [interval[1] for interval in intervals],
        "rejection_counts_by_task": rejected_counts,
        "familywise_rejection_rate": any_rejected_count / simulations,
    }


def simulate_paired_interval_coverage(
    *,
    repetitions: int,
    effect: float,
    discordance: float,
    simulations: int = 5000,
    confidence: float = 0.95,
    seed: int = 20261031,
) -> dict[str, float | int]:
    """Measure coverage of the paired score interval under a feasible model."""
    if repetitions < 1 or simulations < 1:
        raise ValueError("repetitions and simulations must be positive")
    probabilities = estimate_pair_probabilities(0.5, effect, discordance)
    thresholds = []
    total = 0.0
    for name in ("mcp_only_success", "control_only_success", "both_success"):
        total += probabilities[name]
        thresholds.append((total, name))
    rng = random.Random(seed)
    covered = 0
    for _ in range(simulations):
        both_successes = 0
        mcp_only = 0
        control_only = 0
        for _ in range(repetitions):
            draw = rng.random()
            if draw < thresholds[0][0]:
                mcp_only += 1
            elif draw < thresholds[1][0]:
                control_only += 1
            elif draw < thresholds[2][0]:
                both_successes += 1
        interval = newcombe_paired_score_interval(
            both_successes=both_successes,
            mcp_only_successes=mcp_only,
            control_only_successes=control_only,
            count=repetitions,
            confidence=confidence,
        )
        covered += int(interval["lower"] <= effect <= interval["upper"])
    rate = covered / simulations
    interval = _wilson_interval(covered, simulations, 0.95)
    return {
        "repetitions": repetitions,
        "effect": effect,
        "discordance": discordance,
        "simulations": simulations,
        "confidence": confidence,
        "seed": seed,
        "coverage": rate,
        "coverage_lower_95": interval[0],
        "coverage_upper_95": interval[1],
    }


def bootstrap_discordance_upper(
    pairs: Sequence[tuple[bool, bool]],
    *,
    simulations: int = 5000,
    seed: int = 20261031,
    quantile: float = 0.95,
) -> dict[str, Any]:
    """Bootstrap matched binary pairs and return a conservative discordance rate."""
    if not pairs:
        raise ValueError("at least one matched pair is required")
    if simulations < 1 or not 0.5 < quantile < 1.0:
        raise ValueError("bootstrap simulations and quantile are invalid")
    rng = random.Random(seed)
    count = len(pairs)
    baseline_successes = sum(control for control, _ in pairs)
    baseline_rate = baseline_successes / count
    baseline_interval = clopper_pearson_interval(baseline_successes, count, quantile)
    observed_discordances = sum(control != mcp for control, mcp in pairs)
    observed_discordance = observed_discordances / count
    bootstrap_rates = []
    for _ in range(simulations):
        discordant = 0
        for _ in range(count):
            control, mcp = pairs[rng.randrange(count)]
            discordant += control != mcp
        bootstrap_rates.append(discordant / count)
    bootstrap_rates.sort()
    index = min(simulations - 1, math.ceil(quantile * simulations) - 1)
    bootstrap_upper = bootstrap_rates[index]
    # Use the exact binomial bound for sizing. A bootstrap percentile can
    # undercover with small discordant counts and is retained only as a diagnostic.
    upper_estimate = clopper_pearson_upper_bound(observed_discordances, count, quantile)
    upper_method = "one-sided-clopper-pearson"
    return {
        "matched_repetitions": count,
        "baseline_success_rate": baseline_rate,
        "baseline_success_rate_interval": {
            "confidence": quantile,
            "lower": baseline_interval[0],
            "upper": baseline_interval[1],
            "method": "clopper-pearson",
        },
        "observed_discordance_rate": observed_discordance,
        "discordance_upper_bootstrap": bootstrap_upper,
        "discordance_upper_estimate": upper_estimate,
        "discordance_upper_method": upper_method,
        "bootstrap_simulations": simulations,
        "bootstrap_seed": seed,
        "bootstrap_quantile": quantile,
    }


def estimate_pair_probabilities(
    baseline_success: float,
    effect: float,
    discordance: float,
) -> dict[str, float]:
    """Construct a feasible paired 2x2 distribution for a specified marginal effect."""
    if not 0.0 <= baseline_success <= 1.0:
        raise ValueError("baseline success must be a probability")
    if discordance + 1e-12 < abs(effect) or discordance > 1.0:
        raise ValueError("discordance must be at least the absolute effect")
    mcp_only = (discordance + effect) / 2.0
    control_only = (discordance - effect) / 2.0
    both_success = baseline_success - control_only
    both_failure = 1.0 - baseline_success - mcp_only
    probabilities = {
        "both_success": both_success,
        "mcp_only_success": mcp_only,
        "control_only_success": control_only,
        "both_failure": both_failure,
    }
    if min(probabilities.values()) < -1e-12:
        raise ValueError("baseline, effect, and discordance do not define feasible paired outcomes")
    return {key: max(0.0, value) for key, value in probabilities.items()}


def normal_approximate_repetitions(
    *,
    effect: float,
    target_power: float,
    discordance_rates: Sequence[float],
    family_alpha: float = 0.05,
    task_count: int | None = None,
) -> int:
    """Return a conservative planning start using Holm's first-step cutoff.

    The first Holm threshold is family_alpha / task_count. Designing every
    task to meet that stricter threshold is conservative for the full step-down
    procedure and makes the initial sample-size search inexpensive.
    """
    if not 0.0 < abs(effect) <= 1.0:
        raise ValueError("effect must be nonzero and between minus one and one")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target power must be between zero and one")
    if not discordance_rates:
        raise ValueError("at least one task discordance rate is required")
    task_count = task_count or len(discordance_rates)
    if task_count < 1 or family_alpha <= 0.0 or family_alpha >= 1.0:
        raise ValueError("task count and family alpha are invalid")
    per_task_alpha = family_alpha / task_count
    z_alpha = _NORMAL.inv_cdf(1.0 - per_task_alpha / 2.0)
    z_power = _NORMAL.inv_cdf(target_power)
    denominator = effect * effect
    required = []
    for discordance in discordance_rates:
        if discordance + 1e-12 < abs(effect) or discordance > 1.0:
            raise ValueError("discordance rates must be at least the absolute effect and at most one")
        variance = max(0.0, discordance - effect * effect)
        required.append(((z_alpha + z_power) ** 2) * variance / denominator)
    return max(1, math.ceil(max(required)))


def select_simulated_repetitions(
    *,
    effect: float,
    target_power: float,
    discordance_rates: Sequence[float],
    simulations: int = 3000,
    family_alpha: float = 0.05,
    seed: int = 20261031,
) -> dict[str, Any]:
    """Propose a starting size, then certify it with independent simulations.

    A normal approximation proposes the starting size. Final power bounds use
    independent exact binomial lower bounds, with error spent across tasks and
    retries.
    """
    if simulations < 100:
        raise ValueError("at least 100 simulations are required for design selection")
    if not 0.0 < abs(effect) <= 1.0:
        raise ValueError("effect must be nonzero and between minus one and one")
    if not 0.0 < target_power < 1.0:
        raise ValueError("target power must be between zero and one")
    if not discordance_rates:
        raise ValueError("at least one task discordance rate is required")
    for rate in discordance_rates:
        if not 0.0 <= rate <= 1.0 or rate + 1e-12 < abs(effect):
            raise ValueError("discordance rates must be at least the absolute effect and at most one")
    if not 0.0 < family_alpha < 1.0:
        raise ValueError("family alpha must be between zero and one")

    # A normal approximation is only a starting candidate. It makes discovery
    # inexpensive; no evidence claim uses its result or its assumed variance.
    task_count = len(discordance_rates)
    discovery_candidate = normal_approximate_repetitions(
        effect=effect,
        target_power=target_power,
        discordance_rates=discordance_rates,
        family_alpha=family_alpha,
        task_count=task_count,
    )
    if discovery_candidate > 100000:
        raise ValueError("power target requires more than 100000 repetitions")

    # A candidate and every retry are validated on independent streams. The
    # per-attempt error budgets sum to 0.05; each is split across all tasks.
    candidate = discovery_candidate
    attempt = 1
    while True:
        attempt_alpha = 0.05 / (attempt * (attempt + 1))
        per_task_alpha = attempt_alpha / task_count
        confidence = 1.0 - per_task_alpha
        validation_seed = _derive_validation_seed(seed, attempt)
        validation = simulate_paired_mcnemar_power(
            repetitions=candidate,
            effect=effect,
            discordance_rates=discordance_rates,
            simulations=simulations,
            alpha=family_alpha,
            seed=validation_seed,
        )
        lower_bounds = [
            clopper_pearson_lower_bound(count, simulations, confidence)
            for count in validation["rejection_counts_by_task"]
        ]
        confidence_by_task = [confidence] * task_count
        if min(lower_bounds) >= target_power:
            validation.pop("power_lower_95_by_task", None)
            validation.pop("power_upper_95_by_task", None)
            validation.update({
                "power_lower_bound_by_task": lower_bounds,
                "power_lower_confidence_by_task": confidence_by_task,
                "power_bound_method": "one-sided-clopper-pearson",
                "validation_attempt": attempt,
                "validation_alpha_spending": attempt_alpha,
                "discovery_seed": seed,
            })
            return {
                "repetitions": candidate,
                "target_power": target_power,
                "selected_power": min(validation["power_by_task"]),
                "selected_power_lower_bound": min(lower_bounds),
                "selected_power_lower_bound_by_task": lower_bounds,
                "selected_power_confidence_by_task": confidence_by_task,
                "validation_attempt": attempt,
                "validation_alpha_spending": attempt_alpha,
                "discovery_candidate_repetitions": discovery_candidate,
                "power_simulation": validation,
                "selection_rule": (
                    "independently validated with one-sided exact Clopper-Pearson lower bounds; "
                    "alpha 0.05/[attempt*(attempt+1)] is split across tasks"
                ),
            }
        if candidate >= 100000:
            raise ValueError("power target could not be independently validated within 100000 repetitions")
        candidate = min(100000, max(candidate + 1, math.ceil(candidate * 1.05)))
        attempt += 1
