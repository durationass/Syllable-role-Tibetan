"""Acoustic boundary scoring and constrained syllable alignment."""

from dataclasses import dataclass

import numpy as np
from scipy.special import logsumexp


EPS = 1e-10


@dataclass(frozen=True)
class AlignmentResult:
    success: bool
    boundaries: tuple
    spans: tuple
    reason: str = ""


@dataclass(frozen=True)
class PosteriorAlignmentResult:
    success: bool
    boundaries: tuple
    spans: tuple
    boundary_posterior: np.ndarray
    boundary_entropy: np.ndarray
    conditional_boundary_posterior: np.ndarray
    conditional_boundary_entropy: np.ndarray
    top_path_margin: float
    reason: str = ""


@dataclass(frozen=True)
class _AlignmentLattice:
    scores: np.ndarray
    count: int
    start: int
    end: int
    mean_duration: float
    min_duration: int
    max_duration: int
    duration_prior_weight: float


def _as_frame_array(features, names):
    for name in names:
        if name in features:
            values = np.asarray(features[name], dtype=np.float64).reshape(-1)
            if values.size:
                return values
    return np.zeros(0, dtype=np.float64)


def _match_length(values, length):
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == length:
        return values
    output = np.zeros(length, dtype=np.float64)
    output[:min(length, values.size)] = values[:length]
    return output


def _robust_normalize(values):
    values = np.nan_to_num(np.asarray(values, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    if values.size == 0:
        return values
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    scale = 1.4826 * mad
    if scale <= EPS:
        scale = float(np.std(values))
    if scale <= EPS:
        return np.zeros_like(values)
    return np.clip((values - median) / scale, -8.0, 8.0)


def _symmetric_spectral_change(features, length):
    provided = _as_frame_array(features, ("symmetric_spectral_change", "spectral_change"))
    if provided.size:
        return _match_length(provided, length)
    magnitude = None
    for name in ("spectral_magnitude", "magnitude", "mag", "spectrogram"):
        if name in features:
            magnitude = np.asarray(features[name], dtype=np.float64)
            break
    if magnitude is None or magnitude.ndim != 2:
        return _match_length(_as_frame_array(features, ("flux",)), length)
    magnitude = magnitude[:length]
    change = np.zeros(len(magnitude), dtype=np.float64)
    if len(magnitude) > 1:
        forward = np.mean(np.abs(np.diff(magnitude, axis=0)), axis=1)
        change[1:] += forward
        change[:-1] += forward
    return _match_length(change, length)


def compute_acoustic_boundary_score(features):
    """Return an equal-weight, per-frame acoustic boundary score.

    Energy valleys, symmetric spectral changes, and voicing changes are each
    robust-normalized within one utterance before being combined.
    """
    candidates = [_as_frame_array(features, names) for names in (
        ("energy", "energy_norm"),
        ("symmetric_spectral_change", "spectral_change", "flux"),
        ("voicing_change",),
    )]
    lengths = [len(values) for values in candidates if len(values)]
    if not lengths:
        return np.zeros(0, dtype=np.float64)
    length = max(lengths)
    energy = _match_length(candidates[0], length)
    spectral = _symmetric_spectral_change(features, length)
    voiced = _as_frame_array(features, ("voicing_change",))
    if not voiced.size and "voiced" in features:
        voiced_values = _match_length(features["voiced"], length)
        voiced = np.abs(np.diff(voiced_values, prepend=voiced_values[:1]))
    voiced = _match_length(voiced, length)
    energy_valley = -_robust_normalize(energy)
    spectral_change = _robust_normalize(spectral)
    voicing_change = _robust_normalize(voiced)
    return (energy_valley + spectral_change + voicing_change) / 3.0


def _prepare_lattice(
    boundary_score,
    syllable_count,
    start_frame,
    end_frame,
    min_segment_ratio,
    max_segment_ratio,
    duration_prior_weight,
):
    scores = np.asarray(boundary_score, dtype=np.float64).reshape(-1)
    scores = np.where(np.isfinite(scores), scores, -np.inf)
    count = int(syllable_count)
    start = int(start_frame)
    end = int(end_frame)
    if count < 1:
        return None, "invalid_syllable_count"
    if start < 0 or end > len(scores) or end <= start:
        return None, "invalid_frame_range"
    if min_segment_ratio <= 0 or max_segment_ratio < min_segment_ratio:
        return None, "invalid_duration_constraints"
    total = end - start
    mean_duration = total / float(count)
    min_duration = max(1, int(np.ceil(mean_duration * float(min_segment_ratio))))
    max_duration = max(min_duration, int(np.floor(mean_duration * float(max_segment_ratio))))
    if count * min_duration > total or count * max_duration < total:
        return None, "no_feasible_duration_path"
    return _AlignmentLattice(
        scores=scores,
        count=count,
        start=start,
        end=end,
        mean_duration=mean_duration,
        min_duration=min_duration,
        max_duration=max_duration,
        duration_prior_weight=max(0.0, float(duration_prior_weight)),
    ), ""


def _endpoint_bounds(lattice, segments):
    remaining = lattice.count - segments
    low = max(
        lattice.start + segments * lattice.min_duration,
        lattice.end - remaining * lattice.max_duration,
    )
    high = min(
        lattice.start + segments * lattice.max_duration,
        lattice.end - remaining * lattice.min_duration,
    )
    return low, high


def _predecessor_bounds(lattice, segments, endpoint):
    low = max(
        lattice.start + (segments - 1) * lattice.min_duration,
        endpoint - lattice.max_duration,
    )
    high = min(
        endpoint - lattice.min_duration,
        lattice.start + (segments - 1) * lattice.max_duration,
    )
    return low, high


def _transition_score(lattice, prior, endpoint, segments):
    duration = endpoint - prior
    duration_penalty = ((duration / lattice.mean_duration) - 1.0) ** 2
    boundary_value = 0.0 if segments == lattice.count else lattice.scores[endpoint]
    return boundary_value - lattice.duration_prior_weight * duration_penalty


def _failure_alignment(reason):
    return AlignmentResult(False, tuple(), tuple(), reason)


def _failure_posterior(reason, frame_count):
    return PosteriorAlignmentResult(
        False,
        tuple(),
        tuple(),
        np.zeros((0, frame_count), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.zeros((0, frame_count), dtype=np.float64),
        np.zeros((0,), dtype=np.float64),
        np.nan,
        reason,
    )


def map_boundaries_dp(
    boundary_score,
    syllable_count,
    start_frame,
    end_frame,
    min_segment_ratio=0.25,
    max_segment_ratio=4.0,
    duration_prior_weight=0.2,
):
    """Map exactly ``N-1`` boundaries under duration constraints.

    Frames use half-open spans: ``start_frame <= frame < end_frame``. The
    returned boundary positions are the starts of the following spans.
    """
    lattice, reason = _prepare_lattice(
        boundary_score,
        syllable_count,
        start_frame,
        end_frame,
        min_segment_ratio,
        max_segment_ratio,
        duration_prior_weight,
    )
    if lattice is None:
        return _failure_alignment(reason)
    if lattice.count == 1:
        return AlignmentResult(True, tuple(), ((lattice.start, lattice.end),))

    # DP state is (number of completed segments, current endpoint).
    neg_inf = -np.inf
    dp = np.full((lattice.count + 1, lattice.end + 1), neg_inf, dtype=np.float64)
    previous = np.full((lattice.count + 1, lattice.end + 1), -1, dtype=np.int32)
    dp[0, lattice.start] = 0.0
    for segments in range(1, lattice.count + 1):
        low_endpoint, high_endpoint = _endpoint_bounds(lattice, segments)
        for endpoint in range(low_endpoint, high_endpoint + 1):
            previous_low, previous_high = _predecessor_bounds(lattice, segments, endpoint)
            best_value = neg_inf
            best_previous = -1
            for prior in range(previous_low, previous_high + 1):
                if not np.isfinite(dp[segments - 1, prior]):
                    continue
                value = dp[segments - 1, prior] + _transition_score(
                    lattice, prior, endpoint, segments
                )
                if value > best_value:
                    best_value = value
                    best_previous = prior
            dp[segments, endpoint] = best_value
            previous[segments, endpoint] = best_previous

    if not np.isfinite(dp[lattice.count, lattice.end]):
        return _failure_alignment("no_feasible_path")
    endpoints = [lattice.end]
    endpoint = lattice.end
    for segments in range(lattice.count, 0, -1):
        endpoint = int(previous[segments, endpoint])
        if endpoint < 0:
            return _failure_alignment("broken_backtrace")
        endpoints.append(endpoint)
    endpoints.reverse()
    spans = tuple((int(left), int(right)) for left, right in zip(endpoints[:-1], endpoints[1:]))
    boundaries = tuple(int(value) for value in endpoints[1:-1])
    return AlignmentResult(True, boundaries, spans)


def _top_path_margin(lattice):
    top_scores = np.full(
        (lattice.count + 1, lattice.end + 1, 2), -np.inf, dtype=np.float64
    )
    top_scores[0, lattice.start, 0] = 0.0
    for segments in range(1, lattice.count + 1):
        low_endpoint, high_endpoint = _endpoint_bounds(lattice, segments)
        for endpoint in range(low_endpoint, high_endpoint + 1):
            previous_low, previous_high = _predecessor_bounds(lattice, segments, endpoint)
            candidates = []
            for prior in range(previous_low, previous_high + 1):
                transition = _transition_score(lattice, prior, endpoint, segments)
                if not np.isfinite(transition):
                    continue
                for rank in range(2):
                    previous_score = top_scores[segments - 1, prior, rank]
                    if np.isfinite(previous_score):
                        candidates.append(previous_score + transition)
            if candidates:
                candidates.sort(reverse=True)
                top_scores[segments, endpoint, :min(2, len(candidates))] = candidates[:2]
    best, second = top_scores[lattice.count, lattice.end]
    if not np.isfinite(best):
        return np.nan
    if not np.isfinite(second):
        return np.inf
    return max(0.0, float(best - second))


def _conditional_boundary_posteriors(lattice, map_result, frame_count):
    posterior = np.zeros((lattice.count - 1, frame_count), dtype=np.float64)
    entropy = np.zeros((lattice.count - 1,), dtype=np.float64)
    for boundary_index, map_boundary in enumerate(map_result.boundaries):
        segments = boundary_index + 1
        left = int(map_result.spans[boundary_index][0])
        right = int(map_result.spans[boundary_index + 1][1])
        low = max(
            left + lattice.min_duration,
            right - lattice.max_duration,
        )
        high = min(
            left + lattice.max_duration,
            right - lattice.min_duration,
        )
        positions = []
        scores = []
        for endpoint in range(low, high + 1):
            left_score = _transition_score(lattice, left, endpoint, segments)
            right_score = _transition_score(
                lattice, endpoint, right, segments + 1
            )
            value = left_score + right_score
            if np.isfinite(value):
                positions.append(endpoint)
                scores.append(value)
        if not positions or int(map_boundary) not in positions:
            return None, None, "invalid_posterior"

        probabilities = np.exp(np.asarray(scores) - logsumexp(scores))
        probability_sum = float(np.sum(probabilities))
        if not np.isfinite(probability_sum) or not np.isclose(
            probability_sum, 1.0, rtol=1e-7, atol=1e-9
        ):
            return None, None, "invalid_posterior"
        probabilities /= probability_sum
        posterior[boundary_index, positions] = probabilities
        map_probability = posterior[boundary_index, int(map_boundary)]
        if not np.isclose(
            map_probability, np.max(probabilities), rtol=1e-12, atol=1e-15
        ):
            return None, None, "invalid_posterior"
        if len(positions) > 1:
            positive = probabilities > 0.0
            value = -float(
                np.sum(probabilities[positive] * np.log(probabilities[positive]))
            )
            value /= float(np.log(len(positions)))
            entropy[boundary_index] = np.clip(value, 0.0, 1.0)
    return posterior, entropy, ""


def forward_backward_boundaries(
    boundary_score,
    syllable_count,
    start_frame,
    end_frame,
    min_segment_ratio=0.25,
    max_segment_ratio=4.0,
    duration_prior_weight=0.2,
):
    """Return MAP alignment and boundary uncertainty on the MAP-DP lattice."""
    frame_count = np.asarray(boundary_score).size
    lattice, reason = _prepare_lattice(
        boundary_score,
        syllable_count,
        start_frame,
        end_frame,
        min_segment_ratio,
        max_segment_ratio,
        duration_prior_weight,
    )
    if lattice is None:
        return _failure_posterior(reason, frame_count)

    map_result = map_boundaries_dp(
        boundary_score,
        syllable_count,
        start_frame,
        end_frame,
        min_segment_ratio,
        max_segment_ratio,
        duration_prior_weight,
    )
    if not map_result.success:
        return _failure_posterior(map_result.reason, frame_count)
    if lattice.count == 1:
        return PosteriorAlignmentResult(
            True,
            map_result.boundaries,
            map_result.spans,
            np.zeros((0, frame_count), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.zeros((0, frame_count), dtype=np.float64),
            np.zeros((0,), dtype=np.float64),
            np.inf,
        )

    alpha = np.full((lattice.count + 1, lattice.end + 1), -np.inf, dtype=np.float64)
    beta = np.full_like(alpha, -np.inf)
    alpha[0, lattice.start] = 0.0
    for segments in range(1, lattice.count + 1):
        low_endpoint, high_endpoint = _endpoint_bounds(lattice, segments)
        for endpoint in range(low_endpoint, high_endpoint + 1):
            previous_low, previous_high = _predecessor_bounds(lattice, segments, endpoint)
            candidates = []
            for prior in range(previous_low, previous_high + 1):
                transition = _transition_score(lattice, prior, endpoint, segments)
                if np.isfinite(alpha[segments - 1, prior]) and np.isfinite(transition):
                    candidates.append(alpha[segments - 1, prior] + transition)
            if candidates:
                alpha[segments, endpoint] = logsumexp(candidates)

    log_z = alpha[lattice.count, lattice.end]
    if not np.isfinite(log_z):
        return _failure_posterior("no_feasible_path", frame_count)

    beta[lattice.count, lattice.end] = 0.0
    for segments in range(lattice.count - 1, -1, -1):
        next_segments = segments + 1
        low_endpoint, high_endpoint = _endpoint_bounds(lattice, next_segments)
        suffixes = {}
        for endpoint in range(low_endpoint, high_endpoint + 1):
            if not np.isfinite(beta[next_segments, endpoint]):
                continue
            previous_low, previous_high = _predecessor_bounds(
                lattice, next_segments, endpoint
            )
            for prior in range(previous_low, previous_high + 1):
                transition = _transition_score(lattice, prior, endpoint, next_segments)
                if np.isfinite(transition):
                    suffixes.setdefault(prior, []).append(
                        transition + beta[next_segments, endpoint]
                    )
        for prior, candidates in suffixes.items():
            beta[segments, prior] = logsumexp(candidates)

    posterior = np.zeros((lattice.count - 1, frame_count), dtype=np.float64)
    entropy = np.zeros((lattice.count - 1,), dtype=np.float64)
    for boundary_index, segments in enumerate(range(1, lattice.count)):
        feasible = np.isfinite(alpha[segments]) & np.isfinite(beta[segments])
        feasible_positions = np.flatnonzero(feasible)
        if feasible_positions.size == 0:
            return _failure_posterior("invalid_posterior", frame_count)
        log_probability = (
            alpha[segments, feasible_positions]
            + beta[segments, feasible_positions]
            - log_z
        )
        probabilities = np.exp(log_probability)
        probability_sum = float(np.sum(probabilities))
        if not np.isfinite(probability_sum) or not np.isclose(
            probability_sum, 1.0, rtol=1e-7, atol=1e-9
        ):
            return _failure_posterior("invalid_posterior", frame_count)
        probabilities /= probability_sum
        posterior[boundary_index, feasible_positions] = probabilities
        if feasible_positions.size > 1:
            positive = probabilities > 0.0
            value = -float(np.sum(probabilities[positive] * np.log(probabilities[positive])))
            value /= float(np.log(feasible_positions.size))
            entropy[boundary_index] = np.clip(value, 0.0, 1.0)

    margin = _top_path_margin(lattice)
    if np.isnan(margin):
        return _failure_posterior("no_feasible_path", frame_count)
    conditional_posterior, conditional_entropy, reason = (
        _conditional_boundary_posteriors(lattice, map_result, frame_count)
    )
    if conditional_posterior is None:
        return _failure_posterior(reason, frame_count)
    return PosteriorAlignmentResult(
        True,
        map_result.boundaries,
        map_result.spans,
        posterior,
        entropy,
        conditional_posterior,
        conditional_entropy,
        margin,
    )
