"""Version-side evidence acceptance for the calibrated hybrid profile."""

import math


PROFILE = "hybrid_rrf"
TOP_K = 5
SCORE = "max_dense_cosine_in_top_k"
SCORE_DECIMALS = 6
FALSE_REJECT_LIMIT = 0.10


def confidence_score(hits: list[dict], dense_candidates: list[dict]) -> float | None:
    returned = {hit["chunk_id"] for hit in hits}
    score = max(
        (float(hit["score"]) for hit in dense_candidates if hit["chunk_id"] in returned),
        default=None,
    )
    return round(score, SCORE_DECIMALS) if score is not None else None


def accepts(score: float | None, threshold: float) -> bool:
    return score is not None and math.isfinite(score) and score >= threshold


def metrics(rows: list[dict], threshold: float | None) -> dict:
    positives = sum(row["answerable"] for row in rows)
    negatives = len(rows) - positives
    false_accept = false_reject = accepted_answerable = accepted_missing_gold = 0
    for row in rows:
        accepted = row["has_hits"] if threshold is None else accepts(row["score"], threshold)
        if row["answerable"]:
            false_reject += not accepted
            accepted_answerable += accepted
            accepted_missing_gold += accepted and not row["covers_gold"]
        else:
            false_accept += accepted
    return {
        "false_accept": rate(false_accept, negatives),
        "false_reject": rate(false_reject, positives),
        "accepted_answerable_missing_gold": rate(accepted_missing_gold, accepted_answerable),
    }


def rate(count: int, total: int) -> dict:
    return {"count": count, "total": total, "rate": count / total if total else None}


def choose_threshold(dev: list[dict]) -> float:
    if not dev or len({row["answerable"] for row in dev}) != 2:
        raise ValueError("Calibration needs both answerable and unanswerable development sides")
    scores = {row["score"] for row in dev if row["score"] is not None and math.isfinite(row["score"])}
    if not scores:
        raise ValueError("Development set produced no finite confidence scores")
    scores.add(math.nextafter(max(scores), math.inf))
    choices = []
    for threshold in sorted(scores):
        result = metrics(dev, threshold)
        if result["false_reject"]["rate"] <= FALSE_REJECT_LIMIT:
            choices.append((result["false_accept"]["count"], -threshold))
    if not choices:
        raise ValueError("No threshold meets the development false-reject budget")
    return -min(choices)[1]
