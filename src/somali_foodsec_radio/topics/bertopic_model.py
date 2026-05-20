"""BERTopic topic modelling and multi-theme selection."""

from __future__ import annotations

from typing import Any


def fit_topic_model(
    docs: list[str], model_kwargs: dict[str, Any] | None = None
):
    """Fit a BERTopic model and return primary topics plus a probability matrix.

    ``bertopic`` is imported lazily (it requires the ``[analysis]`` extras).

    Returns:
        ``(model, topic_ids, probabilities)`` where *probabilities* is an
        ``N_docs x N_topics`` list of lists.
    """
    from bertopic import BERTopic

    model = BERTopic(**(model_kwargs or {}), calculate_probabilities=True)
    topics, probs = model.fit_transform(docs)
    return model, topics, probs.tolist()


def select_themes_per_doc(
    probabilities: list[list[float]],
    theme_map: dict[int, str],
    prob_threshold: float = 0.1,
) -> list[list[str]]:
    """For each document, collect every theme whose probability >= *prob_threshold*.

    Documents with no theme above the threshold get ``["Other"]``.
    """
    all_themes: list[list[str]] = []
    for doc_probs in probabilities:
        themes = [
            theme_map[tid]
            for tid, p in enumerate(doc_probs)
            if p >= prob_threshold and tid in theme_map
        ]
        all_themes.append(themes or ["Other"])
    return all_themes
