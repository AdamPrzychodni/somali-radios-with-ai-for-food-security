"""BERTopic topic modelling, multi-theme selection and theme-map verification."""

from __future__ import annotations

from typing import Any

from ..config import get_setting


def fit_topic_model(
    docs: list[str],
    model_kwargs: dict[str, Any] | None = None,
    seed: int | None = None,
):
    """Fit a BERTopic model and return primary topics plus a probability matrix.

    UMAP is seeded (``project.seed``) because BERTopic assigns topic ids by cluster
    size: without a fixed seed a rerun reshuffles the ids, and ``theme_map`` — which
    resolves themes *by id* — silently relabels every document.

    ``bertopic`` and ``umap`` are imported lazily (they need the ``[analysis]`` extras).

    Returns:
        ``(model, topic_ids, probabilities)`` where *probabilities* is an
        ``N_docs x N_topics`` list of lists.
    """
    from bertopic import BERTopic
    from umap import UMAP

    if seed is None:
        seed = get_setting("project.seed", 42)

    kwargs = dict(model_kwargs or {})
    kwargs.setdefault(
        "umap_model",
        UMAP(
            n_neighbors=15,
            n_components=5,
            min_dist=0.0,
            metric="cosine",
            random_state=seed,
        ),
    )

    model = BERTopic(**kwargs, calculate_probabilities=True)
    topics, probs = model.fit_transform(docs)
    return model, topics, probs.tolist()


def topic_map_mismatches(
    topic_words: dict[int, list[str]],
    theme_map: dict[int, str],
    theme_keywords: dict[str, list[str]],
) -> dict[int, str]:
    """Return the ``{topic_id: theme}`` entries whose top words no longer fit the theme.

    *topic_words* is topic id -> its top words. A theme with no expected keywords
    configured is skipped rather than reported.
    """
    mismatches: dict[int, str] = {}
    for topic_id, theme in theme_map.items():
        expected = {word.lower() for word in theme_keywords.get(theme, [])}
        if not expected:
            continue
        actual = {word.lower() for word in topic_words.get(topic_id, [])}
        if not expected & actual:
            mismatches[topic_id] = theme
    return mismatches


def verify_theme_map(
    model,
    theme_map: dict[int, str],
    theme_keywords: dict[str, list[str]] | None = None,
    top_n: int = 10,
) -> None:
    """Raise if a fitted model's topics no longer match the labels in *theme_map*.

    A wrong theme label is worse than a crash: it propagates into the IPC update
    without any signal that it happened.
    """
    if theme_keywords is None:
        theme_keywords = get_setting("topics.theme_keywords", {})
    if not theme_keywords:
        return

    topic_words = {
        topic_id: [word for word, _ in (model.get_topic(topic_id) or [])[:top_n]]
        for topic_id in theme_map
    }
    mismatches = topic_map_mismatches(topic_words, theme_map, theme_keywords)
    if mismatches:
        detail = ", ".join(
            f"topic {tid} labelled '{theme}' -> {topic_words.get(tid, [])[:5]}"
            for tid, theme in mismatches.items()
        )
        raise ValueError(
            "topics.theme_map no longer matches the fitted topics. BERTopic ids depend "
            f"on the data, so the labels have drifted: {detail}. Update theme_map (and "
            "theme_keywords) in config/config.yaml before trusting these themes."
        )


def select_themes_per_doc(
    probabilities: list[list[float]],
    theme_map: dict[int, str],
    prob_threshold: float = 0.1,
) -> list[list[str]]:
    """For each document, collect every theme whose probability >= *prob_threshold*.

    Themes are resolved by topic id, i.e. by position in the probability vector —
    which only holds while the model is seeded and :func:`verify_theme_map` passes.

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
