"""Reciprocal rank fusion of lexical and local semantic search results."""


def reciprocal_rank_fusion(
    lexical, semantic, limit: int, smoothing: int = 60,
    lexical_weight: float = 1.0, semantic_weight: float = 1.0,
):
    fused = {}
    content = {}
    for weight, result_set in ((lexical_weight, lexical), (semantic_weight, semantic)):
        for rank, (text, source, _score) in enumerate(result_set, start=1):
            key = (text, source)
            content[key] = (text, source)
            fused[key] = fused.get(key, 0.0) + weight / (smoothing + rank)
    ranked = sorted(fused, key=fused.get, reverse=True)
    return [
        (*content[key], fused[key])
        for key in ranked[:limit]
    ]
