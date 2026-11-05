"""Canonical representation for exact hypothesis identity."""

import hashlib
import json
from typing import Any
'''
Hypothesis 字典
→ 统一文本格式
→ 构造标准化字典
→ 转换成稳定 JSON
→ SHA-256 哈希
→ Hypothesis Signature
'''

def _normalize_text(value: Any) -> str:
    return " ".join(str(value or "").split()).casefold()


def hypothesis_signature(hypothesis: dict[str, Any]) -> str:
    concepts = hypothesis.get("required_concepts")
    normalized_concepts = (
        sorted(_normalize_text(concept) for concept in concepts)
        if isinstance(concepts, list)
        else []
    )
    canonical = {
        "title": _normalize_text(hypothesis.get("title")),
        "claim": _normalize_text(hypothesis.get("claim")),
        "economic_rationale": _normalize_text(hypothesis.get("economic_rationale")),
        "prediction_direction": _normalize_text(hypothesis.get("prediction_direction")),
        "prediction_horizon_days": hypothesis.get("prediction_horizon_days"),
        "required_concepts": normalized_concepts,
    }
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
