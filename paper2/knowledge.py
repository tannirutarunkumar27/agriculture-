"""Evidence-gated, class-aligned soft guidance. No approved production rules yet.

Consumes ORIGINAL units and observation masks, never standardized/imputed values.
Approval fields are an auditable declaration, not automated scientific validation.
"""
from __future__ import annotations

import copy
from datetime import date
import math
import re
from urllib.parse import urlparse

import torch
from torch import nn

from paper2.data import object_hash
from paper2.federated import make_model

SCOPE = ("region", "season", "stage", "water_regime")
DESCRIPTOR = ("unit", "measurement", "period")


def _text(value):
    return isinstance(value, str) and bool(value.strip())


def _number(value):
    return type(value) in (int, float) and math.isfinite(value)


def _digest(value):
    return isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value) is not None


class RuleBank:
    """Validated declarations with immutable-by-copy metadata and identity.

    Only exact descriptor matching is supported; no implicit unit conversions.
    Scope '*' requires explicitly reviewed broad applicability in the source.
    Four-point bounds must be documented, not invented from min/max guidance.
    """

    def __init__(self, registry, feature_contract, classes):
        if len(classes) < 2 or len(set(classes)) != len(classes) or not all(map(_text, classes)):
            raise ValueError("Classes must be unique ordered labels")
        if registry.get("schema_version") != 1 or not isinstance(registry.get("rules"), list):
            raise ValueError("Unsupported rule registry")
        if not _text(feature_contract.get("dataset")) or not _digest(feature_contract.get("source_sha256")):
            raise ValueError("Dataset/source identity required")
        features = feature_contract.get("features", [])
        if not features or len({f.get("name") for f in features}) != len(features):
            raise ValueError("Unique ordered raw feature contract required")
        if not all(_text(f.get("name")) for f in features):
            raise ValueError("Invalid feature name")
        indices = {f["name"]: i for i, f in enumerate(features)}
        rules, ids = [], set()
        for rule in registry["rules"]:
            if not _text(rule.get("id")) or rule["id"] in ids:
                raise ValueError("Duplicate/invalid rule id")
            ids.add(rule["id"])
            if rule.get("crop") not in classes or rule.get("feature") not in indices:
                raise ValueError("Rule crop/feature not bound to the exact schema")
            feature = features[indices[rule["feature"]]]
            if feature.get("verified") is not True or not _text(feature.get("evidence")):
                raise ValueError("Dataset units/measurement provenance unverified")
            if rule.get("dataset") != feature_contract["dataset"]:
                raise ValueError("Rule approval is for another dataset")
            if any(not _text(feature.get(k)) or feature[k] != rule.get(k) for k in DESCRIPTOR):
                raise ValueError("Rule units/measurement/period differ; implicit conversion forbidden")
            bounds = rule.get("bounds")
            if not isinstance(bounds, list) or len(bounds) != 4 or not all(map(_number, bounds)):
                raise ValueError("Four finite source-documented soft bounds required")
            a, b, c, d = bounds
            if not a < b <= c < d:
                raise ValueError("Invalid trapezoidal bounds")
            if not _number(rule.get("weight")) or not 0 < rule["weight"] <= 1:
                raise ValueError("Confidence weight must be in (0,1]")
            source = rule.get("source", {})
            if urlparse(source.get("url", "")).scheme != "https" or not urlparse(source.get("url", "")).netloc:
                raise ValueError("HTTPS source reference required")
            if not all(_text(source.get(k)) for k in ("version", "locator", "scientific_name", "bounds_evidence")):
                raise ValueError("Per-rule source/taxonomy/bounds evidence missing")
            if source.get("independent_of_labels") is not True or not _digest(source.get("artifact_sha256")):
                raise ValueError("Independent archived source evidence required")
            scope = rule.get("scope", {})
            if set(scope) != set(SCOPE) or not all(_text(scope.get(k)) for k in SCOPE):
                raise ValueError("Explicit regional/season/stage/water scope required")
            approval = rule.get("approval", {})
            if approval.get("status") != "approved" or not all(_text(approval.get(k)) for k in ("reviewer", "rationale", "exceptions", "scope_evidence")):
                raise ValueError("Documented scientific approval required")
            try:
                date.fromisoformat(approval["checked_date"])
            except (ValueError, KeyError, TypeError) as exc:
                raise ValueError("ISO approval check date required") from exc
            rules.append(copy.deepcopy(rule))
        self._rules = tuple(rules)
        self._contract = copy.deepcopy(feature_contract)
        self.classes = tuple(classes)
        self.features = tuple(f["name"] for f in features)
        self.identity = object_hash({"registry": registry, "contract": feature_contract, "classes": classes})

    @property
    def rule_count(self):
        return len(self._rules)

    def metadata(self):
        return {"identity": self.identity, "classes": list(self.classes),
                "feature_contract": copy.deepcopy(self._contract), "rules": copy.deepcopy(list(self._rules))}

    def suitability(self, raw, observed, contexts):
        if raw.ndim != 2 or raw.shape[1] != len(self.features) or not raw.is_floating_point():
            raise ValueError("Ordered raw floating-point feature matrix required")
        if observed.shape != raw.shape or observed.dtype != torch.bool or observed.device != raw.device:
            raise ValueError("Explicit original-observation mask required")
        if torch.isinf(raw).any() or not torch.isfinite(raw[observed]).all():
            raise ValueError("Observed raw inputs must be finite; unobserved NaN allowed")
        if not isinstance(contexts, (list, tuple)) or len(contexts) != len(raw):
            raise ValueError("One scope context per sample required")
        if any(not isinstance(ctx, dict) or set(ctx)-set(SCOPE) or any(not _text(v) for v in ctx.values()) for ctx in contexts):
            raise ValueError("Invalid sample scope metadata")
        numerator = raw.new_zeros((len(raw), len(self.classes)))
        denominator = torch.zeros_like(numerator)
        for rule in self._rules:
            j, crop = self.features.index(rule["feature"]), self.classes.index(rule["crop"])
            applicable = torch.tensor([all(rule["scope"][k] == "*" or ctx.get(k) == rule["scope"][k]
                                          for k in SCOPE) for ctx in contexts], device=raw.device)
            valid = observed[:, j] & applicable
            value = torch.where(valid, raw[:, j], raw.new_zeros(len(raw)))
            a, b, c, d = rule["bounds"]
            score = torch.minimum((value-a)/(b-a), (d-value)/(d-c)).clamp(0, 1)
            weight = valid.to(raw.dtype)*rule["weight"]
            numerator[:, crop] += score*weight
            denominator[:, crop] += weight
        covered = denominator > 0
        q = torch.where(covered, numerator/denominator.clamp_min(torch.finfo(raw.dtype).tiny),
                        torch.full_like(numerator, 0.5))
        return q, covered


class KnowledgeGuidedMLP(nn.Module):
    """z'_c = z_c + alpha(q_c-.5); fixed public bank, no new optimizer.

    State dict contains neural weights ONLY. Persist/check manifest alongside it.
    """

    def __init__(self, inputs, hidden, bank, alpha=0.0):
        super().__init__()
        if not isinstance(bank, RuleBank) or not _number(alpha) or alpha < 0:
            raise ValueError("Validated bank and finite nonnegative alpha required")
        if alpha > 0 and not bank.rule_count:
            raise ValueError("Nonzero guidance blocked: no approved compatible rules")
        self.model = make_model(inputs, len(bank.classes), hidden)
        self.bank, self.alpha = bank, float(alpha)

    def manifest(self):
        return {"architecture": "class-aligned-soft-bias-mlp-v1", "alpha": self.alpha,
                "knowledge": self.bank.metadata(), "guidance_enabled": self.alpha > 0,
                "privacy": "NONE", "rule_validation": "review declarations, not automated source authentication"}

    def forward(self, x, raw, observed, contexts):
        if x.ndim != 2 or len(x) != len(raw) or not torch.isfinite(x).all():
            raise ValueError("Finite model inputs aligned with raw rows required")
        q, _ = self.bank.suitability(raw, observed, contexts)
        logits = self.model(x)
        if logits.device != q.device or logits.dtype != q.dtype:
            raise ValueError("Model and raw tensors must have matching dtype/device")
        return logits if self.alpha == 0 else logits + self.alpha*(q-0.5)
