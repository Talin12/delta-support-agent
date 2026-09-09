"""Deterministic escalation rules that sit above the model's own judgement.

An LLM router is fine on the median ticket and unreliable on the tail that actually
hurts: account compromise, legal threats, safety. Those cases are rare enough that
they barely move an accuracy number and severe enough that getting one wrong costs
more than every routine ticket combined. So they are handled by rules the model
cannot overrule, and the model only decides the cases the rules abstain on.

Each rule is deliberately high-precision. A rule firing means ESCALATE; no rule
firing means nothing - the model still gets to escalate on its own judgement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

AUTO = "AUTO"
ESCALATE = "ESCALATE"


@dataclass(frozen=True)
class Rule:
    name: str
    pattern: re.Pattern
    reason: str


def _rx(pattern: str) -> re.Pattern:
    return re.compile(pattern, re.IGNORECASE)


# Ordered by severity: the first match wins, so the stated reason is the worst one.
RULES: list[Rule] = [
    Rule(
        "self_harm_or_safety",
        _rx(r"\b(kill myself|suicide|suicidal|self[- ]harm|end my life|want to die)\b"),
        "Message contains self-harm or immediate-safety language; requires a trained human immediately.",
    ),
    Rule(
        "physical_safety",
        _rx(
            r"\b(injur(y|ed)|assault(ed)?|attack(ed)?|unsafe|danger(ous)?|harass(ed|ment)?|"
            r"discriminat(ed|ion)|racist|threat(ened)?)\b"
        ),
        "Alleges a safety, harassment, or discrimination incident; carries duty-of-care and reputational risk.",
    ),
    Rule(
        "legal_or_regulator",
        _rx(
            r"\b(lawsuit|sue|suing|lawyer|attorney|solicitor|legal action|small claims|"
            r"ombudsman|regulator|FCA|CFPB|BBB complaint|court)\b"
        ),
        "Raises legal action or a regulator; anything sent becomes part of a formal record.",
    ),
    Rule(
        "account_security",
        _rx(
            r"\b(hacked|compromised|unauthori[sz]ed|fraud(ulent)?|stolen|identity theft|"
            r"someone (else )?(used|accessed|logged into)|didn'?t make this (charge|purchase|order))\b"
        ),
        "Possible account compromise or fraud; needs identity verification no public agent can perform.",
    ),
    Rule(
        "explicit_human_request",
        _rx(
            r"\b(speak to (a|an) (human|person|manager|supervisor|agent)|real person|"
            r"talk to (a|an) (human|manager|supervisor)|get me a manager|stop the bot|not a bot)\b"
        ),
        "Customer explicitly asked for a human; auto-replying here reliably escalates anger.",
    ),
    Rule(
        "money_movement",
        _rx(
            r"\b(refund|chargeback|compensat(e|ion)|reimburse(ment)?|double[- ]charged|"
            r"charged twice|overcharged|money back)\b"
        ),
        "Involves moving money or granting compensation; commits the brand to a cost and needs authorisation.",
    ),
    Rule(
        "media_or_virality",
        _rx(r"\b(journalist|reporter|press|going viral|@BBC|@CNN|news story)\b"),
        "Press or virality signal; replies here are read far beyond the customer.",
    ),
]


def check_rules(message: str) -> tuple[str | None, str | None]:
    """Return (rule_name, reason) for the first rule that fires, else (None, None)."""
    for rule in RULES:
        if rule.pattern.search(message):
            return rule.name, rule.reason
    return None, None


def apply(message: str, model_decision: str, model_reason: str, confidence: float,
          *, confidence_floor: float = 0.55) -> dict:
    """Combine rules, model judgement, and a confidence floor into a final route.

    Precedence: hard rules > low confidence > model judgement.
    """
    rule_name, rule_reason = check_rules(message)
    if rule_name:
        return {
            "route": ESCALATE,
            "reason": rule_reason,
            "source": f"rule:{rule_name}",
            "overrode_model": model_decision == AUTO,
        }
    if model_decision == AUTO and confidence < confidence_floor:
        return {
            "route": ESCALATE,
            "reason": (
                f"Intent confidence {confidence:.2f} is below the {confidence_floor:.2f} floor; "
                "an unclassified message is not safe to answer automatically."
            ),
            "source": "rule:low_confidence",
            "overrode_model": True,
        }
    return {
        "route": model_decision,
        "reason": model_reason,
        "source": "model",
        "overrode_model": False,
    }
