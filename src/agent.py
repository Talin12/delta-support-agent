"""The support agent: classify -> ground -> draft and route.

Three stages rather than one prompt, because they fail differently and need to be
measured separately. Classification is a closed-set problem with a cheap ground
truth; drafting is open-ended and needs a judge; routing is a cost-asymmetric
decision that deserves a rule layer above the model. Fusing them into one call
would make it impossible to say which stage a failure came from.
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path

import llm
import policy
from retrieve import ResolutionIndex, format_exemplars

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "prompts"


def load_prompt(name: str) -> str:
    return (PROMPTS / name).read_text(encoding="utf-8")


def load_taxonomy(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def format_taxonomy(taxonomy: dict) -> str:
    lines = []
    for intent in taxonomy["intents"]:
        lines.append(f"- {intent['name']}: {intent['description']}")
        if intent.get("boundary"):
            lines.append(f"    boundary: {intent['boundary']}")
    return "\n".join(lines)


@dataclass
class AgentOutput:
    message: str
    intent: str
    intent_confidence: float
    intent_rationale: str
    reply: str
    route: str
    route_reason: str
    route_source: str
    overrode_model: bool
    model_route: str
    exemplar_thread_ids: list[int] = field(default_factory=list)
    exemplar_scores: list[float] = field(default_factory=list)
    error: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SupportAgent:
    def __init__(
        self,
        brand: str,
        taxonomy: dict,
        index: ResolutionIndex,
        *,
        k_exemplars: int = 4,
        confidence_floor: float = 0.55,
    ):
        self.brand = brand
        self.taxonomy = taxonomy
        self.index = index
        self.k = k_exemplars
        self.confidence_floor = confidence_floor
        self.classify_tmpl = load_prompt("classify.txt")
        self.draft_tmpl = load_prompt("draft_and_route.txt")
        self._intent_names = {i["name"] for i in taxonomy["intents"]}
        self._fallback = taxonomy.get("catch_all", taxonomy["intents"][-1]["name"])
        self._desc = {i["name"]: i["description"] for i in taxonomy["intents"]}

    def classify(self, message: str) -> tuple[str, float, str]:
        prompt = self.classify_tmpl.format(
            brand=self.brand, taxonomy=format_taxonomy(self.taxonomy), message=message
        )
        try:
            out = llm.complete_json(prompt, temperature=0.0, max_tokens=2000, model_tag="classify",
                                     model_override=os.environ.get("CLASSIFY_MODEL", ""))
        except Exception:
            return self._fallback, 0.0, "classifier error"
        intent = str(out.get("intent", "")).strip()
        if intent not in self._intent_names:
            return self._fallback, 0.0, f"model returned unknown intent {intent!r}"
        try:
            conf = float(out.get("confidence", 0.0))
        except (TypeError, ValueError):
            conf = 0.0
        return intent, max(0.0, min(1.0, conf)), str(out.get("rationale", ""))[:300]

    def draft_and_route(self, message: str, intent: str) -> tuple[str, str, str, list]:
        exemplars = self.index.search(message, k=self.k)
        prompt = self.draft_tmpl.format(
            brand=self.brand,
            intent=intent,
            intent_description=self._desc.get(intent, ""),
            exemplars=format_exemplars(exemplars),
            message=message,
        )
        try:
            out = llm.complete_json(prompt, temperature=0.0, max_tokens=3000, model_tag="draft",
                                     model_override=os.environ.get("DRAFT_MODEL", ""))
        except Exception as exc:
            return "", policy.ESCALATE, f"drafting failed: {exc}", exemplars
        reply = str(out.get("reply", "")).strip()
        route = str(out.get("route", "")).strip().upper()
        if route not in (policy.AUTO, policy.ESCALATE):
            route = policy.ESCALATE
        reason = str(out.get("reason", "")).strip()[:400]
        return reply, route, reason, exemplars

    def handle(self, message: str) -> AgentOutput:
        intent, confidence, rationale = self.classify(message)
        reply, model_route, model_reason, exemplars = self.draft_and_route(message, intent)
        decision = policy.apply(
            message,
            model_route,
            model_reason,
            confidence,
            confidence_floor=self.confidence_floor,
        )
        return AgentOutput(
            message=message,
            intent=intent,
            intent_confidence=confidence,
            intent_rationale=rationale,
            reply=reply,
            route=decision["route"],
            route_reason=decision["reason"],
            route_source=decision["source"],
            overrode_model=decision["overrode_model"],
            model_route=model_route,
            exemplar_thread_ids=[e.thread_id for e in exemplars],
            exemplar_scores=[e.score for e in exemplars],
        )
