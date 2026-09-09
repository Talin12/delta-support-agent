"""Baselines the agent has to beat. Neither uses an LLM.

trivial - majority intent, one canned reply, auto-send everything. Shows how much
          of any accuracy number is just class imbalance.
simple  - TF-IDF + logistic regression for intent, nearest historical reply copied
          verbatim, rules-only routing. Costs nothing per ticket, so if the agent
          can't clear it the LLM isn't paying rent.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

import policy
from agent import AgentOutput
from retrieve import ResolutionIndex


CANNED_REPLY = "Sorry to hear about this! Please DM us the details and we'll take a look."


@dataclass
class TrivialBaseline:
    """Majority class everywhere. The floor."""

    majority_intent: str
    canned_reply: str = CANNED_REPLY
    always_route: str = policy.AUTO
    name: str = "trivial"

    def handle(self, message: str) -> AgentOutput:
        return AgentOutput(
            message=message,
            intent=self.majority_intent,
            intent_confidence=1.0,
            intent_rationale="majority class",
            reply=self.canned_reply,
            route=self.always_route,
            route_reason="trivial baseline auto-sends everything",
            route_source="baseline",
            overrode_model=False,
            model_route=self.always_route,
        )

    @classmethod
    def fit(cls, intents: list[str]) -> "TrivialBaseline":
        return cls(majority_intent=Counter(intents).most_common(1)[0][0])


class SimpleBaseline:
    """Learned intent + retrieved reply + rules-only routing. No LLM at inference."""

    name = "simple"

    def __init__(self, index: ResolutionIndex, seed: int = 13, max_false_auto: float = 0.10):
        self.index = index
        self.seed = seed
        self.max_false_auto = max_false_auto
        self.clf: Pipeline | None = None
        # Calibrated in fit(). A logistic regression over 10 classes produces max
        # probabilities in a completely different range from an LLM's self-reported
        # confidence, so reusing the agent's 0.55 floor here would escalate
        # essentially everything and make the baseline a straw man.
        self.confidence_floor = 0.55

    def calibrate(self, messages: list[str], routes: list[str]) -> None:
        """Pick the lowest confidence floor whose train false-auto rate stays in budget.

        Lower floor means more automation and more risk. We take the most automation
        available subject to the false-auto constraint. Tuned on training data only;
        the fold's held-out examples never inform this threshold.
        """
        assert self.clf is not None
        confidences = self.clf.predict_proba(messages).max(axis=1)
        best = 1.01
        for threshold in [i / 100 for i in range(5, 101, 1)]:
            false_auto = escalate_needed = 0
            for msg, conf, truth in zip(messages, confidences, routes):
                rule_fired = policy.check_rules(msg)[0] is not None
                predicted = (
                    policy.ESCALATE if (rule_fired or conf < threshold) else policy.AUTO
                )
                if truth == policy.ESCALATE:
                    escalate_needed += 1
                    if predicted == policy.AUTO:
                        false_auto += 1
            rate = (false_auto / escalate_needed) if escalate_needed else 0.0
            if rate <= self.max_false_auto:
                best = threshold
                break
        self.confidence_floor = best

    def fit(
        self, messages: list[str], intents: list[str], routes: list[str] | None = None
    ) -> "SimpleBaseline":
        self.clf = Pipeline(
            [
                (
                    "tfidf",
                    TfidfVectorizer(
                        ngram_range=(1, 2),
                        min_df=2,
                        sublinear_tf=True,
                        stop_words="english",
                        max_features=30000,
                    ),
                ),
                (
                    "lr",
                    LogisticRegression(
                        max_iter=2000, class_weight="balanced", random_state=self.seed
                    ),
                ),
            ]
        )
        self.clf.fit(messages, intents)
        if routes is not None:
            self.calibrate(messages, routes)
        return self

    def handle(self, message: str) -> AgentOutput:
        assert self.clf is not None, "call fit() first"
        intent = self.clf.predict([message])[0]
        confidence = float(self.clf.predict_proba([message]).max())

        neighbours = self.index.search(message, k=1)
        reply = neighbours[0].brand_reply if neighbours else CANNED_REPLY

        rule_name, rule_reason = policy.check_rules(message)
        if rule_name:
            route, reason, source = policy.ESCALATE, rule_reason, f"rule:{rule_name}"
        elif confidence < self.confidence_floor:
            route, reason, source = (
                policy.ESCALATE,
                f"classifier confidence {confidence:.2f} below calibrated floor "
                f"{self.confidence_floor:.2f}",
                "rule:low_confidence",
            )
        else:
            route, reason, source = (
                policy.AUTO,
                "no escalation rule fired and the classifier was confident",
                "rule:default_auto",
            )

        return AgentOutput(
            message=message,
            intent=intent,
            intent_confidence=confidence,
            intent_rationale="tfidf+logreg",
            reply=reply,
            route=route,
            route_reason=reason,
            route_source=source,
            overrode_model=False,
            model_route=route,
            exemplar_thread_ids=[n.thread_id for n in neighbours],
            exemplar_scores=[n.score for n in neighbours],
        )
