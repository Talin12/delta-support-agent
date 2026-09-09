# Delta support agent — intent, grounded reply, and escalation

An AI support agent for **Delta Air Lines**, built from the [Customer Support on
Twitter](https://www.kaggle.com/datasets/thoughtvector/customer-support-on-twitter)
dataset. It classifies an incoming customer message into one of ten data-derived
intents, drafts a reply grounded in how Delta historically resolved similar cases,
and decides whether to auto-send or escalate — with a stated reason.

The system is the easy half. Most of the work here is the evidence that it is or
is not good enough to trust, which is what the rest of this document is about.

---

## Quickstart

Reproduces the headline numbers in under 15 minutes.

```bash
make setup
```

```bash
make data
```

```bash
make threads
```

```bash
make baselines
```

`make baselines` needs **no API key at all** — both baselines are LLM-free by
design. To run the full agent you need a key; copy `.env.example` to `.env` and set
`GEMINI_API_KEY` (free, no credit card: <https://aistudio.google.com/apikey>).

```bash
make judge
```

Every LLM response is cached in `cache/llm_cache.sqlite`, which is committed. Set
`LLM_OFFLINE=1` in `.env` to replay the cache and reproduce every number in this
report **with no API key and no network calls**.

### Free-tier note

Google's free tier meters requests *per minute, per project, per model* (15 RPM at
time of writing). The pipeline therefore runs its three stages on three different
models and rate-limits each independently — see `_RateLimiter` in `src/llm.py`.
This also means the judge runs on a different model from the generator, which is
desirable for independent reasons (below).

---

## 1. Problem framing

### Why Delta

Chosen on **reply substance, not volume**. AmazonHelp (170k replies) and
AppleSupport (107k) are larger, but 52% of Apple's replies and 82% of T-Mobile's are
some variant of "please DM us". Grounding a reply generator on a brand that mostly
deflects makes the task degenerate — the agent learns to emit "DM us", scores well,
and proves nothing. Delta has 42k replies, only 16% DM-deflection, and its
historical replies contain real substance. Full comparison in
`artifacts/brand_survey.csv`.

### What "good" means for Delta

Delta's public Twitter support is not a resolution channel. It is a **triage and
containment channel**: acknowledge fast, take the specifics private, and stop a
frustrated customer from becoming a public incident. Almost nothing is actually
resolved in public — the resolution happens in DMs the dataset does not contain.

That shapes what "good" has to mean:

1. **Never commit the brand to something it cannot honour.** A promised refund, a
   guaranteed timeline, or an invented policy is worse than a slow reply.
2. **Sound like Delta.** Delta's voice is warm, apologetic, first-person, and signs
   off with agent initials. A generic assistant voice reads as a bot and invites the
   "I want a human" spiral.
3. **Get the handoff right.** The expensive failure is not a mediocre reply. It is
   auto-sending a template to someone alleging a safety incident or threatening
   legal action.
4. **Automate enough to matter.** A system that escalates everything is perfectly
   safe and worthless. Volume handled without a human is the only reason to build
   this.

Goals 3 and 4 are in direct tension, and the evaluation is designed around that
trade-off rather than around a single accuracy number.

### What I chose not to build

- **Multi-turn dialogue.** The agent handles first contact only. Most Delta threads
  are two turns, and the conversations that continue mostly move to DM, which is
  absent from the data. Building a dialogue manager against data that stops at the
  handoff would be fitting to an artefact.
- **Live tool calls / account lookup.** No PNR lookup, bag trace, or rebooking. The
  dataset has no such systems, so any tool layer would be mocked, and mocked tools
  would make escalation look far more solvable than it is. Instead, "needs
  account-specific action" is treated as a *reason to escalate*, which is the honest
  version of the same constraint.
- **Fine-tuning.** With 220 labelled examples, prompting plus retrieval dominates. A
  fine-tune would spend the label budget on the wrong thing.
- **Sentiment/urgency scoring as separate models.** Tested informally and folded
  into the routing decision instead. Two more numbers nobody acts on is not a
  feature.
- **A production serving layer.** No API, queue, or dashboard. Nothing about
  throughput was in question.

---

## 2. System design

```
customer message
      │
      ▼
┌─────────────────┐   closed-set problem, cheap ground truth
│  1. classify    │   → intent + self-reported confidence
└─────────────────┘
      │
      ▼
┌─────────────────┐   TF-IDF kNN over 25,390 historical Delta threads
│  2. retrieve    │   → k=4 (customer message → what Delta replied)
└─────────────────┘   golden threads excluded, so no leakage
      │
      ▼
┌─────────────────┐   grounded in retrieved cases
│  3. draft+route │   → reply + AUTO/ESCALATE + reason
└─────────────────┘
      │
      ▼
┌─────────────────┐   regex rules the model cannot overrule
│  4. policy      │   → final route
└─────────────────┘
```

**Why three stages and not one prompt.** They fail differently and need to be
measured differently. Fusing them would make it impossible to attribute a failure to
a stage. It also lets the cheap stage run on a cheap model.

**Why rules sit above the model.** Safety, legal, fraud, and money cases are rare
enough that they barely move an accuracy number and severe enough that one mistake
costs more than every routine ticket combined. An LLM router is fine on the median
ticket and unreliable exactly on that tail. So those cases are decided by rules the
model cannot overrule, and the model decides only what the rules abstain on. The
cost is false positives, quantified in the failure analysis — and they are not
hypothetical: the rules fire on the word "injured" in a message of pure praise.

### The intent taxonomy

Ten intents, derived from the data in two stages (`src/discover.py`).

First, TF-IDF + KMeans over 6,000 first-contact messages (k=16, deliberately
over-clustered). Then — and this is the important part — **the clusters were treated
as evidence, not as the answer.** Lexical clustering splits on shared vocabulary,
and support cares about what must happen next. Cluster 4 ("crew") and cluster 9
("gate") each contain both effusive praise and furious complaints, because they
share every content word. The taxonomy was therefore cut along the axis of *what the
agent must do*, informed by what the clusters showed the traffic is about.

| Intent | What it means |
|---|---|
| `flight_disruption` | Delay, cancellation, diversion, missed connection |
| `baggage_issue` | Bag lost, delayed, damaged, or contents missing |
| `booking_change` | Change, cancel, or rebook an existing reservation |
| `refund_or_compensation` | Money or its equivalent back |
| `seat_or_upgrade` | Seat assignment, seat change, upgrade, standby list |
| `loyalty_account` | SkyMiles balance, status, redemption, account access |
| `policy_question` | Pre-travel "am I allowed / how do I" with no current problem |
| `service_complaint` | Staff conduct, service quality, or being unable to reach support |
| `praise` | Compliments and shout-outs with no outstanding request |
| `other` | Chatter, photos, aviation enthusiasm, too vague to act on |

`refund_or_compensation` **was added during hand-labelling, not before it.**
Clustering never surfaced it, because refund language is spread thinly across every
topic (delays, bags, Wi-Fi, seats, miles) and so never dominates a centroid. Labelling
made it unmissable: it is 16% of the golden set and the highest-stakes thing the
agent can get wrong, since it commits real money. It is the clearest evidence for
why unsupervised structure was treated as a hypothesis rather than a result.

Full definitions, boundary rules, and worked examples: `data/golden/taxonomy.json`.

---

## 3. The golden set

**220 hand-labelled examples**, each with an intent, a binary route, and a written
reason. Protocol in full: `data/golden/codebook.md`. Labels in
`scripts/apply_labels.py`, kept in source so every one is reviewable and diffable.

### Sampling

Two strata, because one sample cannot answer both questions:

| Stratum | n | Draw | Purpose |
|---|---|---|---|
| `representative` | 120 | Uniform random | The **only** slice a production estimate may be quoted from |
| `enriched` | 100 | Oversampled tail | Measuring the rare cases that cause real harm |

The enriched stratum oversamples rule-triggering messages (40), long threads (25),
messages with no close historical precedent (20), and very short messages (15). At a
~1.8% base rate, a uniform sample of 220 would contain roughly four safety-critical
messages — not enough to say anything about the failure mode most likely to hurt.

The strata stay tagged, and every metric is reported per-stratum. **Escalation base
rate is 15.0% representative vs 45.0% enriched.** Pooling them gives 28.6%, a number
that describes no real inbox.

### Routing definition

The question is precise: **could this reply be sent publicly with no human reading
it first?**

This framing matters. Delta's real first move on most tickets is a templated "DM us
your confirmation number", and that template is genuinely safe to auto-send for a
routine bag delay. Defining escalation as *difficulty* would escalate almost
everything. Defining it as *sendability* isolates the cases where that same template
is actively harmful — a legal threat, a safety allegation, a viral complaint — which
is the decision a support lead actually makes.

Nine escalation conditions (E1–E9) are enumerated in the codebook. Three boundary
rules were added mid-labelling when the first pass proved inconsistent: ambiguity
alone does not escalate (a clarifying question is itself a safe reply); a defection
threat alone does not escalate; initiating a claim is AUTO while chasing a stalled
one is ESCALATE.

### Known weaknesses

- **Single annotator.** No inter-annotator agreement figure exists, so the
  boundaries most likely to be contested (`service_complaint` vs
  `flight_disruption`, `policy_question` vs `booking_change`) carry unmeasured label
  noise. This is the single biggest weakness of the evaluation.
- **Labeller saw Delta's actual reply**, which grounds route labels in real
  operating posture at the cost of hindsight bias. Deliberate, and disclosed.
- **2017 Twitter data.** Delta's policies, tone, and channel mix have moved on.
- **Public tweets only.** The DMs where issues were really resolved are absent, so
  "how the brand resolved it" is truncated to the public opening move.

---

## 4. Results

*(populated by the evaluation run — see `artifacts/results.json`)*

---

## 5. Repo map

```
src/llm.py         provider-agnostic client, disk cache, per-model rate limiter
src/prep.py        twcs.csv -> per-brand threads (timestamp-ordered)
src/discover.py    cluster-based intent discovery
src/retrieve.py    TF-IDF kNN over historical resolutions, leakage-controlled
src/policy.py      deterministic escalation rules
src/agent.py       classify -> ground -> draft & route
src/baselines.py   trivial and simple baselines
eval/harness.py    metrics, bootstrap CIs, per-stratum reporting
eval/judge.py      LLM-as-judge rubric + agreement statistics
data/golden/       taxonomy, codebook, 220 labelled examples
reports/DECISIONS.md   the 20 non-obvious calls and why
```

---

## 6. Citations and borrowed work

- **Dataset**: Thought Vector, *Customer Support on Twitter* (Kaggle, CC BY-NC-SA
  4.0). Retrieved via the HuggingFace mirror `SunidhiSriram/twcs`, verified
  byte-identical in size and schema to the Kaggle original.
- **Quadratic-weighted Cohen's kappa**: standard formulation (Cohen 1968);
  implemented from scratch in `eval/judge.py` rather than pulled from a library, so
  the weighting scheme is explicit and auditable.
- **Bootstrap confidence intervals**: standard percentile bootstrap (Efron 1979).
- **scikit-learn** for TF-IDF, KMeans, and logistic regression.
- **LLM-as-judge with a written rubric and human agreement validation** follows the
  now-standard pattern from the MT-Bench / Chatbot Arena line of work (Zheng et al.,
  2023); the self-preference-bias concern that motivates using a different judge
  model comes from the same literature.
- Prompts, taxonomy, labels, codebook, metric choices, and all analysis in this
  report are my own.
