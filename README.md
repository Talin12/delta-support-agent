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

Google's free tier meters requests **per minute *and* per day, per project, per
model**, and the limits differ enormously between models: the full `flash` models
allow only **20 requests/day**, while `flash-lite` models allow **500/day** at 15
RPM. Two consequences are baked into `src/llm.py`:

- `_RateLimiter` throttles each *(model, key)* pair independently, since that is the
  granularity the quota actually uses.
- `GEMINI_API_KEYS` accepts a comma-separated list and round-robins across them.
  Keys from different Google Cloud projects carry genuinely separate budgets, so
  three project keys give roughly 3× the daily headroom. Rotation also fails over
  when one key is exhausted.

**Watch for model aliases.** `gemini-flash-lite-latest` and `gemini-3.5-flash-lite`
are the same model under two names and share one quota — which is how I discovered
that a judge I believed was independent had been grading its own generator's output.
Verify independence empirically rather than trusting the name.

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

All numbers from `artifacts/results.json`, regenerated by `make judge`. Brackets are
95% bootstrap confidence intervals. Judging runs on the 120-example representative
slice so all three systems share one judge model (see Decision 22).

### 4.1 Intent classification

| System | Accuracy (representative) | Macro-F1 (repr.) | Accuracy (pooled) | Macro-F1 (pooled) |
|---|---|---|---|---|
| Trivial | 22.5% <sub>[15.0, 30.0]</sub> | 0.037 | 22.3% <sub>[16.8, 28.2]</sub> | 0.036 |
| Simple (no LLM) | 35.0% <sub>[26.7, 43.3]</sub> | 0.303 | 41.8% <sub>[35.5, 49.1]</sub> | 0.357 |
| **Agent** | **81.7%** <sub>[74.2, 88.3]</sub> | **0.797** | **81.8%** <sub>[76.8, 86.8]</sub> | **0.822** |

The trivial baseline's 22.5% is the majority-class rate and its macro-F1 of 0.037
shows what that accuracy is actually worth. The gap between the agent and the simple
baseline (81.7% vs 35.0%) is far wider than either confidence interval, so this one
is a real result rather than noise.

### 4.2 Routing

Representative slice (n=120) — the only production-quotable stratum.

| System | Automation rate | False-auto rate | Needless escalation | Safe automation |
|---|---|---|---|---|
| Trivial | 100.0% | **100.0%** (18) | 0.0% (0) | 85.0% <sub>[78.3, 90.8]</sub> |
| Simple (no LLM) | 5.8% | 5.6% (1) | 94.1% (96) | 5.0% <sub>[1.7, 9.2]</sub> |
| Agent | 67.5% | 22.2% (4) | 24.5% (25) | 64.2% <sub>[55.0, 72.5]</sub> |

Enriched slice (n=100), where the hard cases live.

| System | Automation rate | False-auto rate | Needless escalation |
|---|---|---|---|
| Trivial | 100.0% | 100.0% (45) | 0.0% (0) |
| Simple (no LLM) | 5.0% | 0.0% (0) | 90.9% (50) |
| Agent | 38.0% | 4.4% (2) | 34.5% (19) |

The two baselines fail in exactly opposite directions, which is the point of
including both. The trivial baseline automates everything and therefore auto-sends
**100%** of the cases that needed a human. The simple baseline is nearly perfectly
safe and automates 5.8% of traffic, which is worth approximately nothing. The agent
sits between them, and the honest reading is that it buys automation with risk: 67.5%
automation costs 4 auto-sent cases that a human should have seen.

Note the agent escalates *more* on the enriched slice (62%) than the representative
one (32.5%). It is responding to difficulty rather than escalating at a fixed rate,
which is the behaviour you want, though the mechanism is mostly the rule layer firing.

### 4.3 Reply quality (blind LLM judge)

| System | Sendable rate | Grounded | Addresses | Voice | Actionable | Safe | Mean |
|---|---|---|---|---|---|---|---|
| Trivial | 43.3% <sub>[35.0, 52.5]</sub> | 3.92 | 2.31 | 2.79 | 3.83 | 4.94 | **3.56** |
| Simple (no LLM) | 16.7% <sub>[10.8, 23.3]</sub> | 2.52 | 2.19 | **4.92** | 2.44 | 4.86 | **3.39** |
| **Agent** | **80.0%** <sub>[72.5, 87.5]</sub> | 4.82 | 4.12 | 4.89 | 3.38 | 4.89 | **4.42** |

The simple baseline scores the **highest of any system on voice (4.92)** while
scoring 2.19 on addressing the problem. It copies a real Delta reply verbatim, so
the voice is literally perfect and the content belongs to a different customer. It is
the cleanest demonstration in this report of why a per-dimension average is a bad
headline: averaging that 4.92 with everything else produces 3.39, a number that
describes nothing real.

### 4.4 Useful automation (the joint metric)

Reply quality and routing scored **together**. An example counts only if the system
chose AUTO, the human label agreed AUTO, **and** the judge called the reply sendable.

| System | Useful automation | Harmful automation |
|---|---|---|
| Trivial | 32.5% (39/120) | **15.0%** (18/120) |
| Simple (no LLM) | 0.8% (1/120) | 0.8% (1/120) |
| **Agent** | **50.8%** (61/120) | **3.3%** (4/120) |

**This is the headline: the agent usefully handles about half of Delta's public
traffic without a human, while auto-sending something that needed one on 3.3% of all
traffic.** Both halves matter. The trivial baseline reaches 32.5% useful automation
only by also causing harm on 15% of everything — nearly five times the agent's rate.

---

## 5. Does the judge agree with a human?

Every number in §4.3 and §4.4 depends on the judge being a reasonable proxy for a
person. I scored 60 replies by hand — 20 from each system, shuffled and blind to
which system wrote them — against the same rubric the judge receives.

Agreement is quadratic-weighted Cohen's kappa (QWK), the right statistic for ordered
1–5 anchors because it penalises a 5-vs-1 disagreement far more than a 5-vs-4.
Full output: `artifacts/judge_agreement.json`, regenerated by `make agreement`.

| Dimension | QWK | Interpretation | Within ±1 | Human mean | Judge mean | Judge bias |
|---|---|---|---|---|---|---|
| `grounded` | 0.436 | moderate | 78% | 4.47 | 3.98 | −0.48 |
| `addresses` | 0.681 | substantial | 88% | 2.82 | 2.78 | −0.03 |
| `voice` | 0.733 | substantial | 97% | 3.98 | 4.25 | +0.27 |
| `actionable` | 0.459 | moderate | 58% | 2.88 | 3.33 | +0.45 |
| `safe` | −0.038 | see below | 98% | 4.92 | 4.97 | +0.05 |

**`sendable`** (the binary that drives §4.4): **κ = 0.492 (moderate)**, raw agreement
**75%**. Human says sendable 61.7% of the time, judge 53.3%.
**Spearman on mean score: 0.624.**

### Reading this honestly

**`safe` is not broken — the statistic is.** A QWK below zero looks alarming, but ±1
agreement is 98% and both raters sit at ~4.9/5. Kappa is a chance-corrected measure,
and when one rating category dominates this heavily there is almost no variance for
it to explain, so it collapses toward zero or negative regardless of how closely the
raters track each other. This is the well-known kappa paradox (high agreement, low
kappa under skewed marginals). The honest conclusion is not "the judge cannot assess
safety" but **"this rubric dimension is uninformative as written"** — almost nothing
in Delta's traffic tempts either rater to call a reply unsafe, so the dimension
discriminates nothing and should be replaced with a targeted overpromise check.

**Where the judge is trustworthy.** `addresses` and `voice` reach substantial
agreement (0.68, 0.73). These are the dimensions doing real work in §4.3, and the
per-dimension picture there can be taken at face value.

**Where it is not.** `actionable` agrees within ±1 only 58% of the time — the judge is
half a point more generous than I am about what counts as a concrete next step. And
`grounded` at 0.436 with a −0.48 bias means the judge is *stricter* than me on
groundedness while agreeing only moderately on which replies deserve it.

**The judge is biased toward weak systems.** Per-system bias: agent −0.04, simple
−0.19, **trivial +0.38**. It over-rates the generic canned reply by nearly four tenths
of a point relative to a human, while scoring the agent almost exactly as I do. This
cuts *against* the agent's apparent margin — the true gap between the agent and the
trivial baseline is wider than §4.3 shows — but it also means absolute quality
numbers for weak systems are inflated.

**Bottom line.** With κ = 0.492 on `sendable` and Spearman 0.624 on mean score, this
judge is good enough to **rank** systems and not good enough to certify an absolute
quality level. Ranking agreement holds: my blind scores put the systems in the same
order as the judge (agent 90% sendable > trivial 65% > simple 30%). Every comparative
claim in §4 rests on that; no claim of the form "80% of replies are good enough to
send" should be taken from this judge alone.

---

## 6. Failure analysis

Top five, with real examples. Regenerate with
`python scripts/failure_analysis.py --system agent`.

### F1. It fabricates the initials of real Delta employees — in 91% of replies

200 of 220 agent replies end in a sign-off like `*TJF`, `*AMR`, `*HRO`. Delta agents
sign with their own initials, so these are invented attributions to specific real
people.

> **Customer:** "need help MCO is a mess never making this flight"
> **Agent:** "We are sorry MCO is a mess. Could you provide your confirmation number
> via DM so I can take a look for you? Thanks. **\*TDL**"

*Hypothesis:* the retrieved exemplars all carry signatures, and nothing in the prompt
distinguishes "match this voice" from "copy this identity". The model is doing
exactly what it was told. **The judge scored voice 4.89 — it rewarded the
impersonation as good brand voice.** This is the most serious defect found, and the
headline reply-quality metric actively encourages it.

### F2. It is blind to urgency

All 6 false-autos are time-critical cases. The agent classifies the topic correctly
with high confidence and auto-sends a template into a situation that needed a human
*now*.

> **Customer:** "A 2+ hour wait time 2 speak to a representative?! This delay may cost
> me the only chance I have to get to OH in time!"
> **Agent:** intent `service_complaint` (conf 0.95) → **AUTO**
> *"I'm sorry for the long wait times. How can I assist?"*

> **Customer:** "it's now 1:14am and no one called ahead to tell someone to be here
> when we landed to refuel us."
> **Agent:** → **AUTO**, *"My apologies for that delay with refueling at this hour."*

*Hypothesis:* the taxonomy encodes *topic*, and urgency is orthogonal to it. Both
messages look like ordinary complaints lexically. The model's stated reasons say the
reply "matches the standard pattern used in case 1" — retrieval found a similar-topic
case that was handled with a template, and nothing represents that this one is
happening in a closing window. A separate urgency signal, not a new intent, is the fix.

### F3. The deterministic rules misfire on surface keywords

5 needless escalations come from rules matching a word out of context.

> "they upgraded this **injured** gal to Comfort+ to be more accommodating. Fav
> airline." → `rule:physical_safety` → ESCALATE. It is pure praise.

> "**Sue** Cakana says thanks you to for sponsorship." → `rule:legal_or_regulator`
> → ESCALATE. "Sue" is a person's first name.

*Hypothesis:* regexes match tokens, not propositions. This is the acknowledged price
of putting rules above the model (Decision 11) and the trade is still right — 5
needless escalations is cheap insurance — but each rule should be individually
precision-scored rather than trusted.

### F4. `flight_disruption` and `service_complaint` blur into each other

The single largest intent confusion (7 of 40 errors), exactly the boundary flagged as
most contestable in the codebook before any results existed.

*Hypothesis:* this is partly genuine ambiguity rather than model error. "Delayed 3
times then cancelled, had to drive through the night" is both a disruption report and
a grievance, and my own codebook rule ("label what determines the first action") is
doing real work to separate them. With a single annotator I cannot tell how much of
this 7 is model error and how much is label noise — which is why a second annotator
is the top priority in §8.

### F5. A pipeline artifact leaks into customer-facing text

13 replies contain the literal string `<link>` — the placeholder my own text cleaning
substitutes for URLs. The model copies it out of the retrieved exemplars.

> "Please DM your confirmation number so I may look into the matter for you. Thanks!
> \*TJF **\<link\>**"

*Hypothesis:* preprocessing intended for the *index* leaked into the *prompt*.
Exemplars should be shown to the model with URLs stripped entirely rather than
replaced by a token that looks like content. Cheap to fix, and a good reminder that
the retrieval corpus is prompt surface, not just an index.

---

## 7. What is misleading about my headline number

The headline is **50.8% useful automation at 3.3% harmful automation**. Here is why
you should not take it at face value.

**1. On the slice that matters, more than one in five escalation-worthy cases gets
auto-sent.** "3.3% harmful" is a share of *all* traffic, which flatters it. Expressed
as a share of the cases that actually needed a human, the agent's false-auto rate on
the representative slice is **22.2%** — 4 of 18. If those 18 include one legal threat
or one safety allegation, a rate that looks small against total volume is a serious
operational problem. The denominator is doing a lot of work.

**2. n=120 is small, and the intervals are wide.** The agent's safe-automation
interval is [55.0, 72.5]. Any comparison finer than about 10 points on this slice is
not resolvable. The intent result survives this easily; the routing results do not.

**3. The judge measures plausibility, not appropriateness.** Measured directly: the
trivial baseline's single canned reply — *"Sorry to hear about this! Please DM us the
details"* — was judged **sendable on 58 of the 63 cases that needed a human**,
including a customer alleging a Delta employee threatened them. The judge scores text
in isolation and has no idea whether the message should have been auto-sent at all.
This is precisely why §4.4 exists, and why "80% sendable" should never be quoted alone.

**4. The judge rewards the worst defect in the system.** It scored the agent's voice
4.89 *because* replies imitate Delta's house style — including inventing the initials
of real employees (F1). A metric that goes up when the system impersonates staff is
not measuring what its name claims.

**5. One annotator produced both the labels and this analysis.** There is no
inter-annotator agreement figure anywhere in this report. The golden labels, the
codebook the labels follow, and the human half of the judge validation are all mine.
Where the model disagrees with me on a genuinely contestable boundary (F4), the
harness records it as model error by construction. Some unknown fraction of the
agent's 18.3% intent error is my labelling, not its reasoning.

**6. My own harness nearly reported an outage as a finding.** In an earlier run, 127
of 220 draft calls hit a quota cap; the agent's error handling degraded them to
ESCALATE with an empty reply, and the harness scored those as decisions and produced
an 80.9% "needless escalation" rate. It reads exactly like a real behavioural
result — an over-cautious agent — and I was one step from writing it up. The harness
now refuses to score any run containing errors (Decision 20). **The general lesson:
a fallback that is correct in production is a liar in evaluation.** I would not
assume this was the only such path; it is the one I happened to catch.

**7. The "independent" judge was briefly the generator.** `gemini-flash-lite-latest`
and `gemini-3.5-flash-lite` turned out to be the same model aliased, which I only
discovered by tracing a shared quota. For part of the work the judge was grading its
own output. It is fixed, but it means model independence has to be verified
empirically, and the current judge is still the same *family* as the generator.

**8. Reply quality is measured only on the easy half.** Judging is scoped to the
representative slice for quota reasons. The enriched slice — where the agent
escalates 62% of the time — has no reply-quality numbers at all. Quality on hard
cases is simply unmeasured.

**9. Nothing here measures cost or latency.** Two model calls per ticket, thinking
tokens included, unmeasured wall-clock. A 50% automation rate at unacceptable
per-ticket latency is not a shippable result, and I cannot currently tell you which
one this is.

---

## 8. What I'd do next with one more week

In priority order — the first two are worth more than everything below them.

1. **Get a second annotator on 60 examples.** Every number in this report rests on
   one person's labels applied to their own codebook, which is the single largest
   source of unquantified error here. Two days of a colleague's time would produce a
   Cohen's kappa for the intent boundaries and the route call, and would tell me
   whether the agent's residual ~18% intent error is model error or label noise. I
   currently cannot distinguish those, which means I cannot tell whether the
   taxonomy needs fixing or the model does.

2. **Replace the judge's "sendable" call with a paired human decision on a
   larger sample.** The judge demonstrably scores text plausibility rather than
   appropriateness — it passed a generic canned reply on cases alleging staff
   threats. I would keep the per-dimension rubric (which behaves sensibly) and
   retire the single binary, replacing it with the joint routing+quality metric as
   the only headline.

3. **Rebuild retrieval on embeddings and measure whether it matters.** TF-IDF
   retrieval matches on surface words, so a novel phrasing of a common problem
   retrieves nothing useful — exactly the `low_precedent` stratum where the agent
   should be weakest. I would compare TF-IDF against a sentence-embedding index on
   that stratum specifically, because the pooled number would hide the difference.

4. **Ablate the pipeline.** Three stages were a design bet, not a measured result. I
   would run: no retrieval, retrieval without intent conditioning, and a single
   fused prompt. If the fused prompt matches on the joint metric, the extra
   complexity and latency are not earning their place.

5. **Tune the escalation rules against their false-positive cost.** The rules fire
   on the word "injured" in a message of pure praise and on the name "Sue". Each
   rule should be scored individually for precision on the enriched stratum, and the
   weakest ones narrowed with context rather than kept on instinct.

6. **Cost and latency per resolved ticket.** Nothing here measures either. A system
   that automates 40% of traffic at four seconds and two model calls per ticket is a
   different proposition from one that does it at fifteen, and no operations lead
   would approve the second without knowing.

7. **Test on a second brand without retuning.** Every threshold and rule was set
   while looking at Delta. Running the pipeline unchanged on Tesco or SpotifyCares
   would show how much of the performance is the method and how much is fitting to
   one airline's house style.

---

## 9. Repo map

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

## 10. Citations and borrowed work

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
