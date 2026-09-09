# Golden set: sampling and labelling protocol

220 hand-labelled first-contact messages to Delta, drawn from 25,676 eligible
reconstructed threads. Every example carries an `intent` label, a binary `route`
label, and a free-text `route_reason`.

## 1. Sampling

Two strata, kept tagged in the data so metrics can be reported separately.

| Stratum | n | How drawn | What it is for |
|---|---|---|---|
| `representative` | 120 | Uniform random over all eligible threads | The **only** slice from which a production estimate may be quoted |
| `enriched` | 100 | Oversampled tail (below) | Measuring the rare cases that cause real harm |

The enriched stratum breaks down as:

| Tag | n | Rationale |
|---|---|---|
| `rule_triggered` | 40 | Message matches a safety/legal/fraud/money pattern. Base rate is ~1.8%, so a uniform 220 would contain ~4 — too few to measure the failure mode that matters most |
| `long_thread` | 25 | Thread ran ≥6 turns, a proxy for "was genuinely hard to resolve" |
| `low_precedent` | 20 | Lowest max TF-IDF similarity to any other thread — cases where retrieval has nothing useful to offer |
| `short_vague` | 15 | Under 60 characters — tests whether the system abstains instead of guessing |

**Pooled metrics over all 220 are not a production estimate.** The enriched
stratum is roughly 20× over-weighted on hard cases by construction. Every headline
number in the report is quoted on the representative slice, with the pooled and
enriched numbers shown alongside to expose the gap.

Eligibility filter: the thread must contain at least one Delta reply and a first
customer message of ≥12 characters after cleaning.

## 2. What the labeller saw

For each example: the cleaned customer message, the thread length, and **Delta's
actual historical first reply**.

Showing the historical reply is a deliberate trade. It grounds the `route` label in
what the brand really did rather than in the labeller's imagination, which matters
because "should a human handle this?" depends on the brand's real operating
posture. The cost is hindsight bias: knowing Delta escalated makes it easier to call
it escalation-worthy. This is a known limitation and is stated in the report rather
than hidden. The `intent` label does not need the reply and was assigned from the
customer message alone.

## 3. Intent labelling

Nine intents, defined in `taxonomy.json`. Rules applied in order:

1. **Label what the customer wants, not how they feel.** An angry message about a
   delayed bag is `baggage_issue`, not `service_complaint`.
2. **When a message spans two intents, label the one that determines the first
   action.** "Flight was late AND my bag is missing" → `baggage_issue`, because the
   bag trace is the action; the delay is already history.
3. **Praise plus a problem is the problem.** Compliments attached to a request are
   labelled by the request.
4. **`other` is for genuinely no-intent traffic** (photos, aviation chatter, wrong
   account) or text too vague to act on — never as a hedge between two real intents.
5. **Reported failure vs. asked question** separates issue intents from
   `policy_question`. "Can I carry on an umbrella?" is `policy_question`; "your agent
   made me check my umbrella" is `service_complaint`.

## 4. Route labelling

The question is precise: **could this reply be sent publicly with no human reading
it first?**

This framing matters. Delta's real first move on most tickets is a templated "DM us
your confirmation number", and that template is genuinely safe to auto-send for a
routine bag delay. The interesting cases are the ones where that same template is
actively harmful — where sending anything unreviewed makes things worse.

Label **ESCALATE** if any of these hold:

| # | Condition | Why |
|---|---|---|
| E1 | Legal action, lawyer, regulator, or formal complaint threatened | Anything sent enters a formal record |
| E2 | Safety, injury, harassment, or discrimination alleged | Duty of care; a template reads as dismissal |
| E3 | Account compromise, fraud, or unauthorised charges | Needs identity verification impossible in public |
| E4 | Explicit demand for a human, or explicit hostility to templated replies | Auto-replying reliably escalates anger |
| E5 | Money or compensation demanded | Commits brand cost; needs authorisation |
| E6 | Severe distress, or a grievance severe enough that a template inflames it | Reputational risk |
| E7 | Press, journalist, or an already-viral complaint | Read far beyond the customer |
| E8 | Ambiguous **and** urgent or severe, so a clarifying question is not an adequate response | A confident wrong reply is worse than a handoff |
| E9 | Time-critical and in progress (at the gate now, about to misconnect) | Needs someone who can act immediately, not a form reply |

Otherwise label **AUTO**: praise, routine policy questions with public answers, and
routine issues where the standard "share your details" first step is appropriate and
low-risk.

Three boundary rules were added during labelling, when the first pass proved
inconsistent without them:

- **Ambiguity alone does not escalate.** A clarifying question ("what happened?") is
  itself a safe auto-reply. E8 fires only when ambiguity is paired with urgency or
  severity, where asking a question wastes the window in which action was possible.
- **A defection threat alone does not escalate.** "Happy I'm on United" is ordinary
  venting and appears throughout routine traffic; treating it as severe would
  escalate a large share of the inbox for no benefit.
- **Initiating a claim is AUTO; chasing an unresolved one is ESCALATE.** Pointing a
  customer at the standard damaged-bag claim process commits nothing. Following up
  on a claim that has already stalled needs someone with case access and authority.

`route_reason` must name the specific property of that message that drove the call.
"Complex" and "to be safe" are not reasons.

## 5. Process and known limitations

- Labels were assigned by one person (the author) in a single pass against this
  codebook, then re-read a second time; disagreements with the first pass were
  resolved by tightening the rule above rather than by case-by-case judgement.
- **Single-annotator design is the biggest weakness of this golden set.** There is
  no inter-annotator agreement figure, so the intent boundaries most likely to be
  contested (`service_complaint` vs `flight_disruption`, `policy_question` vs
  `booking_change`) carry unmeasured label noise. A second annotator on even 60
  examples would put a number on it. This is the first thing listed under "what I'd
  do next".
- The dataset is Twitter traffic from late 2017. Delta's actual policies, tone, and
  channel mix have moved on. Nothing here should be read as current Delta practice.
- Public tweets only. The DM conversations where most issues were really resolved
  are not in the dataset, so "how the brand resolved it" is truncated to the public
  opening move.
