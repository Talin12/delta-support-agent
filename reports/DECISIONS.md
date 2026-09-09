# Decision log

The non-obvious calls, and why. Roughly in the order they were made.

---

**1. Brand = Delta, chosen on reply substance rather than volume.**
AmazonHelp (170k replies) and AppleSupport (107k) are the two biggest brands, but
52% of Apple's replies and 82% of T-Mobile's are some form of "please DM us".
Grounding a reply generator on a brand that mostly deflects makes the task
degenerate: the agent learns to emit "DM us", scores well, and demonstrates
nothing. Delta has 42k replies with only 16% DM-deflection and 15% link-only, so
its historical replies actually contain the substance the retrieval step is
supposed to surface. Evidence in `artifacts/brand_survey.csv`.

**2. Pulled the dataset from a HuggingFace mirror rather than Kaggle.**
Kaggle needs credentials, which makes "reproduce this in 15 minutes" depend on the
reviewer having an account. The mirror `SunidhiSriram/twcs` serves the identical
file — verified by byte size (516,508,641) and an exact column-schema match. The
Kaggle path is still scripted in `scripts/download_data.sh` as the canonical
source; the mirror is the zero-friction default.

**3. Sorted thread turns by timestamp, not `tweet_id`.**
`tweet_id` in this dump is assigned in *reverse* chronological order. Sorting by it
silently inverts every conversation, so `first_customer_message` becomes the
brand's reply. This produced plausible-looking threads with the roles swapped and
would have quietly corrupted the taxonomy, the golden set, and every metric. Caught
by eyeballing sampled threads rather than by any test — which is itself the lesson.

**4. Dropped threads where a second brand also replied.**
Customers routinely tag two airlines in one complaint. Those threads pollute the
"how does *this brand* resolve it" signal that the whole retrieval step depends on.

**5. Built the taxonomy on the action axis, not on the clusters.**
TF-IDF/KMeans over 6,000 messages produced clean clusters — and they were the wrong
shape. Cluster 4 ("crew") and cluster 9 ("gate") each contain both effusive praise
and furious complaints, because clustering splits on shared vocabulary while
support cares about what has to happen next. So the clusters were treated as
evidence of what the traffic is *about*, and the taxonomy was cut along what the
agent must *do*. Two intents exist only if they imply different actions.

**6. Added `refund_or_compensation` during labelling, not before it.**
Clustering never surfaced refunds as their own cluster, because refund language is
spread thinly across every topic (delays, bags, Wi-Fi, seats, miles) and so never
dominates a centroid. Hand-labelling made it unmissable: it is both frequent (16% of
the golden set) and the highest-stakes thing the agent can get wrong, since it
commits real money. The clearest case for treating unsupervised structure as
evidence rather than as the answer.

**7. Two-stratum golden set, reported separately and never pooled as a headline.**
A uniform sample of 220 would contain roughly four safety-critical messages — too
few to say anything about the failure mode most likely to cause real harm. So 120
are uniform random (the only slice a production estimate may be quoted from) and
100 deliberately oversample rule-triggering, long-running, low-precedent, and very
short messages. Escalation base rate is 15% in the representative slice and 45% in
the enriched one; quoting the pooled 28.6% as a headline would be misleading, and
the report says so explicitly.

**8. The labeller saw Delta's actual historical reply.**
This grounds the route label in the brand's real operating posture instead of the
labeller's imagination. It costs some hindsight bias — knowing Delta escalated makes
it easier to call a case escalation-worthy. Taken deliberately and disclosed, rather
than avoided.

**9. Framed routing as "can this be sent unreviewed?", not "is this hard?".**
Delta's real first move on most tickets is a templated "DM us your confirmation
number", and that template is genuinely safe to auto-send for a routine bag delay.
Defining escalation as difficulty would have made almost everything escalate.
Defining it as *sendability* isolates the cases where a templated reply is actively
harmful — a legal threat, a safety allegation, a viral complaint — which is the
decision a support lead actually makes.

**10. Ambiguity alone does not escalate.**
The first labelling pass was inconsistent on vague messages. The rule that resolved
it: a clarifying question ("what happened?") is itself a perfectly safe auto-reply,
so ambiguity only escalates when paired with urgency or severity, where asking a
question burns the window in which action was possible. Two similar rules were added
the same way — a defection threat alone is not escalation, and initiating a claim is
AUTO while chasing a stalled one is ESCALATE.

**11. Deterministic rules sit *above* the model, not beside it.**
Safety, legal, fraud, and money cases are rare enough that they barely move an
accuracy number and severe enough that one mistake costs more than every routine
ticket combined. An LLM router is fine on the median ticket and unreliable exactly
on that tail. So those cases are decided by regex rules the model cannot overrule,
and the model only decides what the rules abstain on. The cost is false positives,
which are visible and quantified in the failure analysis.

**12. Three stages instead of one prompt.**
Classify, then retrieve, then draft-and-route. One fused call would be cheaper and
would make it impossible to say which stage a failure came from. Splitting them
means intent gets a closed-set metric, replies get a judge, and routing gets a
cost-asymmetric analysis — three different questions with three different answers.

**13. The simple baseline got its own calibrated confidence threshold.**
Initially it inherited the agent's 0.55 confidence floor and escalated 100% of
traffic, which looked like a decisive win for the agent. It was an artefact: a
10-class logistic regression's max probability lives in a completely different range
from an LLM's self-reported confidence. Comparing them on a shared threshold is a
straw man. The baseline now calibrates its own floor on training folds only, picking
the most automation available subject to a ≤10% false-auto budget.

**14. The simple baseline uses 5-fold cross-validation, not LLM-distilled labels.**
Distilling weak labels from an LLM onto a larger training pool would make a stronger
classifier, but it inherits the LLM's biases and quietly makes the "no LLM" baseline
depend on an LLM. Cross-validating on the golden set keeps the baseline genuinely
LLM-free, so `make baselines` runs with no API key at all. Each prediction comes
from a model that never saw that example.

**15. The judge runs on a different model from the generator.**
Scoring replies with the same model that wrote them invites self-preference bias,
where a model rates its own output generously. Generation uses
`gemini-3.6-flash`; judging uses `gemini-3.8-flash`. This reduces the bias but does
not eliminate it — they remain the same model family, which is stated as a
limitation rather than papered over.

**16. The judge scores each reply independently and blind.**
No system identity is attached, and replies are scored one at a time rather than
pairwise. Independent scoring makes blinding trivial and sidesteps position bias
entirely, at the cost of some sensitivity to small quality differences.

**17. Every LLM response is cached to a committed SQLite file.**
Keyed on a hash of (model, system, prompt, temperature, max_tokens). This makes
reruns free and deterministic, and lets a reviewer with no API key reproduce every
headline number via `LLM_OFFLINE=1`. It also means a prompt change invalidates
exactly the affected entries and nothing else.

**18. Golden threads are excluded from the retrieval index.**
Otherwise an evaluated message retrieves its own historical answer and the reply
scores become meaningless. Enforced in `ResolutionIndex(exclude_thread_ids=...)`
rather than left to convention.

**19. False-auto and needless-escalation are reported separately and never averaged.**
Auto-sending something that needed a human can be a legal or PR incident; escalating
a routine ticket costs an agent a minute. Collapsing them into one accuracy or F1
number hides the only distinction that matters operationally.

**20. Bootstrap confidence intervals on every headline number.**
At n=120 in the representative slice, a five-point difference is frequently noise.
The intervals make that impossible to hide, including where they undercut the
agent's own results.
