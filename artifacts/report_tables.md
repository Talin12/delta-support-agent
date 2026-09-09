### 4.1 Intent classification

| System | Accuracy (representative) | Macro-F1 (representative) | Accuracy (pooled) | Macro-F1 (pooled) |
|---|---|---|---|---|
| Trivial | 22.5% <sub>[15.0, 30.0]</sub> | 0.037 | 22.3% <sub>[16.8, 28.2]</sub> | 0.036 |
| Simple (no LLM) | 35.0% <sub>[26.7, 43.3]</sub> | 0.303 | 41.8% <sub>[35.5, 49.1]</sub> | 0.357 |
| Agent | 81.7% <sub>[74.2, 88.3]</sub> | 0.797 | 81.8% <sub>[76.8, 86.8]</sub> | 0.822 |

### 4.2 Routing

Representative slice only (n=120), the one production-quotable stratum.

| System | Automation rate | False-auto rate | Needless escalation | Safe automation |
|---|---|---|---|---|
| Trivial | 100.0% | 100.0% (18) | 0.0% (0) | 85.0% <sub>[78.3, 90.8]</sub> |
| Simple (no LLM) | 5.8% | 5.6% (1) | 94.1% (96) | 5.0% <sub>[1.7, 9.2]</sub> |
| Agent | 67.5% | 22.2% (4) | 24.5% (25) | 64.2% <sub>[55.0, 72.5]</sub> |

Enriched slice (n=100), where the hard cases live.

| System | Automation rate | False-auto rate | Needless escalation |
|---|---|---|---|
| Trivial | 100.0% | 100.0% (45) | 0.0% (0) |
| Simple (no LLM) | 5.0% | 0.0% (0) | 90.9% (50) |
| Agent | 38.0% | 4.4% (2) | 34.5% (19) |

### 4.3 Reply quality (blind LLM judge)

| System | Sendable rate | Grounded | Addresses | Voice | Actionable | Safe | Mean |
|---|---|---|---|---|---|---|---|
| Trivial | 43.3% <sub>[35.0, 52.5]</sub> | 3.92 | 2.31 | 2.79 | 3.83 | 4.94 | **3.56** |
| Simple (no LLM) | 16.7% <sub>[10.8, 23.3]</sub> | 2.52 | 2.19 | 4.92 | 2.44 | 4.86 | **3.39** |
| Agent | 80.0% <sub>[72.5, 87.5]</sub> | 4.82 | 4.12 | 4.89 | 3.38 | 4.89 | **4.42** |

### 4.4 Useful automation (the joint metric)

Reply quality and routing scored **together**. A reply can be perfectly
good text and still be a catastrophic thing to auto-send, so neither number
means anything alone. An example counts as useful automation only if the
system chose AUTO, the human label agreed AUTO, and the judge called the
reply sendable.

| System | Useful automation (representative) | Harmful automation (representative) |
|---|---|---|
| Trivial | 32.5% (39/120) | 15.0% (18/120) |
| Simple (no LLM) | 0.8% (1/120) | 0.8% (1/120) |
| Agent | 50.8% (61/120) | 3.3% (4/120) |
