# Experiment Log

**Append-only.** Every run whose numbers could reach the paper gets an entry, at the time it runs —
not reconstructed later. If a number is in `main.tex` and not in this file, it does not ship.

Add entries at the bottom. Never edit or delete a past entry; if a run was wrong, add a new entry
that supersedes it and say so.

---

## Entry template

```
### <YYYY-MM-DD> · <RUN-ID> · <Task> · <one-line title>
- **Commit:** <sha>
- **Config:** <path to yaml / CLI invocation>
- **Data:** <dataset, split protocol — LOSO fold(s), signer IDs held out>
- **Hardware:** <RTX 4060 16GB / RTX 3050 4GB / Colab T4 — and which for the reported latency>
- **Seeds:** <list>
- **Metrics:** <metric = value ± std, per fold if applicable>
- **Claim(s) touched:** <C-IDs from CLAIMS_LEDGER.md>
- **Verdict:** <supports / weakens / refutes / inconclusive>
- **Notes:** <anything surprising, any fallback taken, any suspicion about the number>
```

### Rules
1. **LOSO or it doesn't count** for EmoSign. Record which signer was held out per fold.
2. **Latency and VRAM only from the RTX 3050.** If the number came from the 4060, label it clearly
   as a training-hardware measurement and do not use it in the efficiency table.
3. **Record fallbacks taken** — pseudo-labelled `L`, RAF-DB instead of DFEW, partial INCLUDE
   download. These become limitations text later; if they are not logged they get forgotten.
4. **Log suspicious results too.** An unexpectedly high score on a low-α emotion class
   (surprise− α=0.119, disgust α=0.166) is a bug signal. Write down the suspicion.
5. Failed and abandoned runs still get entries. Negative results save the next person a week.

---

## Log

### 2026-08-07 · — · T1 · Project initialized
- **Commit:** initial
- **Notes:** Documentation spine created (`project_breakdown.md`, `plan.md`, `team.md`, `paper/`).
  Dataset audit finding recorded: `/home/bhuwan/Videos/data` is **INCLUDE (ISL)**, not WLASL, with
  12 truncated `.part` archives including `Train_Test_Split`. No experiments yet.
- **Verdict:** n/a

<!-- Append new entries below this line. -->
