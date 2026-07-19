# 7-Minute Presentation Script
### Predicting Employee Flight-Risk — Retention & Reskilling in the Age of AI

**Audience:** your tutor (technical). Skip ML basics. Lead with insight, not process.
**Timing:** ~7 min speaking + 3 min Q&A. Word count ≈ 980 (≈140 wpm). Rehearse to trim.
**Rubric weighting:** the *Insights & Justifications* block (slides 12–15) is where the presentation marks live — spend ~4 of your 7 minutes there. Introduction is context only.

> Delivery notes are in [brackets]. Don't read them aloud.

---

## ⏱️ 0:00 – 0:40 — Open on the problem (Slides 1 → 3)
[Slide 1, then click to 3. Don't dwell on the title slide.]

"Good morning. As AI and automation reshape which job roles matter, organisations are facing a quieter cost: they're losing people they could have kept or reskilled — and they only notice at the resignation letter.

My project builds an early-warning model that flags an employee's flight-risk *while there's still time to act*, and — crucially for a workforce-development function — points to *why*, so the response can be a targeted retention or reskilling move rather than a guess.

The AI draft framed this as 'employees leave, it costs money.' I pushed back: that misses *why now*, it ignores the strategic loss of reskillable talent, and it never says who acts or when. My refined framing fixes all three."

[Pace: brisk. This is context — don't linger.]

---

## ⏱️ 0:40 – 1:20 — Persona, JTBD, target (Slide 4)
[Click to 4.]

"The user is Mei, a workforce-development lead. Her job-to-be-done: 'when I review my team, show me who's at risk and why, so I can act before they go.'

That framing drives two modelling decisions. First, the target is a real binary — did the employee leave — not a score derived from other columns, so there's no circular leakage. Second, I use only information available *while the person is still employed* — no post-exit fields — because a score that needs the outcome to compute it is useless in deployment.

Attrition here is about 16% of the data — so this is a rare-event problem, and that shapes everything downstream."

[The leakage + rare-event points are technical credibility signals. Say them clearly.]

---

## ⏱️ 1:20 – 2:20 — Data, features, the AI-exposure proxy (Slides 5 → 6)
[Click to 5, then 6.]

"On the data: 1,470 employees, 35 fields. I dropped four — three near-constant columns with zero variance, and an employee ID the model would just memorise.

Then I engineered features. The one I want to flag is **AI-Exposure**: I mapped each job role to an automation-exposure score, informed by Frey-Osborne and WEF-style rankings. I'll be upfront — this is a *defensible proxy, not a measured value*. And this chart [gesture to slide 6] is why I keep it honest: exposure does *not* rise cleanly with attrition. The highest-exposure and one lower-exposure group both show elevated exits. So exposure is a *contributing signal that interacts with pay and workload* — not a deterministic predictor. I built it in as a driver to analyse, not a label to trust blindly."

[This honesty is a marks-earner. Deliver it with confidence, not apology.]

---

## ⏱️ 2:20 – 3:10 — Models & the imbalance problem (Slides 7 → 8)
[Click to 7, then 8. Move quickly — this is method, not insight.]

"I compared three models on an identical pipeline — Decision Tree as a readable baseline, Random Forest for variance reduction, and XGBoost, which won because it captures interactions like overtime-by-income-by-role.

The key technical move is on slide 8. Default models chase accuracy, which on 16% data means predicting 'stay' and missing most leavers — Stage-1 recall was as low as 13%. So I applied class weighting, scale_pos_weight of about five, and then tuned the decision threshold. I deliberately took the F1-optimal cut-off rather than chasing a headline recall number — a threshold that flags almost everyone is worthless in practice."

[If short on time, this is the section to compress. One sentence per model is fine.]

---

## ⏱️ 3:10 – 3:50 — Results, honestly (Slides 9 → 11)
[Click through 9, 10, 11 fairly quickly. Land the numbers.]

"The champion — weighted, tuned XGBoost — catches about **74% of leavers**, at **36% precision**, with an **AUC of 0.79**.

I want to be straight about that precision. Attrition is genuinely hard; many flagged people stay. But the business asymmetry justifies it: missing someone I could have retained costs far more than a low-cost retention conversation with someone who was fine. The AUC of 0.79 matters here — it shows the recall isn't just over-flagging; the model genuinely *ranks* risk. And I've stated the limits on slide 11: synthetic dataset, US context, proxy feature — findings are directional, to be revalidated on real HR data."

[Resist the urge to apologise for 74/36. Frame it as the *correct* trade-off, defended.]

---

## ⏱️ 3:50 – 6:00 — THE INSIGHTS (Slides 12 → 15) ★ spend the most time here
[This is the graded core. Slow down. Let each plot breathe.]

**[Slide 12 — beeswarm]**
"Now what the model actually learned. Globally, **overtime is the single strongest driver** — people on sustained overtime skew sharply toward leaving. That reads as a burnout signal.

Second — and this validates the engineering — **AI-Exposure is the number-two driver**. Roles more exposed to automation carry higher predicted flight-risk. Reward features — stock options, income — pull the *other* way: ownership and pay retain people. And the SHAP framing matters: these are contributions to *predicted risk*, not guaranteed exits."

**[Slide 13 — FI vs PFI]**
"I cross-checked with permutation importance — shuffling each feature and watching AUC drop. Overtime, AI-exposure, and reward survive *both* tests. That agreement tells me they're real signal, not an artefact of one importance metric — which is what lets me recommend acting on them."

**[Slide 14 — dependence plots]**
"Two drivers up close. Overtime pushes risk up *regardless of pay* — you can't simply buy your way out of burnout. And AI-exposure's contribution is strongest where pay is *low* — the automation-anxious, under-rewarded segment is the sharp end of the risk."

**[Slide 15 — waterfall + scenario]**
"Finally, one individual. This employee is flagged near the top of the risk range. The waterfall decomposes it: high overtime, a brand-new manager relationship, an AI-exposed role, low stock options — each adds risk, and they accumulate.

Here's the payoff for Mei. Before: this exit surfaces at resignation, too late. After: the score surfaces it early, she sees the *drivers*, rebalances the workload, and — because the role is AI-exposed — enrols the person in a reskilling track toward a less-automatable adjacent role. A likely exit becomes a redeployment. The model prompts the conversation; the human decides."

[This scenario is your closer-within-the-body. Deliver the before/after with a beat between them.]

---

## ⏱️ 6:00 – 7:00 — Recommendations & close (Slides 16 → 17)
[Click to 16. Bring it up to the strategic level.]

"So the recommendations. Use the score as an early-warning list, not an auto-action — human-in-the-loop throughout. Aim the limited reskilling budget at high-risk, high-exposure, *reskillable* staff. Treat overtime as a systemic driver, not a case-by-case one. And pair every flag with a concrete action.

The business impact is a shift from reactive to proactive: at-risk talent surfaced while retention is still possible, budget spent where it changes an outcome, and a bridge from 'who leaves' to 'who we can reskill' — which is exactly the workforce-development mission.

I'll close on the honest limitation: synthetic data, a proxy exposure feature, and drift as roles change fast — all reasons to revalidate on real data before trusting the specific numbers. Thank you — happy to take questions."

[End on the limitation *then* thanks — it signals maturity to a technical marker.]

---

# 🎯 Q&A Prep — likely questions & crisp answers

**"Your precision is only 36% — isn't that a bad model?"**
For this problem, no. The cost of missing a leaver I could've retained dwarfs the cost of a false alarm — a retention chat. I tuned to that asymmetry. AUC 0.79 confirms the model ranks risk well; precision is a threshold choice, not a ceiling. I could raise the threshold for a higher-precision, lower-recall list if the use case changed.

**"Isn't the AI-exposure feature just made up?"**
It's a proxy, and I flagged it as one. The scores are informed by Frey-Osborne and WEF rankings. I didn't take it on faith — slide 6 shows it doesn't perfectly track attrition, and permutation importance confirms it's real signal, not noise. In production I'd replace it with measured task-automation data.

**"The dataset is synthetic — does any of this transfer?"**
The *specific numbers* don't claim to. What transfers is the method: the leakage-safe feature set, the imbalance handling, the threshold logic, and the SHAP-driven action framework. On real HR data the pipeline runs unchanged; only the coefficients update.

**"Why not SMOTE / oversampling instead of class weights?"**
Class weighting and scale_pos_weight adjust the loss without fabricating synthetic minority rows, so there's less risk of the model learning artefacts of interpolation. I could compare SMOTE as an extension, but weighting kept the training data honest.

**"Why XGBoost over the interpretable tree, given governance?"**
The single tree is readable but its recall is too low to be useful. XGBoost recovers the recall, and SHAP restores the interpretability — global drivers plus per-employee waterfalls — so I get both detection and transparency for a human-in-the-loop decision.

**"How would you deploy and monitor this?"**
Score at review cycles on live employees. Monitor recall and precision monthly; watch for drift as roles and AI-exposure shift. Retrain when recall drops or the role mix changes. Never auto-action — always a human decision.

---

# ⏲️ Timing cheat-sheet (say these aloud to check pace)
| Block | Slides | Target | Cumulative |
|---|---|---|---|
| Problem | 1–3 | 0:40 | 0:40 |
| Persona/JTBD | 4 | 0:40 | 1:20 |
| Data/features | 5–6 | 1:00 | 2:20 |
| Models/imbalance | 7–8 | 0:50 | 3:10 |
| Results | 9–11 | 0:40 | 3:50 |
| **Insights ★** | 12–15 | **2:10** | 6:00 |
| Recs/close | 16–17 | 1:00 | 7:00 |

If you're running long, compress slides 7–8 (method) — never the insight block.
