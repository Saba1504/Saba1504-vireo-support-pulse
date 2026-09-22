# Memo — Vireo Audio Support Pulse

**To:** Priya Raman, Head of Customer Experience  
**Subject:** What the 18-month support export says and what I would act on

## Executive summary

The export contains 12,528 rows but 653 ticket IDs appear twice across the legacy and current systems. I therefore de-duplicated by `ticket_id` before analysis.

The clearest measurable opportunity is repeat contact. Using the policy definition as an auditable proxy — same customer, same category and product, with another contact within 30 days of resolution — 1,349 of 11,266 eligible resolved/closed tickets (12.0%) had a repeat contact. Those repeat contacts represent about ₹367,360 of contact cost across the 18-month period, or roughly ₹61,227 per quarter at the observed mix.

A practical operating goal is to keep repeat contacts at or below the recent 2026 Q2 rate of 10.2%. Holding the recent rate instead of the 18-month rate would avoid approximately 200 repeat contacts over an 18-month-equivalent volume, worth roughly ₹8,850 per quarter. This is a planning estimate, not a guaranteed saving.

## What the tool does

The attached Support Pulse app:
- produces a weekly category digest;
- extracts recurring language themes from customer messages using TF-IDF + K-means;
- shows a weekly Tier 1 closure leaderboard;
- excludes Tier 2 / Escalations & Warranty from the volume leaderboard, as required by policy;
- surfaces SLA breaches and the associated ₹350 breach-credit exposure.

## Why this is useful

The category counts are already available in the export, so the AI is not being used to manufacture numbers. It is used to help a manager read the free text at weekly scale. The deterministic counts remain the audit layer.

The repeat-contact calculation is deliberately conservative about what can be proven from the supplied data. “Same issue” is a semantic concept in the policy; the prototype uses customer + category + product as a transparent proxy. A production version should sample and label a few hundred pairs to measure proxy accuracy before using the KPI as a hard target.

## Agent leaderboard

The requested closure leaderboard is shown for Tier 1 only. The policy explicitly says Tier 2 cases are multi-touch and should not be compared with Tier 1 on weekly ticket volume.

## Limitations

- The submission pack did not include `submission-form.md`, so that form cannot be completed faithfully.
- The prototype does not call a paid LLM/API, avoiding surprise per-ticket model cost.
- Legacy migration/re-imports require de-duplication before trend or volume reporting.
