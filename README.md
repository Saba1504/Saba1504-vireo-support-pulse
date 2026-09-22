# Vireo Support Pulse

## Run
1. Create a clean Python environment.
2. `pip install -r requirements.txt`
3. `streamlit run app.py`

The app uses only local CSV data. No API key is required.

## Design choices
- Deduplicates repeated ticket IDs because the email thread says some legacy tickets were re-imported.
- Weekly digest is deterministic for counts and AI-assisted for text themes (TF-IDF + K-means).
- Tier 2 / Escalations & Warranty is excluded from the volume leaderboard per policy.
- Repeat-contact analysis uses same customer + category + product within 30 days as an auditable proxy for the policy's “same issue” definition; this should be manually sampled before production use.
