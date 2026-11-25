# Paper 2 — Next Task: IEEE Submission Pre-Checklist
# Created: 2026-11-16
# This is the ordered task list for bringing Paper 2 from its current
# DRAFT_VERIFIED state to IEEE submission readiness.
#
# Local (can do now) items are marked [LOCAL].
# External dependency items are marked [EXTERNAL].

---

## STAGE A — Locally Completable Now

### A1. Bibliographic Metadata Final Check [LOCAL]
File: paper2/manuscript/PAPER2_FULL_MANUSCRIPT_DRAFT.md, References section

For each of the 13 references, verify:
- [ ] Author initials follow IEEE style (e.g., "B. McMahan" not "McMahan B.")
- [ ] Conference/journal names use IEEE-standard abbreviations
  - "Proc. AISTATS" ✓
  - "Proc. MLSys" ✓  
  - "Proc. NeurIPS" ✓
  - "Proc. ACM CCS" ✓
  - "IEEE Transactions on Artificial Intelligence" — spell out in full ✓
- [ ] Page ranges vs. article numbers are consistent (use "art." for article IDs)
- [ ] All DOIs resolve (spot-check 3–4 live)
- [ ] Year appears at the end of each reference entry
- [ ] Remove inline hyperlinks (Markdown link syntax) — IEEE format uses DOI only

Action: Edit references section and save as PAPER2_FULL_MANUSCRIPT_DRAFT_v2.md

---

### A2. Add Author and Affiliation Block [LOCAL — owner must fill]
Currently missing from the manuscript header. Add:

```
Authors: [Your Name(s)]
Affiliation: [Department, Institution, City, Country]
Email: [corresponding author email]
Manuscript received: [date]; revised: [date]; accepted: [date].
```

This block must appear before the Abstract in IEEE format.

---

### A3. Convert Markdown to IEEE Two-Column LaTeX [LOCAL]
The manuscript is in Markdown. IEEE submission requires LaTeX using the
`IEEEtran` document class.

Steps:
1. Install pandoc if not present: `winget install pandoc`
2. Run conversion:
   ```powershell
   pandoc paper2/manuscript/PAPER2_FULL_MANUSCRIPT_DRAFT.md `
     -o paper2/manuscript/paper2_ieee.tex `
     --template=IEEEtran.tex `
     --bibliography=paper2/manuscript/references.bib `
     --citeproc
   ```
3. Manually post-process:
   - Remove Markdown-style hyperlinks converted to LaTeX `\href{}`
   - Replace `\href{DOI}{}` with `\doi{}` from `IEEEtran`
   - Ensure figures use `\begin{figure}[htbp]` with `\centering`
   - Tables: use `\begin{table}[htbp]` with `\caption{}` above
4. Build PDF: `pdflatex paper2_ieee.tex && bibtex paper2_ieee && pdflatex paper2_ieee.tex`
5. Check output for overfull `\hbox` warnings and fix line breaks

---

### A4. Figure Placement and Captions [LOCAL]
Current figures (pre-existing audited plots from paper2/outputs/figures/) need:
- [ ] All figures are ≥300 DPI (or vector PDF/EPS)
- [ ] Each figure has a self-contained caption (no "see text" references)
- [ ] Figure captions end with a period
- [ ] Figures are referenced in-text before they appear ("Fig. 1 shows...")
- [ ] In two-column layout, figures wider than one column use `figure*` environment

Specific figures to check:
- R13 equal-group RMSE comparison bar chart
- R14 noise-utility degradation line plot  
- R13/R14 support panel charts (seen/unseen species)
- Geographic group performance table (or figure)

---

### A5. Table Formatting [LOCAL]
IEEE style for tables:
- [ ] `\toprule`, `\midrule`, `\bottomrule` (booktabs style)
- [ ] Caption above the table, not below
- [ ] No vertical rules (remove `|` column separators)
- [ ] Numbers aligned on decimal point

Tables to check:
- Table I: Method comparison (equal-group RMSE, pooled RMSE for 8 methods)
- Table II: Geographic group breakdown (systems, experiments per group)
- Table III: R14 noise-utility degradation (RMSE at σ_mult = 2, 4, 8)

---

### A6. Word/Page Count Check [LOCAL]
IEEE Transactions papers: 8–10 pages in two-column format (including references).
IEEE conference papers: typically 6 pages.

After LaTeX conversion, run: `texcount paper2_ieee.tex`
- If over limit: condense Limitations, merge small subsections, shorten captions
- If under limit: expand key Results subsections with additional analysis detail

---

## STAGE B — External Dependencies (Cannot Complete Locally)

### B1. Linux Dependency Lock Verification [EXTERNAL — WSL or VM]
```bash
python3 -m venv /tmp/paper2-verify
source /tmp/paper2-verify/bin/activate
pip install -r requirements-linux-lock-proposed.txt
pip-licenses --format=markdown --order=license > linux_license_audit.txt
pip freeze > requirements-linux-lock-verified.txt
```
Save both outputs to paper2/ and record SHA-256 hashes in the release gate.

### B2. Clean Linux Reproduction [EXTERNAL — fresh VM required]
Run the full experiment pipeline on a machine that has never seen this codebase.
This clears the `clean_linux_reproduction` blocker in completion_audit_v10.

### B3. Dataset Redistribution Permission [EXTERNAL]
Contact Zenodo record owner at DOI 10.5281/zenodo.14160885.
Ask for explicit written permission to link/redistribute derived artifacts.
Until granted: direct users to the DOI only (DATASET_NOTICE.txt is in place).

### B4. Multi-Host Federated Deployment [EXTERNAL]
Minimum: 2 separate Linux hosts. Target: 7 hosts (one per geographic group).
Best free option: Oracle Always Free (2 Ampere A1 VMs, always-free tier).
Protocol is frozen at paper2/paper2-dp-secagg-deploy/.

### B5. Independent Privacy/Security Review [EXTERNAL]
Submit paper2/results/restart_authenticated_runtime_privacy_accounting_v1/
independent_review.template.json to a qualified reviewer.

### B6. Agronomic Rule Review [EXTERNAL]
Submit paper2/results/restart_agronomic_review_packet_v1/ to an agronomist.
8 candidate rules, 0 approved. Keep guidance disabled until review completes.

---

## Current Status After 2026-11-16 Local Fixes

| Gate | Status |
|------|--------|
| project_code_license | ✅ RESOLVED — Apache-2.0 |
| third_party_license_inventory | ⚠️ PROPOSED — needs Linux verify |
| RR03 dependence/collisions | ✅ RESOLVED — documented |
| Canonical audit paths | ✅ RESOLVED — CANONICAL_AUDIT_PATHS.txt |
| manuscript scope | ✅ CORRECT — regression only, no unsupported claims |
| IEEE conversion | 🔲 NEXT TASK (Stage A above) |
| bibliographic metadata | 🔲 NEXT TASK (A1) |
| author/affiliation block | 🔲 OWNER ACTION REQUIRED (A2) |
| Linux lock verified | 🔲 EXTERNAL (B1) |
| clean Linux reproduction | 🔲 EXTERNAL (B2) |
| dataset redistribution | 🔲 EXTERNAL (B3) |
| multi-host deployment | 🔲 EXTERNAL (B4) |
| privacy/security review | 🔲 EXTERNAL (B5) |
| agronomic review | 🔲 EXTERNAL (B6) |
