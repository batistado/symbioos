# SymbioOS — *IEEE Transactions on Parallel and Distributed Systems* Manuscript Pipeline

Contains the two-column **IEEE Transactions on Parallel and Distributed Systems (`IEEEtran`, `10pt,journal,compsoc`)** manuscript (`main.tex`), BibTeX bibliography (`references.bib`), 5 Type-42 TrueType embedded vector PDF figures + 300-DPI PNG previews (`figures/`), static LaTeX/desk-rejection/Humanizer validator (`validate_latex.py`), and one-click Overleaf submission archive (`symbioos_tpds_overleaf.zip`).

## Quick Commands

### 1. Regenerate All 5 Figures from Verified Telemetry
```bash
python3 ../scripts/generate_paper_figures.py
```

### 2. Run Static LaTeX, Citation, Cross-Reference, Abstract, Page-Budget & Humanizer Validator
```bash
python3 validate_latex.py
```

### 3. Compile PDF
* **Overleaf (1-click):** Upload `symbioos_tpds_overleaf.zip` directly as a new project on Overleaf (`pdfLaTeX` compiler, main document `main.tex`).
* **Local TeX Live / MacTeX (if installed):**
  ```bash
  pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex
  ```

---

## IEEE Transactions on Parallel and Distributed Systems (TPDS) Submission Checklist

### Format & Desk-Rejection Compliance
* **Document Class:** `\documentclass[10pt,journal,compsoc]{IEEEtran}` (`IEEEtran.cls` v1.8b).
* **Page Estimate:** **11 pages** (comfortably within IEEE TPDS's 12-page regular-paper free limit, inclusive of references, Appendix A formal proofs, and disclosures).
* **Abstract:** Single paragraph, **193 words** (`<= 200` words strictly), zero inline math, zero citations, zero footnotes.
* **Mandatory Disclosures:** Author ORCID (`0009-0009-9058-5375`), `IEEEkeywords`, Data and Code Availability, and Acknowledgments and Competing Interests.
* **PDF eXpress Figure Compliance:** All 5 vector PDFs embed Type-42 TrueType fonts (`pdf.fonttype = 42`, `ps.fonttype = 42`) using the colorblind-safe Okabe-Ito palette.

### Author Roster

| Order | Name | Email | Affiliation | Role / ORCID |
| :---: | :--- | :--- | :--- | :--- |
| 1 | Mohammed Kamran Syed | `mohammedkamran.syed@ieee.org` | IEEE Independent Researcher, Mountain House, CA, USA | Corresponding & Submitting Author; ORCID: `0009-0009-9058-5375` |
| 2 | Anudeep Rentala | `anudeep.rentala@ieee.org` | IEEE Independent Researcher, Fremont, CA, USA | Co-Author |
| 3 | Somesh Rahul | `someshrahul@ieee.org` | IEEE Independent Researcher, Fremont, CA, USA | Co-Author |
| 4 | Raghuram Malpe Pai | `raghuram.pai@ieee.org` | IEEE Independent Researcher, Sunnyvale, CA, USA | Co-Author |
| 5 | Krishna Suman Dara | `krishna-suman.dara@ieee.org` | IEEE Independent Researcher, Sunnyvale, CA, USA | Co-Author |
