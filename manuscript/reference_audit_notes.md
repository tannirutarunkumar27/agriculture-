# Bibliography correction notes

Checked 11 October 2026. These are publication-metadata corrections, not a
systematic review or an endorsement of each study's scientific quality.
No experimental result or manuscript paragraph was changed.

## Deliverables

- references_verified.bib: 36 unique, source-identified entries.
- Of the 42 supplied entries, 25 were retained/corrected as identifiable
  publications; two were reconstructed as an identifiable repository report
  and dataset (AgWise and CropRec-BD); 15 were excluded pending identification.
- Nine additions: three references needed by the revised Related Work,
  five foundational methods references, and one recent close related work.
- The supplied bibliography remains unchanged in the user's attachment.
- Crossref lookup evidence is in reference_audit_metadata.json and
  reference_followup_metadata.json. HTTP failures and unrelated title
  candidates in those files must not be treated as identification evidence.
- The output passed local structural checks, not a complete IEEEtran
  LaTeX/BibTeX compilation. A full manuscript and bibliography build is
  still needed to check formatting and undefined citations.

## Important citation-key compatibility notes

Most retained keys are unchanged to avoid breaking existing citations.
A citation key does not determine the actual authors or publication year.

- thilakarathne2025machine now correctly identifies Singh et al. (2025),
  Sustainability 17(10), 4711, DOI 10.3390/su17104711.
  Change any prose saying "Thilakarathne et al." for this title to "Singh et al."
- chen2026interpretable identifies Venkateswara and Padmanaban (2025),
  Scientific Reports 15, 41721, DOI 10.1038/s41598-025-26910-4.
  Change any prose saying "Chen et al. (2026)" for this title.
- fuzzy2026comprehensive identifies Subhash, Patel, and Nikita in the
  ICECA 2025 proceedings, pp. 1351–1356; it is not a 2026 conference paper.
- fttransformer2026benchmarking is an IEEE Access journal article by
  Asal and Yalciner, vol. 14, pp. 11660–11681 (2026), not the supplied
  software-engineering conference paper.
- agwise2025modular was reconstructed as the identifiable CGIAR report
  described below, not the supplied working-paper-series record.
- bangladesh2025dataset was reconstructed as the identifiable Mendeley
  dataset described below, not the supplied Data in Brief article.
  Confirm these two reconstructed records are the works you intended
  before retaining their citations.

For any excluded key, remove or revise the corresponding manuscript claim,
or provide a publisher link, DOI, or first page so the intended work can be
identified. Do not merely swap in a related paper while retaining a claim
that the replacement does not support.

In the originally supplied Related Work, reddy2025comparative is cited.
That exact title/author combination could not be identified. If it remains
in the latest manuscript, revise the citation. xgboost2026comparative
supports discussion of the RF/SVM/XGBoost comparison; grinsztajn2022trees
supports a broader tabular-model benchmark discussion. Neither is
automatically a substitute for every claim attached to the excluded key.

## Audit of all supplied keys

For DOI-identified records, the full DOI is in references_verified.bib.
The publisher's deposited metadata was checked using Crossref, with
publisher/repository pages used for sources not covered by Crossref.

| Supplied key | Disposition / principal correction |
| --- | --- |
| singhal2025state | Retained; DOI, authors, volume and pages match. |
| mancer2025systematic | Corrected Terrissa's given names, full symposium title, CEUR volume 3992 and source URL. |
| sivaganesh2026machine | Changed article to conference paper; corrected coauthor name and added DOI. |
| gamya2025comparative | Corrected first-author parsing to publisher-deposited Gamya K; added DOI and corrected proceedings title. |
| phull2025advances | Added DOI; preserved single-name author Nikita. |
| chettri2026deep | Added article identifier 35; verified 2026 publication. |
| xie2024recent | Retained; title/DOI metadata verified. |
| beyond2026data | Corrected full author names; volume is 180, article identifier is 128261. |
| reddy2025comparative | Excluded: exact supplied title/author pairing not identified; multiple same-title works make substitution unsafe. |
| xgboost2026comparative | Added DOI; preserved single-name author Mohebbanaaz. |
| ai2024comprehensive | Replaced placeholder with six deposited authors; corrected ICDSIS61075 to ICDSIS61070 in DOI; added pp. 1–5. |
| gorishniy2021revisiting | Retained; corrected publisher to Curran Associates and added official proceedings URL. |
| fttransformer2026benchmarking | Replaced placeholders; corrected to IEEE Access 14, 11660–11681 with DOI. |
| transformer2026comprehensive | Corrected all four authors, issue 7s, pp. 1136–1149 and DOI. |
| tabtransformer2026agricultural | Excluded: no identifiable exact publication record found. |
| thilakarathne2025machine | Corrected to Singh et al.; issue 10, article 4711 and DOI. |
| patel2020state | Retained; DOI metadata verified. |
| sharma2026explainable | Excluded: supplied DOI did not resolve in the audit; exact title/author record not identified. |
| chen2026interpretable | Corrected authors, year, journal, article identifier and DOI to the actual Scientific Reports paper. |
| deciphering2026interactive | Changed to journal article; corrected authors, volume 38(2), pp. 219–259 and DOI; used 2026 issue year rather than 2025 online date. |
| fairness2026evaluation | Excluded: supplied title/author/journal combination not identified; a differently titled IEEE conference paper is not a verified identity match. |
| tian2025knowledge | Excluded: supplied DOI did not resolve; title/author record not identified. A related Tian wheat-yield paper has a different title. |
| agwise2025modular | Reconstructed as the identifiable CGIAR Sustainable Farming Science Program report; corrected title and author attribution. Confirm intended identity. |
| smartcrop2026fusing | Corrected authors to P. Latha and P. Kumaresan and restored actual title order. |
| land2026suitability | Excluded: supplied authors and subtitle not identified; a shorter-titled article by a different author is not a verified identity match. |
| fuzzy2026comprehensive | Replaced placeholder authors; corrected conference and year to ICECA 2025; added pp. 1351–1356 and DOI. |
| hybrid2026ensemble | Excluded: supplied DOI did not resolve; exact record not identified. |
| robust2026uncertainty | Excluded: supplied DOI did not resolve; exact record not identified. |
| missing2026feature | Excluded: supplied DOI did not resolve; exact record not identified. |
| noisy2026robust | Excluded: supplied DOI did not resolve; exact record not identified. |
| edge2026deployment | Excluded: supplied DOI did not resolve; exact record not identified. |
| mobile2026app | Excluded: supplied DOI did not resolve; exact record not identified. |
| federated2026privacy | Excluded: supplied DOI did not resolve; exact publication/venue record not identified. |
| cropdataset | Retained official Kaggle dataset URL and uploader; unverified upload year omitted; access date added. |
| dataset2026augmented | Excluded: supplied DOI/article/authors not identified. A related Zenodo dataset exists but is a different bibliographic record. |
| bangladesh2025dataset | Reconstructed as CropRec-BD v1 on Mendeley Data, 2025, DOI 10.17632/dtf278skpw.1; preserved contributor display name. Confirm intended identity. |
| india2026multi | Excluded: exact dataset and stable government repository record not identified. |
| boopathy2026iot | Corrected book chapter type, actual book title and pp. 17–29; retained series separately. |
| sundaresan2023smartfarming | Changed to journal article; corrected deposited compound author names. |
| vidyasree2023one | Changed to book chapter; retained actual book title and added series. |
| seshanth2025population | Added pp. 522–528; author parsing follows publisher-deposited given/family names. |
| desini2025aiassist | Corrected actual book title, chapter type, pp. 365–375 and compound surname parsing. |

"Not identified" means insufficient reliable evidence to use the supplied
record, not a finding that the underlying work is necessarily nonexistent.

## Added sources and where they belong

| Key | Appropriate use |
| --- | --- |
| grinsztajn2022trees | Comparison of tree-based and deep tabular models in Related Work. |
| karpatne2017theory | General theory/knowledge-guided learning background. |
| yang2023kgmlda | Prior knowledge-guided agricultural prediction; not a crop-classification benchmark comparator. |
| reddy2026geotab | Recent graph–tabular hybrid crop recommendation; discuss similarities and differences before citing. |
| chen2016xgboost | Original XGBoost method in the architecture/baselines section. |
| breiman2001random | Random Forest baseline definition. |
| guo2017calibration | Calibration and temperature-scaling methods. |
| gal2016dropout | MC-Dropout methodological background. |
| lundberg2017shap | SHAP explanations, only if SHAP is actually used/discussed. |

The GeoTab-CRS paper uses graph-attention/tabular-transformer fusion and
reports robustness and uncertainty-calibration evaluations. It makes a
blanket "no prior graph–Transformer crop recommendation with robustness
or uncertainty evaluation" claim unsafe. KG-Hybrid should instead be
distinguished by its actual crop–feature suitability prior, XGBoost leaf
representation and implemented fusion/evaluation design. This is not a
claim that the two systems have equivalent performance.

Bibliography entries alone do not establish novelty or support a specific
claim; read and cite the associated work where the comparison is made.
Do not compare reported scores across different datasets/splits as if
they were a controlled head-to-head experiment.

## Primary records and exports

- [CEUR SYSTEM 2025 proceedings](https://ceur-ws.org/Vol-3992/)
- [FT-Transformer original NeurIPS paper](https://proceedings.neurips.cc/paper_files/paper/2021/hash/9d86d83f925f2149e9edb0ac3b49229c-Abstract.html)
- [Tree-based tabular-model benchmark](https://proceedings.neurips.cc/paper_files/paper/2022/hash/0378c7692da36807bdec87ab043cdadc-Abstract-Datasets_and_Benchmarks.html)
- [Theory-guided data science](https://doi.org/10.1109/TKDE.2017.2720168)
- [KGML-DA agroecosystem prediction](https://doi.org/10.1016/j.rse.2023.113880)
- [CGIAR AgWise report](https://cgspace.cgiar.org/items/babae878-20ab-456b-9049-9b90eb0428c0)
- [GeoTab-CRS graph/tabular crop recommendation](https://link.springer.com/article/10.1007/s10791-026-10487-1)
- [Smart crop recommendation, Latha and Kumaresan](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2026.1910196/full)
- [Interpretable fertilizer/crop recommendation](https://www.nature.com/articles/s41598-025-26910-4)
- [CropRec-BD v1 dataset](https://data.mendeley.com/datasets/dtf278skpw/1)
- [Original Kaggle benchmark](https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset)
- [Related augmented dataset, not the supplied Data in Brief article](https://zenodo.org/records/19709807)
- [Calibration reference](https://proceedings.mlr.press/v70/guo17a.html)
- [MC-Dropout reference](https://proceedings.mlr.press/v48/gal16.html)

Dataset repository metadata is not a guarantee of data quality, collection
provenance or relevance to the experiments. Do not cite CropRec-BD or the
augmented Zenodo dataset as evidence about the original Kaggle benchmark.

## IEEEtran integration

Place references_verified.bib alongside the main .tex file and use:

~~~latex
\bibliographystyle{IEEEtran}
\bibliography{references_verified}
~~~

The filename is supplied without the .bib extension. Keep only relevant
citations in the manuscript; BibTeX normally prints cited entries only.
If using a project subfolder, update the bibliography path accordingly.
After changing references, rebuild LaTeX → BibTeX → LaTeX → LaTeX.
Check all undefined-citation warnings and revise excluded keys.

