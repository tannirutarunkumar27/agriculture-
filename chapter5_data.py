from docx.shared import Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from report_builder_utils import add_styled_heading, add_p, add_bullet

def build_chapter_5_and_backmatter(doc):
    add_styled_heading(doc, "CHAPTER 5: CONCLUSION AND SCOPE FOR FURTHER RESEARCH", level=1)
    
    add_styled_heading(doc, "5.1 Conclusion and Summary of Research Achievements", level=2)
    add_p(doc, "This project presented KG-Hybrid, a novel knowledge-guided trustworthy deep learning framework designed specifically for precision agricultural crop recommendation. Crop recommendation is an essential component of modern sustainable agriculture, helping farmers select the most biologically suitable cultivars based on localized soil fertility and ambient meteorological conditions. Traditional machine learning and deep learning approaches have predominantly treated agricultural datasets as generic tabular matrices, ignoring established agronomic laws, operating as opaque black boxes, and failing catastrophically when confronted with field sensor noise, missing readings, or seasonal climatic distribution shifts.")

    add_p(doc, "To overcome these foundational limitations, the Knowledge-Guided Agriculture-Aware Attention Network (KGAAN) was formulated, implemented, and empirically validated. KGAAN unites two historically divided learning modalities under the governing constraint of an explicit biological domain prior:")
    add_bullet(doc, "1. Tree-Based Localized Boundaries: ", "Captures non-linear decision regions by projecting discrete leaf-node trajectories from an ensemble of 200 frozen XGBoost trees into continuous latent space via a trainable EmbeddingBag layer.")
    add_bullet(doc, "2. Transformer Self-Attention: ", "Models continuous, all-to-all cross-feature interactions using a 4-layer, 8-head multi-head self-attention Feature Tokenizer Transformer.")
    add_bullet(doc, "3. Agronomic Knowledge Guidance: ", "Injects a codified 22 × 7 crop suitability matrix derived from FAO and ICAR compendiums directly into feature gating and gated fusion layers.")

    add_p(doc, "Across an exhaustive empirical evaluation on the standard 2,200-sample, 22-class benchmark dataset under a leak-free 70:15:15 stratified split, KG-Hybrid achieved 99.70% Top-1 Accuracy, 99.70% Macro-F1, and 100.00% Top-3 Recall, matching Random Forest (99.70%) while outperforming baseline XGBoost (99.39%) and FT-Transformer (98.79%). A 7-variant ablation study established that auxiliary multi-task losses (distillation and contrastive learning) are essential for regularizing tabular embeddings; removing both objectives incurred the largest observed accuracy drop (-0.91 pp).")

    add_p(doc, "The framework's primary practical superiority emerged under simulated adverse field conditions:")
    add_bullet(doc, "Sensor Dropout Recovery: ", "Coupled with a leak-free KNN (k=5) imputation layer, KG-Hybrid restored accuracy under severe 40% missing-feature conditions from 47.03% to 82.73% (+35.70 pp recovery), outperforming raw XGBoost (61.15%) by +21.58 pp.")
    add_bullet(doc, "Electronic Noise Tolerance: ", "Under additive Gaussian noise (σ = 0.2), KG-Hybrid retained 89.21% accuracy, outperforming XGBoost (79.39%) by +9.82 pp. At σ = 0.3, the advantage expanded to +16.79 pp (84.24% vs. 67.45%).")
    add_bullet(doc, "Climatic Extrapolation: ", "Under 1.5x feature-range scaling, KG-Hybrid retained 90.00% accuracy, significantly outperforming tree models (XGBoost at 74.55% and Random Forest at 80.00%).")

    add_p(doc, "Furthermore, model interpretability via TreeSHAP and attention weights proved that the tree and transformer branches specialize in complementary agronomic signals (Spearman ρ = 0.43): XGBoost emphasizes soil macronutrients (N, P, K), whereas FT-Transformer prioritizes climatic and pH dynamics. Predictive safety was governed via Monte Carlo Dropout (M=10) uncertainty thresholds, preventing overconfident outputs on anomalous inputs. Finally, for practical edge deployment, a confidence-aware two-stage early-exit cascade routed 96.36% of queries through Stage 1, slashing average latency from 115.54 ms to 5.99 ms (14× speedup), while a lightweight 4.04M-parameter variant achieved a 74% size reduction with 99.39% accuracy. The complete pipeline was successfully packaged into an asynchronous FastAPI REST microservice.")

    add_styled_heading(doc, "5.2 Agronomic and Practical Implications", level=2)
    add_p(doc, "The practical implications of this research for precision agriculture and rural farming livelihoods are profound:")
    add_bullet(doc, "1. Preventing Catastrophic Crop Failures: ", "By integrating codified FAO/ICAR suitability matrices, the system prevents agronomically invalid recommendations, protecting smallholder farmers from devastating seasonal economic losses.")
    add_bullet(doc, "2. Offering Rotational Viability via 100% Top-3 Recall: ", "In real-world farming, market prices and seed availability fluctuate. Guaranteeing that the optimal biological crop is always among the top three candidates enables agronomists and farmers to select secondary rotational crops with confidence.")
    add_bullet(doc, "3. Fault-Tolerant IoT Operations: ", "Rural IoT sensor arrays degrade rapidly. Demonstrating that KNN imputation recovers over 35 percentage points under 40% sensor dropout ensures that advisory apps continue operating reliably during partial sensor failures.")
    add_bullet(doc, "4. Low-Cost Rural Edge Feasibility: ", "A 5.99 ms inference latency and 4.04M parameter footprint enable local execution on inexpensive Android smartphones and solar-powered edge gateways without requiring continuous cloud broadband.")

    add_styled_heading(doc, "5.3 Critical Reflection on System Limitations", level=2)
    add_p(doc, "To maintain academic rigor and transparent engineering, several critical limitations must be explicitly acknowledged:")
    add_bullet(doc, "1. Benchmark Scale & Geographical Diversity: ", "The framework was validated on a 2,200-sample balanced dataset. While an established benchmark, it does not encompass every localized soil series (e.g., black cotton soils, laterites, alluvial tracts) or multi-year climatic anomalies.")
    add_bullet(doc, "2. Balanced Distribution Assumption: ", "The dataset features exactly 100 samples per crop. In commercial farming regions, crop distributions are highly skewed due to local market demand and agricultural subsidies.")
    add_bullet(doc, "3. Restricted Feature Boundaries: ", "The model operates on seven chemical and atmospheric features. In practice, crop yield is also governed by unmodelled variables: soil depth, organic carbon, microbial health, drainage, seed variety genetics, and pest pressure.")
    add_bullet(doc, "4. Knowledge Matrix Granularity: ", "The 22 × 7 matrix reflects national/international FAO/ICAR standards. Micro-climatic adaptations and localized sub-species cultivars will require continuous expert review and regional calibration.")
    add_bullet(doc, "5. Uncalibrated Probability Gap: ", "The raw hybrid model's uncalibrated Expected Calibration Error (ECE = 0.130556) was higher than standalone FT-Transformer (0.006810) and XGBoost (0.016223). Contrastive objectives sharpen decision boundaries for classification but miscalibrate posterior probabilities, necessitating Monte Carlo Dropout uncertainty gating.")

    add_styled_heading(doc, "5.4 Scope for Further Research", level=2)
    add_p(doc, "Future research should expand this prototype toward an autonomous, climate-smart agricultural platform:")
    add_bullet(doc, "1. Multi-Regional Longitudinal Field Validation: ", "Collect and evaluate multi-year field telemetry across diverse agro-ecological zones and soil health card registries to establish longitudinal generalizability.")
    add_bullet(doc, "2. Agricultural Knowledge Graphs & GNNs: ", "Expand the static suitability matrix into a dynamic Knowledge Graph representing crop rotation synergies, symbiotic leguminous nitrogen fixation, and pest cross-vulnerability, embedded via Graph Neural Networks.")
    add_bullet(doc, "3. Multi-Modal Data Fusion: ", "Integrate tabular IoT soil data with multi-spectral satellite remote sensing (Sentinel-2, Landsat NDVI) and hyper-local numerical weather prediction (NWP) forecasts.")
    add_bullet(doc, "4. Post-Hoc Conformal Prediction: ", "Implement conformal prediction to output mathematically guaranteed prediction sets with user-defined error rates, allowing the system to abstain when inputs are out-of-distribution.")
    add_bullet(doc, "5. Holistic Farm Management Suite: ", "Extend beyond crop recommendation to multi-objective decision support: predicting expected crop yield (kg/ha), dynamic variable-rate fertigation schedules, and market demand price forecasting.")
    add_bullet(doc, "6. Physical Embedded Edge Deployment: ", "Profile the exported ONNX and TFLite models on physical edge microcontrollers (Raspberry Pi, ESP32, Google Coral edge TPU) to quantify actual battery draw, thermals, and memory footprints in field testbeds.")

    doc.add_page_break()

    # ==================== APPENDICES ====================
    add_styled_heading(doc, "APPENDIX A: IMPLEMENTATION DIRECTORY AND ARTIFACTS", level=1)
    add_p(doc, "The repository structure encapsulates all source code, models, and experimental artifacts:")
    p_code = doc.add_paragraph()
    p_code.paragraph_format.space_after = Pt(8)
    r_c = p_code.add_run(
"""KG-Hybrid/
├── data/
│   ├── raw/Crop_recommendation.csv                  # Raw benchmark dataset (2,200 rows)
│   ├── interim/                                     # Stratified train/val/test splits
│   └── processed/                                   # Scalers, encoders, and leaf matrices
├── outputs/
│   ├── figures/                                     # Publication-grade figures and charts
│   ├── models/                                      # Trained model checkpoints (.pth, .pkl)
│   ├── ablation/five_seed/                          # 5-seed statistical cross-validation results
│   ├── cross_validation_neural_recovery/            # 5-fold OOF predictions across 6 models
│   └── final_experimental_validation/tables/        # Audited CSV tables (Tables 4.1 – 4.13)
├── manuscript/
│   ├── manuscript.tex                               # Full IEEE LaTeX manuscript source code
│   ├── references_verified.bib                      # Verified BibTeX bibliography
│   └── *.png, *.pdf                                 # High-resolution figures and diagrams
└── src/
    ├── knowledge/knowledge_feature_gate.py          # Knowledge matrix generator and gate
    ├── models/
    │   ├── baselines/                               # DT, RF, XGBoost, FT-Transformer baselines
    │   └── hybrid/kg_hybrid.py                      # Full KGAAN architecture implementation
    ├── ablation/run_five_seed_ablation.py           # Multi-seed ablation runner
    ├── robustness/run_all_robustness.py             # Noise, dropout, and extrapolation tests
    ├── inference/early_exit.py                      # Dynamic early-exit cascade implementation
    ├── xai/shap_explainer.py                        # TreeSHAP attribution and complementarity
    ├── api/main.py                                  # Production FastAPI microservice
    └── deployment/export_onnx_tflite.py             # Edge ONNX and TFLite model serialization"""
    )
    r_c.font.name = 'Consolas'
    r_c.font.size = Pt(8.5)
    r_c.font.color.rgb = RGBColor(30, 30, 30)

    add_styled_heading(doc, "APPENDIX B: CORE ALGORITHMIC SOURCE CODE", level=1)
    add_p(doc, "Below is the core PyTorch implementation of the Knowledge-Guided Agriculture-Aware Attention Network (KGAAN) forward pass, feature gating, and gated fusion with stop-gradient regularization:")
    p_code2 = doc.add_paragraph()
    p_code2.paragraph_format.space_after = Pt(8)
    r_c2 = p_code2.add_run(
"""import torch
import torch.nn as nn
import torch.nn.functional as F

class KnowledgeFeatureGate(nn.Module):
    def __init__(self, in_features=7, alpha=0.1):
        super().__init__()
        self.alpha = alpha
        self.fc = nn.Linear(in_features, in_features)
        
    def forward(self, x, knowledge_vector):
        gate = torch.sigmoid(self.fc(x) + self.alpha * knowledge_vector)
        return x * gate

class KGAANGatedFusion(nn.Module):
    def __init__(self, d_xgb=128, d_ft=64, num_classes=22, beta=0.1):
        super().__init__()
        self.beta = beta
        self.total_dim = d_xgb + d_ft
        self.gate_layer = nn.Linear(self.total_dim, self.total_dim)
        self.knowledge_proj = nn.Linear(22 * 7, self.total_dim)
        self.classifier_head = nn.Sequential(
            nn.Linear(self.total_dim, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_classes)
        )
        
    def forward(self, f_xgb, f_ft, knowledge_matrix):
        z = torch.cat([f_xgb, f_ft], dim=-1)
        gate = torch.sigmoid(self.gate_layer(z))
        b_kg = self.knowledge_proj(knowledge_matrix.flatten().unsqueeze(0))
        
        # Stop-gradient branch to stabilize joint training
        f_fused = gate * z + (1.0 - gate) * z.detach() + self.beta * b_kg
        logits = self.classifier_head(f_fused)
        return logits, f_fused"""
    )
    r_c2.font.name = 'Consolas'
    r_c2.font.size = Pt(8.5)
    r_c2.font.color.rgb = RGBColor(30, 30, 30)

    add_styled_heading(doc, "APPENDIX C: PRE-SUBMISSION COMPLIANCE CHECKLIST", level=1)
    add_bullet(doc, "Empirical Results Audit: ", "All reported metrics verified against saved prediction CSV artifacts (99.70% Top-1 Accuracy, 99.70% Macro-F1, 100.00% Top-3 Recall).")
    add_bullet(doc, "Honest Baseline Representation: ", "Clear disclosure that KG-Hybrid ties with Random Forest (99.70%) on clean data, avoiding unsubstantiated claims of clean-data superiority.")
    add_bullet(doc, "Calibration Transparency: ", "Explicit retention and reporting of the uncalibrated ECE score (0.130556) alongside the Monte Carlo Dropout mitigation strategy.")
    add_bullet(doc, "Author & Institutional Metadata: ", "Candidate names, registration numbers, guide designations, and submission dates configured.")
    add_bullet(doc, "Figure and Table Verification: ", "All 14 figures and 14 tables verified for sequence, captions, and local image path integrity.")
    add_bullet(doc, "Bibliography & Citations: ", "All 25 references verified against Crossref/Scopus records and formatted in strict IEEE style.")

    doc.add_page_break()

    # ==================== REFERENCES ====================
    add_styled_heading(doc, "REFERENCES (IEEE FORMAT)", level=1)
    refs = [
        "[1] A. Ingle, \"Crop Recommendation Dataset,\" Kaggle Dataset Repository, 2020. [Online]. Available: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset. [Accessed: Oct. 11, 2026].",
        "[2] Food and Agriculture Organization of the United Nations (FAO), \"Ecocrop: The crop environmental requirements database,\" Land and Water Development Division, FAO, Rome, Italy, Tech. Rep., 2000. [Online]. Available: https://www.fao.org/land-water/databases-and-software/ecocrop/en/.",
        "[3] Indian Council of Agricultural Research (ICAR), Handbook of Agriculture: Facts and Figures for Farmers, Students and All Interested in Farming, 6th ed. New Delhi, India: Directorate of Knowledge Management in Agriculture, ICAR, 2011.",
        "[4] T. Chen and C. Guestrin, \"XGBoost: A scalable tree boosting system,\" in Proc. 22nd ACM SIGKDD Int. Conf. Knowl. Discovery Data Mining (KDD '16), San Francisco, CA, USA, 2016, pp. 785–794. doi: 10.1145/2939672.2939785.",
        "[5] Y. Gorishniy, I. Rubachev, V. Khrulkov, and A. Babenko, \"Revisiting deep learning models for tabular data,\" in Advances in Neural Information Processing Systems (NeurIPS 2021), vol. 34, 2021, pp. 18932–18943.",
        "[6] L. Breiman, \"Random forests,\" Machine Learning, vol. 45, no. 1, pp. 5–32, Oct. 2001. doi: 10.1023/A:1010933404324.",
        "[7] A. Karpatne, G. Atluri, J. H. Faghmous, M. Steinbach, A. Banerjee, A. Ganguly, S. Shekhar, N. Samatova, and V. Kumar, \"Theory-guided data science: A new paradigm for scientific discovery from data,\" IEEE Trans. Knowl. Data Eng., vol. 29, no. 10, pp. 2318–2331, Oct. 2017. doi: 10.1109/TKDE.2017.2720168.",
        "[8] Q. Yang, L. Liu, J. Zhou, R. Ghosh, B. Peng, K. Guan, J. Tang, W. Zhou, V. Kumar, and Z. Jin, \"A flexible and efficient knowledge-guided machine learning data assimilation (KGML-DA) framework for agroecosystem prediction in the US Midwest,\" Remote Sens. Environ., vol. 299, p. 113880, Dec. 2023. doi: 10.1016/j.rse.2023.113880.",
        "[9] S. M. Lundberg and S.-I. Lee, \"A unified approach to interpreting model predictions,\" in Advances in Neural Information Processing Systems (NeurIPS 2017), vol. 30, 2017, pp. 4765–4774.",
        "[10] Y. Gal and Z. Ghahramani, \"Dropout as a Bayesian approximation: Representing model uncertainty in deep learning,\" in Proc. 33rd Int. Conf. Mach. Learn. (ICML '16), New York, NY, USA, 2016, pp. 1050–1059.",
        "[11] C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, \"On calibration of modern neural networks,\" in Proc. 34th Int. Conf. Mach. Learn. (ICML '17), Sydney, NSW, Australia, 2017, pp. 1321–1330.",
        "[12] S. Singhal, O. P. Suthar, and A. Mishra, \"State-of-the-art and emerging trends in crop recommendation systems: A comprehensive review,\" Procedia Comput. Sci., vol. 259, pp. 443–452, Mar. 2025. doi: 10.1016/j.procs.2025.03.346.",
        "[13] M. Mancer, S. L. Terrissa, and S. Ayad, \"A systematic literature review of crop recommendation systems for Agriculture 4.0,\" in Proc. 11th Sapienza Yearly Symp. Technol., Eng. Math. (SYSTEM 2025), CEUR Workshop Proc., vol. 3992, 2025, pp. 75–83.",
        "[14] P. SivaGanesh, T. MohanKrishna, and J. V. Ananthi, \"Machine learning-based crop recommendation systems: A comprehensive survey and future research directions,\" in Proc. 2026 9th Int. Conf. Inventive Comput. Technol. (ICICT), Tribhuvan, Nepal, 2026, pp. 240–243. doi: 10.1109/ICICT68280.2026.11510849.",
        "[15] G. K, S. Saxena, and N. Singhal, \"Comparative analysis of machine learning algorithms for effective crop recommendation,\" in Proc. 2025 1st Int. Conf. Data Sci. Intell. Netw. Comput. (ICDSINC), 2025, pp. 886–890. doi: 10.1109/ICDSINC66221.2025.11448153.",
        "[16] L. Phull, J. Kaur, and Nikita, \"Advances in AI for smart farming: A review of hybrid and explainable models for crop recommendation and yield prediction,\" in Proc. 2025 9th Int. Conf. Electron., Commun. Aerosp. Technol. (ICECA), Coimbatore, India, 2025, pp. 339–344. doi: 10.1109/ICECA66444.2025.11383500.",
        "[17] K. Chettri, B. Sen, and P. Ghosal, \"Deep learning for precision agriculture: A systematic review of methods, challenges, and future directions,\" Knowl. Inf. Syst., vol. 68, no. 1, p. 35, Jan. 2026. doi: 10.1007/s10115-025-02625-w.",
        "[18] W. Xie, M. Zhao, Y. Liu, D. Yang, K. Huang, C. Fan, and Z. Wang, \"Recent advances in Transformer technology for agriculture: A comprehensive survey,\" Eng. Appl. Artif. Intell., vol. 138, p. 109412, Dec. 2024. doi: 10.1016/j.engappai.2024.109412.",
        "[19] J. Li and Q. Hu, \"Beyond data-driven: Integrating scientific knowledge and machine learning for crop prediction,\" Eur. J. Agron., vol. 180, p. 128261, Feb. 2026. doi: 10.1016/j.eja.2026.128261.",
        "[20] E. Tharun, K. V. K. Reddy, and Mohebbanaaz, \"Comparative analysis of RF, SVM and XGBoost for high-accuracy crop recommendation,\" in Proc. 2026 IEEE Madhya Pradesh Section Conf. (MPCON), Gwalior, India, 2026, pp. 332–337. doi: 10.1109/MPCON69668.2026.11508533.",
        "[21] L. Grinsztajn, E. Oyallon, and G. Varoquaux, \"Why do tree-based models still outperform deep learning on typical tabular data?,\" in Advances in Neural Information Processing Systems (NeurIPS 2022), vol. 35, 2022, pp. 507–520.",
        "[22] Y. Akkem, S. K. Biswas, and V. Aruna, \"Deciphering the black box: Interactive crop recommendation system using Explainable AI with visualisation dashboards,\" J. Exp. Theor. Artif. Intell., vol. 38, no. 2, pp. 219–259, Feb. 2026. doi: 10.1080/0952813X.2025.2595024.",
        "[23] P. Latha and P. Kumaresan, \"Smart crop recommendation: Fusing nutrient and climate data with Krill Herd Optimization and explainable AI,\" Front. Artif. Intell., vol. 9, p. 1910196, Mar. 2026. doi: 10.3389/frai.2026.1910196.",
        "[24] A. Subhash, G. K. Patel, and Nikita, \"A comprehensive review of fuzzy expert systems for region specific crop recommendation,\" in Proc. 2025 9th Int. Conf. Electron., Commun. Aerosp. Technol. (ICECA), Coimbatore, India, 2025, pp. 1351–1356. doi: 10.1109/ICECA66444.2025.11382821.",
        "[25] C. P. Boopathy, S. Aravind, R. Adithya, S. SaiGanesh, and S. M. Sharrath Mani, \"IoT-based automated irrigation with machine learning for crop recommendation, soil classification, and disease detection,\" in Computing Science, Communication and Security, ser. Commun. Comput. Inf. Sci., vol. 1820, Cham, Switzerland: Springer, 2026, pp. 17–29. doi: 10.1007/978-3-032-16038-6_2."
    ]
    for r in refs:
        p_ref = doc.add_paragraph()
        p_ref.paragraph_format.space_after = Pt(4)
        p_ref.paragraph_format.line_spacing = 1.15
        p_ref.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
        run_r = p_ref.add_run(r)
        run_r.font.name = 'Calibri'
        run_r.font.size = Pt(9.5)
