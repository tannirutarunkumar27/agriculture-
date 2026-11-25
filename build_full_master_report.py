import os
import sys
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH

from report_builder_utils import add_styled_heading, add_p, add_bullet
from chapter1_data import build_chapter_1
from chapter2_data import build_chapter_2
from chapter3_data import build_chapter_3
from chapter4_data import build_chapter_4
from chapter5_data import build_chapter_5_and_backmatter

def build_front_matter(doc):
    # ==================== TITLE PAGE ====================
    p_title = doc.add_paragraph()
    p_title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_title.paragraph_format.space_before = Pt(40)
    p_title.paragraph_format.space_after = Pt(14)
    r = p_title.add_run("KG-HYBRID: A KNOWLEDGE-GUIDED TRUSTWORTHY AI FRAMEWORK FOR ROBUST CROP RECOMMENDATION")
    r.font.name = 'Arial'
    r.font.size = Pt(18)
    r.font.bold = True
    r.font.color.rgb = RGBColor(16, 44, 87)

    p_sub = doc.add_paragraph()
    p_sub.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_sub.paragraph_format.space_after = Pt(36)
    r = p_sub.add_run("A PROJECT REPORT\nSubmitted in partial fulfillment for the Degree of\nBachelor of Technology in Computer Science and Engineering\nSchool of Computing")
    r.font.name = 'Calibri'
    r.font.size = Pt(12)
    r.font.italic = True

    p_by = doc.add_paragraph()
    p_by.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_by.paragraph_format.space_after = Pt(30)
    r = p_by.add_run("Submitted by\nG. SAI SAGAR\nM. GUNA VARSHITH\nT. V. TARUN KUMAR\n(Register Numbers: <REGISTER NUMBERS>)")
    r.font.name = 'Arial'
    r.font.size = Pt(11)
    r.font.bold = True

    p_guide = doc.add_paragraph()
    p_guide.alignment = WD_ALIGN_PARAGRAPH.CENTER
    p_guide.paragraph_format.space_after = Pt(50)
    r = p_guide.add_run("Under the Guidance of\n<GUIDE NAME>\nDepartment of Computer Science and Engineering")
    r.font.name = 'Calibri'
    r.font.size = Pt(12)

    p_inst = doc.add_paragraph()
    p_inst.alignment = WD_ALIGN_PARAGRAPH.CENTER
    r = p_inst.add_run("DEPARTMENT OF COMPUTER SCIENCE AND ENGINEERING\nAMRITA VISHWA VIDYAPEETHAM\nAmritapuri Campus (INDIA)\nNovember 2026")
    r.font.name = 'Arial'
    r.font.size = Pt(12)
    r.font.bold = True
    r.font.color.rgb = RGBColor(16, 44, 87)

    doc.add_page_break()

    # ==================== CERTIFICATE ====================
    add_styled_heading(doc, "BONAFIDE CERTIFICATE", level=1)
    add_p(doc, "This is to certify that this project report entitled “KG-Hybrid: A Knowledge-Guided Trustworthy AI Framework for Robust Crop Recommendation” is a bonafide record of the work carried out by G. Sai Sagar, M. Guna Varshith, and T. V. Tarun Kumar, under my guidance and supervision, in partial fulfillment of the requirements for the award of the Degree of Bachelor of Technology in Computer Science and Engineering at Amrita Vishwa Vidyapeetham, Amritapuri Campus.", space_after=60)
    
    p_sig = doc.add_paragraph()
    p_sig.alignment = WD_ALIGN_PARAGRAPH.JUSTIFY
    p_sig.paragraph_format.line_spacing = 1.5
    r = p_sig.add_run("<GUIDE NAME>                                    <PROJECT COORDINATOR NAME>                     <HEAD OF DEPARTMENT>\n(Project Guide)                                        (Project Coordinator)                                      (Head of Department)")
    r.font.name = 'Calibri'
    r.font.size = Pt(10.5)
    r.font.bold = True

    doc.add_page_break()

    # ==================== ACKNOWLEDGEMENTS ====================
    add_styled_heading(doc, "ACKNOWLEDGEMENTS", level=1)
    add_p(doc, "We express our sincere gratitude to Amrita Vishwa Vidyapeetham, Amritapuri Campus, for fostering an inspiring research environment and providing the computing infrastructure necessary to develop and benchmark the KG-Hybrid framework.")
    add_p(doc, "We are deeply indebted to our project guide, <GUIDE NAME>, for their invaluable mentorship, insightful technical critique, and consistent encouragement throughout this endeavor. Their emphasis on mathematical rigor, experimental reproducibility, and practical agricultural deployment fundamentally shaped the trajectory of our research.")
    add_p(doc, "We extend our sincere thanks to the Department of Computer Science and Engineering faculty, our project coordinator, and the laboratory staff for their logistical support and constructive feedback during internal progress reviews.")
    add_p(doc, "Finally, we extend our heartfelt appreciation to our families and friends for their enduring patience, encouragement, and understanding during the development of this work.")
    
    p_st = doc.add_paragraph()
    p_st.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    p_st.paragraph_format.space_before = Pt(20)
    r = p_st.add_run("G. SAI SAGAR\nM. GUNA VARSHITH\nT. V. TARUN KUMAR")
    r.font.name = 'Arial'
    r.font.size = Pt(11)
    r.font.bold = True

    doc.add_page_break()

    # ==================== CONTENTS ====================
    add_styled_heading(doc, "CONTENTS", level=1)
    contents_items = [
        ("ABSTRACT", "iv"),
        ("LIST OF FIGURES", "vi"),
        ("LIST OF TABLES", "vii"),
        ("CHAPTER 1: INTRODUCTION", "1"),
        ("   1.1 Background and Agricultural Context", "1"),
        ("   1.2 Problem Addressed", "3"),
        ("   1.3 Motivation", "5"),
        ("   1.4 Scope and Objectives of the Project", "6"),
        ("   1.5 Report Organization", "8"),
        ("CHAPTER 2: LITERATURE REVIEW / EXISTING SYSTEM", "9"),
        ("   2.1 Existing System Study and Theoretical Foundations", "9"),
        ("       2.1.1 Conventional Machine Learning Paradigms", "9"),
        ("       2.1.2 Deep Tabular Transformers", "11"),
        ("       2.1.3 Theory-Guided Data Science and Knowledge Guidance", "12"),
        ("       2.1.4 Explainability, Uncertainty, and Trustworthiness", "13"),
        ("   2.2 Review of Existing Research Work", "14"),
        ("   2.3 Review of Existing Tools and Technologies", "16"),
        ("   2.4 Research Gaps and Opportunities for Innovation", "17"),
        ("   2.5 Formal Problem Statement and Research Contributions", "19"),
        ("CHAPTER 3: PROPOSED WORK", "21"),
        ("   3.1 Architectural Overview of KG-Hybrid", "21"),
        ("       3.1.1 Methodological Progression", "22"),
        ("   3.2 Dataset Selection and Agronomic Profiles", "24"),
        ("       3.2.1 Agronomic Characteristics of the 22 Crop Categories", "25"),
        ("   3.3 Leak-Free Preprocessing Pipeline", "27"),
        ("   3.4 Agronomic Domain Knowledge Engineering", "28"),
        ("   3.5 Mathematical Architecture of KGAAN", "29"),
        ("       3.5.1 Knowledge-Guided Feature Gate", "29"),
        ("       3.5.2 Branch A: XGBoost Decision Boundaries & Leaf Embeddings", "30"),
        ("       3.5.3 Branch B: Feature Tokenizer Transformer (FT-Transformer)", "31"),
        ("       3.5.4 Knowledge-Guided Adaptive Gated Fusion", "32"),
        ("       3.5.5 Multi-Task Auxiliary Training Objectives", "33"),
        ("   3.6 Two-Stage Training Protocol", "34"),
        ("   3.7 Dynamic Two-Stage Early-Exit Pipeline", "35"),
        ("   3.8 Fault-Tolerance: Missing Data Imputation", "36"),
        ("CHAPTER 4: EXPERIMENTATION AND RESULT ANALYSIS", "37"),
        ("   4.1 Experimental Setup and Reproducibility Specifications", "37"),
        ("   4.2 Dataset Description and Exploratory Analysis", "38"),
        ("   4.3 Evaluation Metrics Formulation", "40"),
        ("   4.4 Comprehensive Baseline Comparison", "41"),
        ("       4.4.1 Analysis of Baseline Performance", "42"),
        ("   4.5 Hyperparameter Tuning Configuration", "43"),
        ("   4.6 Training Dynamics and Convergence Logs", "44"),
        ("   4.7 Detailed Per-Crop Class-by-Class Analysis", "45"),
        ("   4.8 Multi-Dimensional Model Comparison", "46"),
        ("   4.9 Systematic 7-Variant Ablation Study", "47"),
        ("   4.10 Robustness Stress Test 1: Missing Feature Imputation", "48"),
        ("   4.11 Robustness Stress Test 2: Sensor Noise and Extrapolation", "49"),
        ("   4.12 Explainability and Feature Complementarity Analysis", "51"),
        ("   4.13 Uncertainty Quantification and Probability Calibration", "52"),
        ("   4.14 Inference Optimization and Edge Deployment", "54"),
        ("   4.15 Summary of Experimental Findings and Practical Significance", "55"),
        ("CHAPTER 5: CONCLUSION AND SCOPE FOR FURTHER RESEARCH", "56"),
        ("   5.1 Conclusion and Summary of Research Achievements", "56"),
        ("   5.2 Agronomic and Practical Implications", "57"),
        ("   5.3 Critical Reflection on System Limitations", "58"),
        ("   5.4 Scope for Further Research", "59"),
        ("APPENDIX A: IMPLEMENTATION DIRECTORY AND ARTIFACTS", "61"),
        ("APPENDIX B: CORE ALGORITHMIC SOURCE CODE", "62"),
        ("APPENDIX C: PRE-SUBMISSION COMPLIANCE CHECKLIST", "63"),
        ("REFERENCES (IEEE FORMAT)", "64")
    ]
    for item, pg in contents_items:
        p_c = doc.add_paragraph()
        p_c.paragraph_format.space_after = Pt(2)
        r1 = p_c.add_run(item)
        r1.font.name = 'Calibri'
        r1.font.size = Pt(9.5)
        if "CHAPTER" in item or "APPENDIX" in item or "REFERENCES" in item or "ABSTRACT" in item:
            r1.font.bold = True
        dots_len = max(4, 76 - len(item))
        r2 = p_c.add_run(" " + "." * dots_len + " " + pg)
        r2.font.name = 'Calibri'
        r2.font.size = Pt(9.5)

    doc.add_page_break()

    # ==================== LIST OF FIGURES ====================
    add_styled_heading(doc, "LIST OF FIGURES", level=1)
    figs = [
        ("Figure 3.1: Detailed Architecture of KGAAN", "21"),
        ("Figure 3.2: End-to-End Training and Inference Pipeline of KG-Hybrid", "23"),
        ("Figure 3.3: Knowledge-Guided Gated Fusion and Bias Injection Mechanism", "33"),
        ("Figure 3.4: Two-Stage Confidence-Based Early-Exit Decision Flow", "36"),
        ("Figure 4.1: Agricultural Feature Correlation Heatmap", "39"),
        ("Figure 4.2: FAO/ICAR 22x7 Agronomic Knowledge Suitability Heatmap", "40"),
        ("Figure 4.3: Comprehensive Baseline Model Comparison Across Evaluation Metrics", "42"),
        ("Figure 4.4: Training Convergence Curves: Cross-Entropy Loss and Validation Accuracy", "44"),
        ("Figure 4.5: Test Confusion Matrix for KG-Hybrid Across 22 Balanced Crop Categories", "45"),
        ("Figure 4.6: Ablation Study Performance Degradation Across Variants A1–A7", "47"),
        ("Figure 4.7: Sensor Dropout Robustness: Raw Degradation vs. KNN Imputation Recovery", "49"),
        ("Figure 4.8: Performance Under Additive Gaussian Noise (σ ∈ [0, 0.3])", "50"),
        ("Figure 4.9: Feature Complementarity: TreeSHAP vs. FT-Transformer Attention Weights", "51"),
        ("Figure 4.10: Model Reliability Diagrams and Probability Calibration Curves", "53"),
        ("Figure 4.11: Parameter Count vs. Standalone Latency Trade-Off", "54"),
        ("Figure 4.12: Production FastAPI Schema Contract and Advisory Workflow", "55")
    ]
    for item, pg in figs:
        p_c = doc.add_paragraph()
        p_c.paragraph_format.space_after = Pt(3)
        r1 = p_c.add_run(item)
        r1.font.name = 'Calibri'
        r1.font.size = Pt(10)
        dots_len = max(4, 74 - len(item))
        r2 = p_c.add_run(" " + "." * dots_len + " " + pg)
        r2.font.name = 'Calibri'
        r2.font.size = Pt(10)

    doc.add_page_break()

    # ==================== LIST OF TABLES ====================
    add_styled_heading(doc, "LIST OF TABLES", level=1)
    tables = [
        ("Table 2.1: Systematic Review of Benchmark Agricultural Machine Learning Works", "15"),
        ("Table 2.2: Tools and Technologies Used in the Project", "16"),
        ("Table 3.1: Methodological Progression of the KG-Hybrid Framework", "23"),
        ("Table 3.2: Agronomic Description and Ranges of Benchmark Input Features", "24"),
        ("Table 3.3: Leak-Free Stratified Data Partition Summary", "26"),
        ("Table 3.4: Core Software Packages, Libraries, and Execution Versions", "34"),
        ("Table 4.1: Experimental Hardware and Reproducibility Specifications", "38"),
        ("Table 4.2: Descriptive Statistical Summary of the Crop Recommendation Dataset", "39"),
        ("Table 4.3: Comparative Performance of KG-Hybrid Against Six Baseline Models", "41"),
        ("Table 4.4: Tuned Hyperparameter Configuration for the KG-Hybrid Pipeline", "43"),
        ("Table 4.5: Multi-Dimensional Robustness and Operational Comparison Matrix", "46"),
        ("Table 4.6: Ablation Study Metric Degradation (A1 through A7)", "47"),
        ("Table 4.7: Model Accuracy Under Progressive Missing-Feature Sensor Dropout", "48"),
        ("Table 4.8: Accuracy Retention Under Additive Gaussian Noise Perturbations", "50"),
        ("Table 4.9: Out-of-Distribution Generalization Under 1.5x Feature Extrapolation", "50"),
        ("Table 4.10: Feature Attribution Weights and Spearman Rank Correlation (ρ = 0.43)", "51"),
        ("Table 4.11: Probability Calibration Comparison (ECE, Brier Score, and NLL)", "53"),
        ("Table 4.12: Early-Exit Inference Routing Latency, Accuracy, and Traffic Distribution", "54"),
        ("Table 4.13: Parameter Footprint and Execution Latency: Full vs. Lightweight KG-Hybrid", "55")
    ]
    for item, pg in tables:
        p_c = doc.add_paragraph()
        p_c.paragraph_format.space_after = Pt(3)
        r1 = p_c.add_run(item)
        r1.font.name = 'Calibri'
        r1.font.size = Pt(10)
        dots_len = max(4, 74 - len(item))
        r2 = p_c.add_run(" " + "." * dots_len + " " + pg)
        r2.font.name = 'Calibri'
        r2.font.size = Pt(10)

    doc.add_page_break()

    # ==================== ABSTRACT ====================
    add_styled_heading(doc, "ABSTRACT", level=1)
    add_p(doc, "In precision agriculture, aligning crop selection with localized soil biochemistry and micro-climatic dynamics is essential for maximizing yield and protecting rural investments. Standard machine learning models—ranging from gradient-boosted decision trees to deep tabular transformers—treat agricultural data as abstract numerical matrices. In doing so, they ignore fundamental agronomic principles, operate as uninterpretable black boxes, output uncalibrated confidence scores, and degrade sharply when exposed to sensor dropout, measurement noise, or seasonal shifts.")
    add_p(doc, "To resolve these vulnerabilities, this project introduces KG-Hybrid, a knowledge-guided, trustworthy deep learning framework designed specifically for robust agricultural crop recommendation. At the center of our architecture is the Knowledge-Guided Agriculture-Aware Attention Network (KGAAN), which unifies two complementary learning paradigms:")
    add_bullet(doc, "Branch A (Tree Decision Partitions): ", "Embeds discrete leaf-node trajectories from an ensemble of 200 frozen XGBoost trees into continuous latent representations via a trainable embedding bag.")
    add_bullet(doc, "Branch B (Contextual Attention): ", "Uses a 4-layer Feature Tokenizer Transformer (FT-Transformer) with 8 attention heads to model continuous, multi-order interactions across soil and weather variables.")
    add_p(doc, "These branches are dynamically reconciled via a gated fusion module conditioned on an explicit 22 × 7 agronomic suitability matrix compiled from Food and Agriculture Organization (FAO) and Indian Council of Agricultural Research (ICAR) guidelines. Training is stabilized through a two-stage protocol combining cross-entropy with symmetric knowledge distillation and supervised contrastive objectives.")
    add_p(doc, "Evaluated on the standard 2,200-sample, 22-class benchmark dataset under a leak-free 70:15:15 stratified split, KG-Hybrid achieves 99.70% Top-1 Accuracy, 99.70% Macro-F1, and 100.00% Top-3 Recall, matching Random Forest (99.70%) while outperforming baseline XGBoost (99.39%) and FT-Transformer (98.79%). Under simulated field degradation, KG-Hybrid demonstrates superior resilience:")
    add_bullet(doc, "Gaussian Noise Resilience: ", "Retains 89.21% accuracy under Gaussian sensor noise (σ = 0.2), outperforming XGBoost by +9.82 percentage points.")
    add_bullet(doc, "Feature Extrapolation: ", "Retains 90.00% accuracy under a 1.5× environmental feature extrapolation, whereas XGBoost drops to 74.55%.")
    add_bullet(doc, "Sensor Dropout Recovery: ", "Couples with a leak-free K-Nearest Neighbors (KNN, k=5) imputation module to restore accuracy from 47.03% to 82.73% (+35.70 pp recovery) under 40% missing sensor data.")
    add_p(doc, "Explainability analysis via TreeSHAP and self-attention weights reveals that the two branches capture distinct, complementary signals (Spearman ρ = 0.43): XGBoost focuses on soil macronutrients (N, P, K), while FT-Transformer prioritizes climate indicators (temperature, humidity, pH). Prediction safety is governed by Monte Carlo Dropout (M=10), which flags high-uncertainty samples (σ > 0.15). For practical field deployment, a confidence-aware two-stage early-exit pipeline routes 96.36% of queries through Stage 1, cutting average latency from 115.54 ms to 5.99 ms (14× speedup), while a lightweight 4.04M-parameter variant achieves a 74% size reduction. The complete system is deployed as an asynchronous FastAPI microservice.")
    add_p(doc, "Precision Agriculture, Crop Recommendation, Knowledge-Guided Machine Learning, XGBoost Leaf Embeddings, FT-Transformer, Trustworthy AI, Explainable AI (XAI), Monte Carlo Dropout, Robust Machine Learning, Edge Computing.", bold_prefix="Keywords: ", space_after=12)

    doc.add_page_break()

def generate_master_document():
    doc = Document()
    
    # Configure 1-inch margins
    for s in doc.sections:
        s.top_margin = Inches(1.0)
        s.bottom_margin = Inches(1.0)
        s.left_margin = Inches(1.0)
        s.right_margin = Inches(1.0)
        
    img_dir = os.path.join(os.getcwd(), "manuscript")
    
    print("Building Front Matter...")
    build_front_matter(doc)
    
    print("Building Chapter 1: Introduction...")
    build_chapter_1(doc)
    doc.add_page_break()
    
    print("Building Chapter 2: Literature Review...")
    build_chapter_2(doc)
    doc.add_page_break()
    
    print("Building Chapter 3: Proposed Work...")
    build_chapter_3(doc, img_dir)
    doc.add_page_break()
    
    print("Building Chapter 4: Experimentation and Result Analysis...")
    build_chapter_4(doc, img_dir)
    doc.add_page_break()
    
    print("Building Chapter 5, Appendices, and References...")
    build_chapter_5_and_backmatter(doc)
    
    output_path = os.path.join(os.getcwd(), "KG_Hybrid_Comprehensive_Project_Report_40_50_Pages.docx")
    doc.save(output_path)
    print(f"Master document successfully built and saved to: {output_path}")

if __name__ == "__main__":
    generate_master_document()

