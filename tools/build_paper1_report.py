"""Build the Paper 1 implementation report from audited KG-Hybrid evidence."""
from __future__ import annotations

import csv
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.platypus import (
    BaseDocTemplate, Flowable, Frame, KeepTogether, PageBreak, PageTemplate,
    Paragraph, Spacer, Table, TableStyle,
)

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output" / "pdf" / "Paper1_Implementation_Report.pdf"
EVIDENCE = ROOT / "KG-Hybrid" / "outputs" / "final_experimental_validation"

NAVY = colors.HexColor("#17365D")
BLUE = colors.HexColor("#D9EAF7")
PALE = colors.HexColor("#F5F8FB")
GRAY = colors.HexColor("#D9D9D9")


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def p(text: str, style: ParagraphStyle) -> Paragraph:
    return Paragraph(text, style)


def table(data, widths, header=True, font_size=8.2):
    value = Table(data, colWidths=widths, repeatRows=1 if header else 0, hAlign="LEFT")
    commands = [
        ("GRID", (0, 0), (-1, -1), 0.35, GRAY),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING", (0, 0), (-1, -1), 5),
        ("RIGHTPADDING", (0, 0), (-1, -1), 5),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
        ("FONTNAME", (0, 0), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
    ]
    if header:
        commands += [
            ("BACKGROUND", (0, 0), (-1, 0), NAVY),
            ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("ALIGN", (1, 0), (-1, -1), "CENTER"),
        ]
        for index in range(1, len(data)):
            if index % 2 == 0:
                commands.append(("BACKGROUND", (0, index), (-1, index), PALE))
    value.setStyle(TableStyle(commands))
    return value


class Pipeline(Flowable):
    """Simple vector pipeline for the methodology chapter."""
    width = 16.5 * cm
    height = 12.4 * cm

    def __init__(self):
        super().__init__()
        self.width = 16.5 * cm
        self.height = 12.4 * cm

    def wrap(self, available_width, available_height):
        """Reserve the full diagram height so ReportLab never overlays prose."""
        return self.width, self.height

    def draw_box(self, canvas, x, y, w, h, title, detail, fill=colors.white):
        canvas.setFillColor(fill)
        canvas.setStrokeColor(NAVY)
        canvas.roundRect(x, y, w, h, 7, fill=1, stroke=1)
        canvas.setFillColor(NAVY)
        canvas.setFont("Helvetica-Bold", 8.5)
        canvas.drawCentredString(x + w / 2, y + h - 13, title)
        canvas.setFillColor(colors.black)
        canvas.setFont("Helvetica", 7.2)
        for i, line in enumerate(detail):
            canvas.drawCentredString(x + w / 2, y + h - 27 - i * 9, line)

    def arrow(self, canvas, x1, y1, x2, y2):
        canvas.setStrokeColor(NAVY)
        canvas.setFillColor(NAVY)
        canvas.setLineWidth(1)
        canvas.line(x1, y1, x2, y2)
        canvas.circle(x2, y2, 1.8, stroke=0, fill=1)

    def draw(self):
        c = self.canv
        w = self.width
        bw, bh = 4.8 * cm, 1.45 * cm
        x = (w - bw) / 2
        ys = [10.55 * cm, 8.60 * cm, 6.65 * cm, 4.70 * cm, 2.75 * cm, 0.80 * cm]
        boxes = [
            ("Input observations", ["N, P, K, temperature,", "humidity, pH and rainfall"]),
            ("Data preparation", ["Stratified splits or folds", "Scaler fit only on training data"]),
            ("Two learned representations", ["XGBoost leaf embeddings", "FT Transformer numeric features"]),
            ("Knowledge guided fusion", ["Crop profile attention", "Adaptive gate and layer normalization"]),
            ("Crop classifier", ["22 crop classes", "Probability distribution and label"]),
            ("Evaluation and audit", ["Accuracy, macro F1, calibration", "OOF validation and saved predictions"]),
        ]
        for index, (title, detail) in enumerate(boxes):
            self.draw_box(c, x, ys[index], bw, bh, title, detail, BLUE if index in (1, 3, 5) else colors.white)
            if index < len(boxes) - 1:
                self.arrow(c, x + bw / 2, ys[index], x + bw / 2, ys[index + 1] + bh)
        c.setFillColor(colors.HexColor("#555555"))
        c.setFont("Helvetica-Oblique", 7.3)
        c.drawCentredString(w / 2, 0.38 * cm, "Figure 1. Paper 1 KG-Hybrid implementation and evaluation pipeline")


def build_styles():
    styles = getSampleStyleSheet()
    return {
        "title": ParagraphStyle("Title", parent=styles["Title"], fontName="Helvetica-Bold", fontSize=22,
                                leading=27, alignment=TA_CENTER, textColor=colors.black, spaceAfter=12),
        "subtitle": ParagraphStyle("Subtitle", parent=styles["Normal"], fontName="Helvetica", fontSize=12,
                                   leading=16, alignment=TA_CENTER, textColor=colors.black, spaceAfter=8),
        "chapter": ParagraphStyle("Chapter", parent=styles["Heading1"], fontName="Helvetica-Bold", fontSize=17,
                                  leading=22, textColor=colors.black, spaceBefore=12, spaceAfter=16),
        "h1": ParagraphStyle("H1", parent=styles["Heading2"], fontName="Helvetica-Bold", fontSize=13,
                              leading=17, textColor=colors.black, spaceBefore=11, spaceAfter=6),
        "h2": ParagraphStyle("H2", parent=styles["Heading3"], fontName="Helvetica-Bold", fontSize=11.3,
                              leading=15, textColor=colors.black, spaceBefore=9, spaceAfter=4),
        "body": ParagraphStyle("Body", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.2,
                                leading=15, alignment=TA_JUSTIFY, textColor=colors.black, spaceAfter=8),
        "small": ParagraphStyle("Small", parent=styles["BodyText"], fontName="Helvetica", fontSize=8.5,
                                 leading=11, alignment=TA_LEFT, textColor=colors.HexColor("#333333"), spaceAfter=5),
        "thead": ParagraphStyle("TableHeader", parent=styles["BodyText"], fontName="Helvetica-Bold", fontSize=8.1,
                                 leading=9.5, alignment=TA_LEFT, textColor=colors.white, spaceAfter=0),
        "toc": ParagraphStyle("TOC", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.5,
                               leading=17, leftIndent=10, textColor=colors.black),
        "bullet": ParagraphStyle("Bullet", parent=styles["BodyText"], fontName="Helvetica", fontSize=10.2,
                                  leading=14, leftIndent=16, firstLineIndent=-10, spaceAfter=5),
    }


def header_footer(canvas, doc):
    canvas.saveState()
    page = canvas.getPageNumber()
    if page > 1:
        canvas.setStrokeColor(GRAY)
        canvas.setLineWidth(0.4)
        canvas.line(doc.leftMargin, A4[1] - 1.55 * cm, A4[0] - doc.rightMargin, A4[1] - 1.55 * cm)
        canvas.setFont("Helvetica", 8)
        canvas.setFillColor(colors.HexColor("#555555"))
        canvas.drawString(doc.leftMargin, A4[1] - 1.25 * cm, "Knowledge Guided Hybrid Crop Recommendation System")
        canvas.drawRightString(A4[0] - doc.rightMargin - 0.45 * cm, 0.8 * cm, str(page))
    canvas.restoreState()


def section(story, title, body, s):
    story.append(p(title, s["h1"]))
    for paragraph in body:
        story.append(p(paragraph, s["body"]))


def bullet(story, text, s):
    story.append(p("&bull; " + text, s["bullet"]))


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    styles = build_styles()
    doc = BaseDocTemplate(
        str(OUT), pagesize=A4, leftMargin=2.2 * cm, rightMargin=2.0 * cm,
        topMargin=2.05 * cm, bottomMargin=1.75 * cm,
    )
    frame = Frame(doc.leftMargin, doc.bottomMargin, doc.width, doc.height, id="body")
    doc.addPageTemplates([PageTemplate(id="report", frames=[frame], onPage=header_footer)])
    s = styles
    story = []

    # Cover
    story += [Spacer(1, 4.2 * cm), p("Knowledge Guided Hybrid Crop Recommendation System", s["title"]),
              p("Paper 1 Implementation Report", s["subtitle"]), Spacer(1, 1.2 * cm),
              p("A Project Report", s["subtitle"]),
              p("Prepared in the Project Phase I report structure", s["subtitle"]), Spacer(1, 5.0 * cm),
              p("Department of Computer Science and Engineering", s["subtitle"]),
              p("School of Computing", s["subtitle"]),
              p("October 2026", s["subtitle"]), PageBreak()]

    # Contents and abstract
    story += [p("Contents", s["chapter"])]
    for item in [
        "Abstract", "Chapter 1  Introduction", "Chapter 2  Literature Review and Existing System",
        "Chapter 3  Proposed Work and Methodology", "Chapter 4  Experimentation and Result Analysis",
        "Chapter 5  Conclusion and Scope for Further Research", "References",
    ]:
        story.append(p(item, s["toc"]))
    story += [Spacer(1, 0.6 * cm), p("Abstract", s["chapter"]),
              p("This report documents the Phase I implementation of a knowledge-guided hybrid crop recommendation system. The implementation classifies one of 22 crop labels from seven soil and climate measurements: nitrogen, phosphorus, potassium, temperature, humidity, pH and rainfall. The system combines a gradient-boosted tree branch, an FT-Transformer branch and a crop-profile knowledge matrix through an adaptive fusion module. The project also retains conventional baselines so that the hybrid design can be evaluated against simpler alternatives rather than assumed to be superior.", s["body"]),
              p("The verified evidence consists of complete five-fold out-of-fold predictions for six models and a separate five-seed controlled comparison of hybrid variants. On the five-fold protocol, XGBoost obtained the highest pooled out-of-fold accuracy and macro-F1, both 99.50%. KG-Hybrid obtained 99.32% for both measures. The report therefore presents KG-Hybrid as an evaluated hybrid representation model, not as the best-performing model on the recorded cross-validation protocol. The five-seed ablation results are internally consistent but use repeated held-out partitions and should be interpreted as comparative evidence rather than an untouched final generalization test. The system is limited to a balanced benchmark dataset and does not yet establish field performance, regional suitability or economic recommendation quality.", s["body"]),
              p("Keywords: crop recommendation, tabular learning, XGBoost, FT-Transformer, knowledge-guided fusion, agricultural decision support", s["small"]), PageBreak()]

    # Chapter 1
    story += [p("Chapter 1", s["chapter"]), p("Introduction", s["chapter"])]
    section(story, "1.1 Background", [
        "Crop selection depends on interactions among soil nutrients, acidity, moisture conditions and weather. Digital decision-support systems use such information to narrow crop options and make advisory work more consistent. The Food and Agriculture Organization describes soil and crop decision-support systems as tools that translate soil and crop information into actionable outputs, while noting that recommendations must ultimately be tested under real farming conditions [1].",
        "The Paper 1 implementation addresses the prediction part of this broader decision process. It receives a compact tabular observation and predicts a crop label. It does not prescribe fertilizer, estimate profit, verify a farm's local constraints, or replace agronomic advice. Those boundaries matter because high benchmark accuracy alone does not establish practical suitability for a particular farm.",
    ], s)
    section(story, "1.2 Problem Addressed", [
        "The project addresses multi-class crop-label prediction from seven numerical soil and climate variables. Conventional tabular models can perform strongly on this task, but they do not explicitly expose a mechanism for combining learned feature representations with a crop-profile prior. Conversely, a neural model can learn feature interactions but may not outperform a well-tuned tree ensemble on structured tabular data. The implementation therefore evaluates both families rather than assuming that a more complex model is automatically better.",
    ], s)
    section(story, "1.3 Objectives", [], s)
    for text in [
        "Build a reproducible crop-label classification workflow for 22 classes using seven measured inputs.",
        "Implement and compare logistic regression, decision tree, random forest, XGBoost, FT-Transformer and KG-Hybrid models under shared folds.",
        "Construct a fusion model that combines XGBoost leaf representations, FT-Transformer features and a crop-profile knowledge matrix.",
        "Retain saved predictions and independently recompute reported metrics from those artifacts.",
    ]:
        bullet(story, text, s)
    section(story, "1.4 Scope of the Project", [
        "This report covers the local crop-recommendation implementation, its dataset, model architecture, baselines, controlled ablations and recorded evaluation. Later-stage distributed, privacy and deployment work is outside its scope. The report is organized around the supplied Phase I template: Chapters 1 and 2 define the problem and evidence base; Chapter 3 explains the system; Chapter 4 reports verified results and their limits; Chapter 5 concludes the Phase I work.",
    ], s)
    story.append(PageBreak())

    # Chapter 2
    story += [p("Chapter 2", s["chapter"]), p("Literature Review and Existing System", s["chapter"])]
    section(story, "2.1 Existing Research and Tools", [
        "Gradient-boosted decision trees remain strong baselines for tabular classification. XGBoost is a scalable implementation of tree boosting [2]. Random forests provide another ensemble baseline based on independently randomized trees [3]. Both are included in the project because they provide an important reference point for a small, structured dataset.",
        "For neural tabular learning, the project uses an FT-Transformer branch. Gorishniy et al. introduced FT-Transformer as a Transformer adaptation for tabular data and cautioned that no single model family is universally best across tabular tasks [4]. That caution is reflected in the evaluation design: the hybrid model is compared against tree and neural baselines using the same completed folds.",
        "The system is related to agricultural decision support, where soil and crop information are converted into an advisory output. In this project, the output is a dataset-level crop label. The implementation should not be read as a location-specific farm recommendation engine because the benchmark input has no farm identifier, market variable, cultivar record or seasonal production outcome.",
    ], s)
    section(story, "2.2 Research Gap and Scope for Improvement", [
        "The practical gap addressed in Phase I is not a lack of classifiers. It is the need for an auditable comparison in which a hybrid model can be inspected beside simpler baselines. The implementation records shared folds, training-only scaling, out-of-fold predictions, stored probability arrays and prediction-derived metrics. It also includes a crop-profile matrix whose role can be tested through controlled variants.",
        "Several limits remain. The active crop-profile weights are manually specified and their exact cell-level institutional derivation is not independently verified. The benchmark is balanced, with 100 samples per class, and may be easier than heterogeneous field data. The saved ablation prediction files do not retain sample identifiers, so their sample-level identity checks cannot be reconstructed. These are reporting constraints, not reasons to hide the evidence.",
    ], s)
    section(story, "2.3 Problem Statement and Contributions", [
        "Given a seven-variable soil and climate observation, develop and evaluate an auditable classifier that predicts one of 22 benchmark crop classes while examining whether a knowledge-guided fusion mechanism changes performance or calibration relative to conventional baselines.",
    ], s)
    for text in [
        "A modular KG-Hybrid implementation that combines XGBoost leaf embeddings, FT-Transformer features and sample-conditioned crop-profile attention.",
        "A complete recovered five-fold out-of-fold evaluation covering six models and 13,200 model-prediction rows.",
        "A controlled five-seed hybrid comparison with saved predictions, probability arrays, parameter counts and paired tests.",
        "An audit package that recomputes metrics from saved prediction artifacts and records limitations alongside validated results.",
    ]:
        bullet(story, text, s)
    story.append(PageBreak())

    # Chapter 3
    story += [p("Chapter 3", s["chapter"]), p("Proposed Work and Methodology", s["chapter"])]
    section(story, "3.1 Proposed Work", [
        "The proposed system is a multi-branch classifier. A boosted-tree model converts an observation into terminal-leaf identifiers. These identifiers are embedded into a learned XGBoost representation. In parallel, the same scaled numerical features enter an FT-Transformer. The two representations are concatenated and fused with a sample-conditioned context derived from a 22 by 7 crop-profile matrix. A final classifier converts the fused representation into class probabilities.",
    ], s)
    section(story, "3.2 Methodology Overview", [
        "Figure 1 shows the Paper 1 pipeline. The critical evaluation rule is that feature scaling is fitted on training data only. In five-fold cross-validation, this occurs separately inside each fold. In the controlled five-seed study, the data are partitioned into stratified 70%, 15% and 15% training, validation and test subsets. The validation subset is used for early stopping, and test predictions are retained as artifacts for later verification.",
    ], s)
    story += [PageBreak(), Pipeline(), PageBreak()]
    section(story, "3.2.1 Dataset Selection and Preparation", [
        "The primary dataset file contains 2,200 observations, eight columns and 22 crop labels. The inputs are N, P, K, temperature, humidity, pH and rainfall. Each crop class contains 100 observations, making the class distribution balanced. Labels are encoded before model training. The project stores stratified split information and applies StandardScaler only after the training partition has been identified.",
        "The dataset is appropriate for controlled algorithm comparison because each model sees the same variables and class structure. It is not a substitute for multi-season, location-specific field records. The report treats it as a benchmark classification dataset and avoids claims about actual yield, profitability or farm adoption.",
    ], s)
    section(story, "3.2.2 Algorithm and Model Design", [
        "Let x denote the seven standardized inputs. The XGBoost branch maps x to terminal leaf identifiers l and then to a learned embedding e_xgb(l). The FT-Transformer produces a numeric representation e_ft(x). Their concatenation is z = [e_xgb(l), e_ft(x)]. The fusion module computes a gate g = sigmoid(MLP(z)). It also encodes crop profiles from the knowledge matrix K, compares a query projected from z with those profiles, and produces a weighted context vector c. The fused representation is LayerNorm(g multiplied by z plus beta multiplied by the projected c). A classifier maps the fused vector to 22 logits.",
        "The knowledge matrix is used as a learned compatibility prior, not as a physical rule that directly constrains sensor values. In the anchored-trainable policy, deviations from the stored prior receive a mean-squared anchor penalty. In the fixed policy, the matrix does not change. The comparative study contains variants without K, without the adaptive gate and without individual representation branches so that the architecture can be examined component by component.",
    ], s)
    section(story, "3.2.3 Tools and Technologies", [
        "The implementation is written in Python. The recorded requirements include NumPy, pandas, scikit-learn, PyTorch, XGBoost, SciPy, matplotlib, seaborn, FastAPI and Uvicorn. PyTorch implements the FT-Transformer and fusion model; scikit-learn provides preprocessing, classical baselines and metrics; XGBoost supplies the boosted-tree branch. The report does not claim a single hardware configuration because the final audited evidence does not retain one complete hardware manifest.",
    ], s)
    story.append(PageBreak())

    # Chapter 4
    story += [p("Chapter 4", s["chapter"]), p("Experimentation and Result Analysis", s["chapter"])]
    section(story, "4.1 Experimental Setup", [
        "The final experimental validation recomputed metrics from saved prediction artifacts without retraining models or changing the original runs. The audit found complete five-fold out-of-fold predictions for Decision Tree, Logistic Regression, Random Forest, XGBoost, FT-Transformer and KG-Hybrid. Each model has 2,200 out-of-fold rows, totaling 13,200 evaluated model-prediction rows. Sample identifiers, labels and fold assignments were checked for the cross-validation evidence.",
        "The baseline set consists of logistic regression, a decision tree, random forest, XGBoost and FT-Transformer. The KG-Hybrid model is evaluated on the same five folds. Accuracy and macro-F1 are reported because the dataset is balanced, while stored probabilities also enable calibration measures such as expected calibration error, Brier score and negative log-likelihood in the recorded experiments.",
    ], s)
    section(story, "4.2 Verified Five Fold Cross Validation Results", [
        "Table 1 contains metrics recomputed from complete out-of-fold predictions. XGBoost is the highest-ranked model in pooled accuracy and macro-F1. KG-Hybrid is close, but lower on this protocol. This result must be reported directly: the available evidence does not support a claim that KG-Hybrid outperforms XGBoost on the completed five-fold cross-validation study.",
    ], s)
    cv = rows(EVIDENCE / "tables" / "table2_cv_results_all_models.csv")
    cv_data = [[p("Model", s["thead"]), p("Completed folds", s["thead"]), p("OOF samples", s["thead"]), p("Pooled accuracy", s["thead"]), p("Pooled macro F1", s["thead"])]]
    for r in cv:
        cv_data.append([p(r["Model"].replace("Baseline", " baseline").replace("Proposed", ""), s["small"]),
                        p(r["Folds_Complete"], s["small"]), p(r["OOF_Samples"], s["small"]),
                        p(f"{float(r['Pooled_OOF_Accuracy'])*100:.2f}%", s["small"]),
                        p(f"{float(r['Pooled_OOF_Macro_F1'])*100:.2f}%", s["small"])])
    story += [table(cv_data, [4.5 * cm, 2.3 * cm, 2.5 * cm, 3.4 * cm, 3.5 * cm]),
              p("Table 1. Independently recomputed complete five-fold out-of-fold results.", s["small"])]
    section(story, "4.3 Controlled Five Seed Comparison", [
        "A separate controlled study used five seeds (42, 123, 2024, 3407 and 9999), stratified 70/15/15 partitions, a training-only scaler, a maximum of 25 epochs and validation negative-log-likelihood early stopping. The hybrid variants share the seed-specific booster, FT backbone, leaf-embedding initialization, classifier initialization and training budget. The table below reports the stored five-seed means and standard deviations.",
    ], s)
    ab = rows(EVIDENCE / "tables" / "table1_ablation_5seed_validated.csv")
    ab_data = [[p("Variant", s["thead"]), p("Configuration", s["thead"]), p("Accuracy mean +/- SD", s["thead"]), p("Macro F1 mean +/- SD", s["thead"])]]
    for r in ab:
        ab_data.append([p(r["Variant"], s["small"]), p(r["Configuration"], s["small"]),
                        p(r["Accuracy (Mean ± SD)"], s["small"]), p(r["Macro-F1 (Mean ± SD)"], s["small"])])
    story.append(PageBreak())
    story += [table(ab_data, [1.7 * cm, 4.6 * cm, 4.2 * cm, 4.5 * cm], font_size=7.7),
              p("Table 2. Five-seed controlled comparison. Values are comparative repeated-split evidence.", s["small"])]

    story += [p("Chapter 4 Continued", s["chapter"])]
    section(story, "4.4 Analysis of Results", [
        "The verified cross-validation comparison shows that the task is highly separable under the available benchmark: XGBoost achieved 99.50% pooled accuracy and macro-F1, Random Forest achieved 99.00%, FT-Transformer achieved 98.18%, and KG-Hybrid achieved 99.32%. The result is consistent with the broader tabular-learning literature, which does not identify a universally best deep model for all structured datasets [4]. The hybrid branch is competitive, but complexity alone does not justify a superiority claim.",
        "The controlled five-seed comparison reinforces that conclusion. The full KG-Hybrid configuration averaged 99.39% accuracy. Several reduced configurations had similar or higher mean accuracy, including the configuration without auxiliary losses at 99.76%. The recorded paired tests did not establish a statistically significant accuracy difference between the full variant and the ablations after the documented multiple-comparison procedure. This does not mean that the fusion mechanism is useless; it means the recorded experiment does not establish an accuracy advantage for every component on this dataset.",
        "The full gated variant had the lowest mean expected calibration error in the controlled comparison, 0.004269. However, its calibration improvement over XGBoost was not statistically significant after the recorded familywise adjustment. A separate corrected single held-out checkpoint has 99.70% accuracy but ECE 0.121839. Those values come from a different protocol and are not substituted for the cross-validation findings.",
    ], s)
    section(story, "4.5 Observations and Limitations", [], s)
    for text in [
        "The 2,200-row benchmark is perfectly balanced at 100 observations per class. This supports clean comparison but does not reflect class imbalance often seen in farm records.",
        "Complete out-of-fold cross-validation evidence is available for all six models, and its sample identities were verified against the raw dataset.",
        "The five-seed ablation metrics match all 45 saved prediction files, but those files do not contain sample identifiers. Their repeated held-out partitions also cannot be treated as an untouched post-selection test.",
        "The crop-profile prior is manually specified. It is an auditable software artifact, but its exact cell-level institutional derivation remains unverified.",
        "The system predicts crop labels from seven variables. It does not validate cultivar, planting density, disease pressure, irrigation access, price, farmer preference or local policy constraints.",
    ]:
        bullet(story, text, s)
    section(story, "4.6 Reproducibility", [
        "The final validation package records that 99 of 99 regression tests, 8 of 8 API integration checks, 35 of 35 model-seed results and 11,550 of 11,550 held-out predictions were checked in the stored project audit. The report's quantitative tables were drawn from the final experimental validation package, which independently recomputes metrics from saved predictions. Reproduction should use a new output directory for new training runs so that historical evidence is preserved.",
    ], s)
    # Conclusion
    story += [p("Chapter 5", s["chapter"]), p("Conclusion and Scope for Further Research", s["chapter"])]
    section(story, "5.1 Conclusion", [
        "Paper 1 delivers a functioning and auditable crop-label classification implementation. The system integrates tree-leaf embeddings, an FT-Transformer and sample-conditioned crop-profile attention in a KG-Hybrid architecture. It also retains strong classical baselines and verifies saved prediction evidence rather than relying only on training logs.",
        "The most important empirical conclusion is narrow and evidence-based. On the complete five-fold out-of-fold protocol, XGBoost is the strongest recorded model at 99.50% pooled accuracy and macro-F1. KG-Hybrid achieves 99.32% for both metrics and remains competitive, but it is not the top model under this protocol. The controlled comparison shows that the full hybrid design has low average calibration error, while the available tests do not establish a statistically significant clean-accuracy advantage over the simpler variants.",
    ], s)
    section(story, "5.2 Scope for Further Research", [
        "The next research steps within the Paper 1 topic are to validate the system on real, geographically diverse and seasonally separated farm data; derive every knowledge-matrix value from traceable agronomic sources; pre-register a final untouched evaluation protocol; and evaluate recommendation usefulness with agronomists and farmers. Future testing should include cultivar, management, market and risk variables before the system is described as a field-deployment recommendation tool.",
    ], s)
    section(story, "References", [], s)
    refs = [
        "[1] Food and Agriculture Organization of the United Nations, \"Decision Support Systems,\" SoilFER Programme, 2026. Available: https://www.fao.org/in-action/soilfer/in-action/decision-support-systems-%28dss/en)",
        "[2] T. Chen and C. Guestrin, \"XGBoost: A Scalable Tree Boosting System,\" Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, pp. 785-794, 2016, doi: 10.1145/2939672.2939785.",
        "[3] L. Breiman, \"Random Forests,\" Machine Learning, vol. 45, pp. 5-32, 2001, doi: 10.1023/A:1010933404324.",
        "[4] Y. Gorishniy, I. Rubachev, V. Khrulkov and A. Babenko, \"Revisiting Deep Learning Models for Tabular Data,\" Advances in Neural Information Processing Systems, 2021. Available: https://arxiv.org/abs/2106.11959.",
        "[5] KG-Hybrid project evidence package, Crop_recommendation.csv, source code, saved out-of-fold predictions and final experimental validation artifacts, accessed October 2026.",
    ]
    for ref in refs:
        story.append(p(ref, s["small"]))
    story.append(PageBreak())
    story += [p("Appendix A", s["chapter"]), p("Evidence Summary", s["chapter"]),
              p("The final validation report recomputed metrics only from saved prediction artifacts. It did not train a model, replace a checkpoint or manufacture a missing measurement. The selected cross-validation evidence is the isolated recovery directory containing complete neural and classical out-of-fold predictions. The older incomplete neural cross-validation directory is preserved separately and is not used for the reported comparison.", s["body"]),
              p("For practical use, the project source contains scripts for model training, evaluation, audit and evidence generation. The report does not reproduce raw code because the Phase I template reserves the appendix for project code and the active implementation is maintained in the project repository.", s["body"])]

    doc.build(story)
    print(OUT)


if __name__ == "__main__":
    main()
