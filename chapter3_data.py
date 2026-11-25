import os
from docx.shared import Inches, Pt
from report_builder_utils import add_styled_heading, add_p, add_bullet, add_table_data, add_image_if_exists

def build_chapter_3(doc, img_dir):
    add_styled_heading(doc, "CHAPTER 3: PROPOSED WORK", level=1)
    
    add_styled_heading(doc, "3.1 Architectural Overview of KG-Hybrid", level=2)
    add_p(doc, "The proposed KG-Hybrid framework establishes a knowledge-guided, trustworthy deep learning architecture for precision agricultural crop recommendation. Designed to transcend the limitations of purely statistical tabular models, KG-Hybrid operates by ingesting seven localized continuous measurements: soil Nitrogen (N), Phosphorus (P), Potassium (K), ambient Temperature, atmospheric Humidity, soil pH, and cumulative Rainfall. The framework outputs a calibrated probability distribution across 22 candidate agricultural crop classes.")

    add_p(doc, "At its conceptual core, KG-Hybrid bridges two historically divided machine learning paradigms through the governing constraint of explicit biological science:")
    add_bullet(doc, "1. Branch A (XGBoost Leaf Embeddings): ", "An ensemble of 200 gradient-boosted decision trees partitions the feature space into localized hyper-rectangles. Rather than discarding tree structure, we extract the discrete leaf-node trajectory reached by a sample in each tree and project these indices into continuous vector space via a trainable EmbeddingBag layer, capturing sharp non-linear decision boundaries.")
    add_bullet(doc, "2. Branch B (FT-Transformer Attention): ", "A 4-layer Feature Tokenizer Transformer projects continuous scalar features into 64-dimensional token embeddings. A prepended [CLS] classification token interacts with all feature tokens through multi-head self-attention, modeling continuous, multi-order, cross-feature correlations.")
    add_bullet(doc, "3. Agronomic Knowledge Matrix Prior: ", "A normalized 22 × 7 matrix K compiled from Food and Agriculture Organization (FAO) and Indian Council of Agricultural Research (ICAR) compendiums is injected directly into feature gating and gated fusion layers, ensuring that neural representations conform to established agronomic tolerance envelopes.")

    add_image_if_exists(doc, os.path.join(img_dir, "fig1_kg_hybrid_detailed_architecture.png"), "Figure 3.1: Detailed Architecture of the Knowledge-Guided Agriculture-Aware Attention Network (KGAAN)", width=Inches(5.8))

    add_styled_heading(doc, "3.1.1 Methodological Progression", level=3)
    add_p(doc, "Table 3.1 outlines the systematic eight-stage engineering workflow implemented across the project:")

    t31_headers = ["Methodological Stage", "Core Technical Operations", "Generated Technical Artifacts"]
    t31_rows = [
        ["1. Data Ingestion & Sanitization", "Stratified 70:15:15 partitioning; fit StandardScaler and LabelEncoder strictly on training split", "Scaled matrices X_train, X_val, X_test, scaler.pkl, label_encoder.pkl"],
        ["2. Agronomic Knowledge Engineering", "Curate physiological crop thresholds from FAO/ICAR guidelines into normalized [0, 1] scores", "Domain Knowledge Matrix K ∈ R^(22x7), Heatmap matrix visualization"],
        ["3. Tree Ensemble Training", "Train XGBoost (T=200, depth=6) on training split; extract discrete leaf indices; freeze trees", "xgboost_model.pkl, leaf paths l ∈ Z^200, EmbeddingBag (d_xgb=128)"],
        ["4. Tabular Transformer Pre-training", "Tokenize 7 features to 64-D; prepend [CLS]; pre-train 4 Transformer layers (H=8) for 100 epochs", "Pre-trained FT-Transformer backbone weights, contextual vectors f_ft"],
        ["5. Knowledge Gated Fusion & Tuning", "Concatenate embeddings; compute dynamic sigmoid gate; inject knowledge bias; fine-tune", "Trained kg_hybrid_best.pth checkpoint, loss/accuracy convergence logs"],
        ["6. Comparative & Ablation Benchmarking", "Benchmark against 6 baselines; execute 7 ablation variants (A1–A7); multi-seed validation", "Audited performance CSV tables, confusion matrices, ablation charts"],
        ["7. Robustness Stress-Testing", "Evaluate 20%, 40%, 60% feature dropout with KNN (k=5); test Gaussian noise and 1.5x shift", "Robustness degradation curves, missing-data recovery tables"],
        ["8. Explainability, UQ & Edge Serving", "Extract TreeSHAP; run M=10 MC Dropout passes; build early exit (τ=0.90) and FastAPI microservice", "SHAP attribution plots, reliability diagrams, lightweight 4.04M model, REST API"]
    ]
    add_table_data(doc, "Table 3.1: Methodological Progression of the KG-Hybrid Framework", t31_headers, t31_rows, [Inches(1.5), Inches(3.2), Inches(2.1)])

    add_image_if_exists(doc, os.path.join(img_dir, "fig4_training_inference_pipeline.png"), "Figure 3.2: End-to-End Training and Inference Pipeline of KG-Hybrid", width=Inches(5.6))

    add_styled_heading(doc, "3.2 Dataset Selection and Agronomic Profiles", level=2)
    add_p(doc, "The empirical evaluation is based on the benchmark Crop Recommendation Dataset (Crop_recommendation.csv). It contains 2,200 instances perfectly balanced across 22 distinct agricultural crop categories (exactly 100 observations per class). Table 3.2 summarizes the statistical parameters and physiological roles of the seven continuous features:")

    t32_headers = ["Input Feature Name", "Physiological Role in Plant Biology", "Observed Range", "Mean (μ)", "Std (σ)", "Unit"]
    t32_rows = [
        ["Nitrogen (N)", "Essential building block of proteins, amino acids, and leaf chlorophyll", "0.00 – 140.00", "50.55", "36.92", "mg/kg"],
        ["Phosphorus (P)", "Drives cellular ATP energy transfer, root architecture, and flowering", "5.00 – 145.00", "53.36", "32.99", "mg/kg"],
        ["Potassium (K)", "Regulates stomatal transpiration, cellular turgor, and disease defense", "5.00 – 205.00", "48.15", "50.65", "mg/kg"],
        ["Temperature", "Dictates metabolic reaction kinetics and growing degree day accumulation", "8.83 – 43.68", "25.62", "5.06", "°C"],
        ["Humidity", "Governs atmospheric vapor pressure deficit and transpirational cooling", "14.26 – 99.98", "71.48", "22.26", "%"],
        ["Soil pH", "Controls mineral nutrient solubility, chemical toxicity, and microbial health", "3.50 – 9.94", "6.47", "0.77", "-log[H+]"],
        ["Rainfall", "Supplies plant root hydration and dissolved soil mineral transport", "20.21 – 298.56", "103.46", "54.96", "mm"]
    ]
    add_table_data(doc, "Table 3.2: Agronomic Description and Ranges of Benchmark Input Features", t32_headers, t32_rows, [Inches(1.2), Inches(2.2), Inches(1.1), Inches(0.8), Inches(0.8), Inches(0.7)])

    add_styled_heading(doc, "3.2.1 Agronomic Characteristics of the 22 Crop Categories", level=3)
    add_p(doc, "The dataset encompasses cereals, legumes, fruits, and commercial cash crops. Understanding their biological classifications illustrates the multi-class complexity of the recommendation task:")
    add_bullet(doc, "Cereals & Grains (Rice, Maize): ", "Rice requires continuous saturated moisture (rainfall > 200 mm) and high nitrogen in warm sub-tropical environments; Maize requires well-drained loam, balanced N-P-K, and moderate precipitation.")
    add_bullet(doc, "Leguminous Pulses (Chickpea, Kidneybeans, Pigeonpeas, Mothbeans, Mungbean, Blackgram, Lentil): ", "Possess symbiotic root nodules hosting Rhizobium bacteria capable of fixing atmospheric nitrogen. They exhibit low nitrogen requirements but require elevated phosphorus for root nodulation and thrive under semi-arid conditions.")
    add_bullet(doc, "Horticultural Fruits (Banana, Mango, Grapes, Apple, Orange, Papaya, Pomegranate, Watermelon, Muskmelon): ", "Exhibit highly specialized climatic envelopes: Apple requires temperate chill hours; Grapes thrive under high potassium and modest humidity; Banana requires heavy potassium and continuous tropical warmth; Watermelon and Muskmelon require sandy loam soils with high summer heat.")
    add_bullet(doc, "Plantation & Fiber Cash Crops (Cotton, Jute, Coconut, Coffee): ", "Cotton requires deep black soils with warm temperatures; Jute requires prolonged monsoon humidity (>80%) and standing water; Coconut thrives in coastal sandy soils with high rainfall; Coffee demands high altitude, acidic soils (pH 5.0–6.0), and temperate shade.")

    t33_headers = ["Partition Subset", "Sample Count", "Percentage Share", "Role in Experimental Protocol"]
    t33_rows = [
        ["Training Partition", "1,540 samples", "70.0%", "Fitting standard scalers, training XGBoost, and fine-tuning neural weights"],
        ["Validation Partition", "330 samples", "15.0%", "Hyperparameter optimization, learning rate scheduling, and early stopping"],
        ["Held-Out Test Partition", "330 samples", "15.0%", "Final uncorrupted benchmark evaluation, ablation verification, and stress testing"],
        ["Total Dataset", "2,200 samples", "100.0%", "Balanced across 22 classes (exactly 15 test samples per crop)"]
    ]
    add_table_data(doc, "Table 3.3: Leak-Free Stratified Data Partition Summary", t33_headers, t33_rows, [Inches(1.8), Inches(1.4), Inches(1.3), Inches(2.3)])

    add_styled_heading(doc, "3.3 Leak-Free Preprocessing Pipeline", level=2)
    add_p(doc, "Data leakage occurs when information from the evaluation or test dataset inadvertently influences data transformation parameters. To ensure absolute experimental integrity, our preprocessing pipeline strictly adheres to train-only fitting:")
    add_p(doc, "1. Label Encoding: Crop target strings are mapped into discrete integer indices c ∈ {0, 1, ..., 21} using a Scikit-Learn LabelEncoder fitted solely on the 1,540 training instances.")
    add_p(doc, "2. Standard Scaling: For continuous features x_f, the sample mean μ_f^(train) and sample variance (σ_f^(train))^2 are computed strictly on the training partition. Feature values across all partitions are normalized via:")
    add_p(doc, "x̃_f = (x_f - μ_f^(train)) / (σ_f^(train) + ε),   where ε = 1e-8 prevents division by zero.")

    add_styled_heading(doc, "3.4 Agronomic Domain Knowledge Engineering", level=2)
    add_p(doc, "Unlike conventional models that treat feature values agnostically, KG-Hybrid incorporates an explicit biological suitability matrix K ∈ R^(22x7). Developed from the FAO Ecocrop database and the ICAR Handbook of Agriculture, each matrix element K_(c,f) represents the normalized suitability of feature f for crop cultivar c:")
    add_p(doc, "K_(c,f) ∈ [0, 1],   where 1.0 indicates optimal physiological growth conditions, and 0.0 indicates a biological stress threshold.")
    add_p(doc, "For example, the knowledge matrix encodes high rainfall affinity for rice (0.95) and jute (0.90), while assigning low rainfall suitability to drought-tolerant mothbeans (0.15) and chickpea (0.20). Legumes receive lower nitrogen suitability scores (reflecting atmospheric fixation), whereas nitrogen-hungry maize and cotton receive scores exceeding 0.85.")

    add_styled_heading(doc, "3.5 Mathematical Architecture of KGAAN", level=2)
    add_p(doc, "The Knowledge-Guided Agriculture-Aware Attention Network (KGAAN) combines five mathematical processing stages:")

    add_styled_heading(doc, "3.5.1 Knowledge-Guided Feature Gate", level=3)
    add_p(doc, "Before feature representations are learned, raw standardized inputs x ∈ R^7 are modulated by a knowledge-informed gating layer:")
    add_p(doc, "g_kg = σ(W_g x + b_g + α K_c^T) ∈ R^7")
    add_p(doc, "where W_g ∈ R^(7x7) and b_g ∈ R^7 are trainable affine parameters, σ(z) = 1 / (1 + exp(-z)) is the sigmoid activation function, K_c is the suitability vector for the class prior, and α = 0.1 scales the domain influence. The modulated feature vector is computed via Hadamard product:")
    add_p(doc, "x̃ = x ⊙ g_kg")

    add_styled_heading(doc, "3.5.2 Branch A: XGBoost Decision Boundaries & Leaf Embeddings", level=3)
    add_p(doc, "XGBoost trains an ensemble of T=200 gradient-boosted decision trees (max depth 6, learning rate 0.1) on the training split. For any input sample x, the decision tree path traverses through split nodes until reaching a terminal leaf node in each tree. The leaf indices form an integer vector:")
    add_p(doc, "l = [l_1, l_2, ..., l_200]^T ∈ Z^200,   where l_t ∈ {1, ..., N_leaves^(t)}")
    add_p(doc, "Rather than relying on hand-engineered one-hot encodings, these discrete leaf indices are projected into continuous space via a trainable nn.EmbeddingBag layer:")
    add_p(doc, "f_xgb = EmbeddingBag(l) ∈ R^(d_xgb),   where d_xgb = 128 (full model) or d_xgb = 32 (lightweight model).")
    add_p(doc, "During joint neural training, the XGBoost tree structures remain completely frozen, providing static, highly reliable spatial decision partitions.")

    add_styled_heading(doc, "3.5.3 Branch B: Feature Tokenizer Transformer (FT-Transformer)", level=3)
    add_p(doc, "To capture continuous, high-order cross-feature correlations, Branch B implements an FT-Transformer:")
    add_p(doc, "1. Feature Tokenizer: Each continuous feature x̃_f is projected into a 64-dimensional token embedding:")
    add_p(doc, "e_f = W_f^(tok) x̃_f + b_f^(tok) ∈ R^64,   f ∈ {1, ..., 7}")
    add_p(doc, "2. [CLS] Token Augmentation: A learnable classification token e_cls ∈ R^64 is prepended to the sequence:")
    add_p(doc, "E^(0) = [e_cls, e_1, e_2, ..., e_7] ∈ R^(8x64)")
    add_p(doc, "3. Multi-Head Self-Attention (MHA): The sequence passes through L=4 Transformer encoder layers. For layer l with H=8 attention heads (d_k = 64/8 = 8):")
    add_p(doc, "Q^(l) = E^(l-1) W_Q^(l),   K^(l) = E^(l-1) W_K^(l),   V^(l) = E^(l-1) W_V^(l)")
    add_p(doc, "Attention(Q, K, V) = softmax(Q K^T / √d_k) V")
    add_p(doc, "E^(l) = LayerNorm(E^(l-1) + MHA(E^(l-1)))")
    add_p(doc, "E^(l) = LayerNorm(E^(l) + FFN(E^(l)))")
    add_p(doc, "4. Contextual Vector Extraction: The final contextual representation is extracted directly from the [CLS] token:")
    add_p(doc, "f_ft = E_cls^(4) ∈ R^64")

    add_styled_heading(doc, "3.5.4 Knowledge-Guided Adaptive Gated Fusion", level=3)
    add_p(doc, "The branch embeddings are concatenated into a joint representation vector:")
    add_p(doc, "z = [f_xgb || f_ft] ∈ R^(128 + 64) = R^192")
    add_p(doc, "An adaptive gating vector is computed to balance branch contributions:")
    add_p(doc, "g = σ(W_fusion z + b_fusion) ∈ R^192")
    add_p(doc, "The domain knowledge matrix is flattened into a 154-dimensional vector and projected into the fusion dimension:")
    add_p(doc, "b_kg = W_kg · vec(K) ∈ R^192")
    add_p(doc, "The representations are synthesized using a stop-gradient branch to prevent destructive gradient interference between the tree embeddings and attention weights:")
    add_p(doc, "f_fused = g ⊙ z + (1 - g) ⊙ stop_grad(z) + β b_kg,   where β = 0.1.")
    add_p(doc, "The class probability distribution is emitted by a multi-layer classification head:")
    add_p(doc, "p_fused = softmax(W_head f_fused + b_head) ∈ Δ^22")

    add_image_if_exists(doc, os.path.join(img_dir, "knowledge_guided_fusion_architecture.png"), "Figure 3.3: Knowledge-Guided Gated Fusion and Bias Injection Mechanism", width=Inches(5.4))

    add_styled_heading(doc, "3.5.5 Multi-Task Auxiliary Training Objectives", level=3)
    add_p(doc, "To prevent the neural network from overfitting small tabular datasets, training optimizes a composite loss function:")
    add_p(doc, "L_total = L_CE + λ1 L_distill + λ2 L_contrast,   where λ1 = 0.5, λ2 = 0.1.")
    add_p(doc, "1. Supervised Cross-Entropy Loss: Penalizes prediction error on ground-truth crop labels:")
    add_p(doc, "L_CE = - Σ (c=1 to 22) y_c ln(p_fused,c)")
    add_p(doc, "2. Symmetric Knowledge Distillation Loss: Enforces distributional alignment between the fused model and individual branches:")
    add_p(doc, "L_distill = KL(p_fused || p_xgb) + KL(p_fused || p_ft) = Σ p_fused,c ln(p_fused,c / p_xgb,c) + Σ p_fused,c ln(p_fused,c / p_ft,c)")
    add_p(doc, "3. Supervised Contrastive Loss: Pulls latent embeddings of identical crop categories close together while repelling differing crops in embedding space:")
    add_p(doc, "L_contrast = - Σ (1/|P(i)|) Σ_(p∈P(i)) ln[ exp(f_i · f_p / τ) / Σ_(a≠i) exp(f_i · f_a / τ) ],   where τ = 0.5 is the temperature hyperparameter.")

    add_styled_heading(doc, "3.6 Two-Stage Training Protocol", level=2)
    add_p(doc, "Optimization proceeds in two distinct stages:")
    add_p(doc, "Stage 1 (FT-Transformer Pre-training): The FT-Transformer backbone is trained independently on the training split for 100 epochs using the AdamW optimizer (learning rate 1e-4, weight decay 1e-5) with standard cross-entropy loss. The model checkpoint exhibiting the lowest validation loss is preserved as backbone initialization.")
    add_p(doc, "Stage 2 (Joint Knowledge-Guided Fine-Tuning): The pre-trained FT-Transformer is assembled alongside the frozen XGBoost model, the trainable EmbeddingBag, the feature gate, and the gated fusion module. Gradients are propagated through all neural parameters while the tree structures remain frozen. Training runs for up to 100 epochs using AdamW with ReduceLROnPlateau scheduling (factor 0.5, patience 5) and EarlyStopping (patience 15). The model converged at epoch 27, having reached its peak validation accuracy of 99.39% at epoch 17.")

    add_styled_heading(doc, "3.7 Dynamic Two-Stage Early-Exit Pipeline", level=2)
    add_p(doc, "In precision agriculture, the majority of field sensor queries represent clear-cut, unambiguous soil profiles. Executing a heavy deep transformer for routine queries wastes computational power and battery life. We introduce a two-stage early-exit cascade:")
    add_p(doc, "Stage 1 (Fast Tree Evaluation): An incoming sample x_new is evaluated by the fast XGBoost model. The top-1 class confidence is calculated: τ = max(p_xgb). If τ ≥ 0.90, the early-exit condition is satisfied, and the XGBoost prediction is returned immediately. In our benchmarks, 96.36% of queries exit at Stage 1, slashing average latency to 5.99 ms.")
    add_p(doc, "Stage 2 (Full Neural KGAAN Evaluation): If confidence falls below 0.90, the sample is routed to the complete KG-Hybrid neural network. The sample undergoes M=10 stochastic forward passes via Monte Carlo Dropout to evaluate predictive uncertainty (σ). If max(σ) > 0.15 or τ < 0.60, a Safety Warning is triggered, prompting the farmer to verify sensor connectivity or conduct laboratory soil testing.")

    add_image_if_exists(doc, os.path.join(img_dir, "fig5_early_exit_flow_diagram.png"), "Figure 3.4: Two-Stage Confidence-Based Early-Exit Decision Flow", width=Inches(5.2))

    add_styled_heading(doc, "3.8 Fault-Tolerance: Missing Data Imputation", level=2)
    add_p(doc, "When field sensors fail, input vectors arrive with missing attributes. To ensure zero test-to-train leakage, we train a K-Nearest Neighbors (KNN, k=5) imputer strictly on the training partition. Missing values are imputed via Euclidean distance weighted neighbor averaging:")
    add_p(doc, "x̃_f = (1/k) Σ_(i∈N_k) x_i^(f),   where N_k represents the k nearest training samples with complete feature f.")

