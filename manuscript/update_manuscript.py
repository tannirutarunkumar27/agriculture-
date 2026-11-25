import sys
from pathlib import Path

tex_path = Path(r"d:\Final Project\implementation_for _crop recommendation\manuscript\manuscript.tex")
content = tex_path.read_text(encoding="utf-8")

# 1. Remove duplicate Section V if present
dup_start = content.find(r"\section{Robustness and Generalization Analysis}")
dup_second = content.find(r"\section{Robustness and Generalization Analysis}", dup_start + 40)
if dup_second != -1:
    print("Found duplicate Section V at index", dup_second)
    content = content[:dup_start] + content[dup_second:]
    print("Cleaned duplicate Section V.")

# 2. Insert Calibration Subsection into Section IV (right after Baseline Comparison)
baseline_end_marker = "\\end{enumerate}\n\n\\subsection{Ablation Study}"
calibration_sub = r"""\end{enumerate}

\subsection{Probability Calibration and Reliability Analysis}

Beyond raw discriminative classification metrics, trustworthy decision support in precision agriculture requires that a model's predicted class probabilities faithfully reflect its true empirical likelihood of correctness. We conduct a rigorous probability calibration evaluation across all six evaluated classifiers on the held-out test partition ($N_{\text{test}} = 330$ across all 22 crop classes), using the frozen model checkpoints without applying any post-hoc calibration adjustments (such as temperature scaling or Platt scaling) and without creating an artificial calibration split.

Calibration quality is quantified using three complementary metrics:
\begin{enumerate}
    \item \textbf{Expected Calibration Error (ECE):} Computed using $M=10$ equal-width confidence bins partitioned over $[0, 1]$. For each bin $\mathcal{B}_m$, the weighted absolute difference between empirical accuracy $\text{acc}(\mathcal{B}_m)$ and average predicted confidence $\text{conf}(\mathcal{B}_m)$ is calculated, where empty bins ($|\mathcal{B}_m|=0$) naturally contribute zero weight:
    \begin{equation}
    \text{ECE} = \sum_{m=1}^{M} \frac{|\mathcal{B}_m|}{N} \left| \text{acc}(\mathcal{B}_m) - \text{conf}(\mathcal{B}_m) \right|
    \end{equation}
    \item \textbf{Brier Score:} The multiclass quadratic loss measuring the mean squared distance between the full predicted probability vector $\mathbf{p}_i \in \Delta^{22}$ and the one-hot target indicator vector $\mathbf{y}_i$:
    \begin{equation}
    \text{BS} = \frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{22} \left( p_{i,c} - y_{i,c} \right)^2
    \end{equation}
    \item \textbf{Negative Log-Likelihood (NLL):} The cross-entropy loss evaluated strictly on the probability assigned to the ground-truth class $y_i$ with numerical stability constant $\epsilon = 10^{-12}$:
    \begin{equation}
    \text{NLL} = -\frac{1}{N} \sum_{i=1}^{N} \ln\left( p_{i, y_i} + \epsilon \right)
    \end{equation}
\end{enumerate}

Table~\ref{tab:calibration} reports the dynamic calibration metrics obtained on the held-out test partition. Additionally, Fig.~\ref{fig:reliability_diagram} displays the corresponding reliability diagrams, where empirical accuracy is plotted against mean predicted confidence across all populated confidence bins alongside the ideal diagonal reference line ($y=x$).

\begin{table}[t]
\centering
\caption{Probability Calibration Metrics on the Held-Out Test Set ($N=330$, 22 Classes)}
\label{tab:calibration}
\begin{tabular}{lcccc}
\hline
\textbf{Model} & \textbf{Top-1 Accuracy} & \textbf{ECE} & \textbf{Brier Score} & \textbf{NLL} \\
\hline
Logistic Regression & 0.9788 & 0.144003 & 0.074166 & 0.203679 \\
Decision Tree & 0.9818 & 0.018182 & 0.036364 & 0.502382 \\
Random Forest & 0.9970 & 0.045303 & 0.014935 & 0.054992 \\
XGBoost Baseline & 0.9909 & 0.016223 & 0.011897 & 0.033439 \\
FT-Transformer Baseline & 0.9879 & 0.006810 & 0.020304 & 0.044111 \\
\textbf{KG-Hybrid (Proposed)} & \textbf{0.9909} & \textbf{0.130556} & \textbf{0.045856} & \textbf{0.165085} \\
\hline
\end{tabular}
\end{table}

\begin{figure}[t]
    \centering
    \includegraphics[width=1.0\columnwidth]{reliability_diagram.pdf}
    \caption{Reliability diagrams for the six evaluated classifiers on the held-out test partition. Each curve is constructed from populated confidence bins using mean predicted confidence and empirical accuracy.}
    \label{fig:reliability_diagram}
\end{figure}

\textbf{Interpretation and Analysis:}
The empirical evaluation reveals important nuances regarding confidence quality across model families:
\begin{itemize}[leftmargin=1.2em]
    \item \textbf{FT-Transformer Calibration:} FT-Transformer exhibits the lowest Expected Calibration Error ($\text{ECE} = 0.006810$), following the ideal diagonal closely in the high-confidence bins where its predictions are concentrated.
    \item \textbf{XGBoost Probability Quality:} XGBoost achieves the lowest multiclass Brier score ($0.011897$) and lowest NLL ($0.033439$), demonstrating strong probabilistic separation across all 22 crop classes.
    \item \textbf{KG-Hybrid Behavior:} Although KG-Hybrid achieves high discriminative classification performance ($99.09\%$ top-1 accuracy on the test set), its uncalibrated predictive probabilities yield an $\text{ECE}$ of $0.130556$, a Brier score of $0.045856$, and an $\text{NLL}$ of $0.165085$.
    \item \textbf{Classification vs. Calibration Divergence:} These results demonstrate that strong discriminative classification accuracy does not automatically translate into well-calibrated confidence estimates under standard unregularized training. The dual-branch feature concatenation and gating mechanism prioritize boundary margin separation over posterior calibration, indicating that confidence scores in raw hybrid models should be coupled with uncertainty gating (such as MC-Dropout) or post-hoc recalibration before high-stakes field deployment.
\end{itemize}

\subsection{Ablation Study}"""

if baseline_end_marker in content:
    content = content.replace(baseline_end_marker, calibration_sub)
    print("Inserted Calibration Subsection into Section IV.")
else:
    print("Warning: baseline_end_marker not found!")

# 3. Add Calibration Discussion into Section VII
interp_marker = r"\subsubsection{Role of Knowledge Guidance}"
calib_discussion = r"""\subsubsection{Discriminative Accuracy vs. Confidence Calibration}
A critical finding from our empirical evaluation is the distinction between classification performance and probability calibration. While KG-Hybrid matches or exceeds baseline models in top-1 accuracy ($99.09\%$) and robust generalization under noise and missing features, its uncalibrated posterior probabilities exhibit a higher ECE ($0.130556$) than standalone FT-Transformer ($0.006810$) and XGBoost ($0.016223$). This divergence highlights that optimizing neural architectures with auxiliary distillation and contrastive objectives sharpens decision boundaries for accurate classification but does not inherently guarantee calibrated posterior confidence estimates. In practical agricultural deployment, this motivates our incorporation of Monte Carlo Dropout uncertainty thresholds to explicitly flag low-certainty recommendations rather than relying solely on raw softmax peak values.

\subsubsection{Role of Knowledge Guidance}"""

if interp_marker in content:
    content = content.replace(interp_marker, calib_discussion)
    print("Inserted Calibration Discussion into Section VII.")

# 4. Add Calibration Limitation to Section VIII
limit_marker = "\\subsubsection{Methodological Limitations}\n\n\\begin{itemize}[leftmargin=1.2em]"
calib_limitation = r"""\subsubsection{Methodological Limitations}

\begin{itemize}[leftmargin=1.2em]
    \item \textbf{Probability Calibration Optimization:} The current training objective was not explicitly optimized for probability calibration. Future work should therefore evaluate calibration-aware objectives and post-hoc calibration methods (such as temperature scaling, vector scaling, or Dirichlet calibration), preferably across external and geographically diverse datasets."""

if limit_marker in content:
    content = content.replace(limit_marker, calib_limitation)
    print("Inserted Calibration Limitation into Section VIII.")

tex_path.write_text(content, encoding="utf-8")
print("Successfully saved updated manuscript.tex")

# Also sync to Downloads copy if present
dl_path = Path(r"C:\Users\tanne\Downloads\KG_Hybrid_Updated.tex")
if dl_path.exists():
    try:
        dl_path.write_text(content, encoding="utf-8")
        print(r"Successfully synced to C:\Users\tanne\Downloads\KG_Hybrid_Updated.tex")
    except Exception as e:
        print(f"Sync to Downloads skipped: {e}")
