"""Measure inference time and parameter counts for all models."""
import torch, joblib, time, numpy as np, pandas as pd, sys, json
sys.path.insert(0, '.')
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb_lib
from src.models.ft_transformer.model import FTTransformer
from src.models.hybrid.xgb_leaf_embedder import XGBLeafEmbedder
from src.models.hybrid.knowledge_fusion import KnowledgeGuidedFusion
from src.models.hybrid.kg_hybrid_model import KGHybrid
from src.models.ft_transformer.config import FTConfig

ROOT   = Path('.')
DATA   = ROOT / 'data' / 'processed'
MODELS = ROOT / 'outputs' / 'models'
FEATURES = ['N','P','K','temperature','humidity','ph','rainfall']
N_TRIALS = 200

train_df = pd.read_csv(DATA / 'train.csv')
test_df  = pd.read_csv(DATA / 'test.csv')
X_train = train_df.drop('label', axis=1).values.astype('float32')
y_train = train_df['label'].values
X_test  = test_df.drop('label', axis=1).values.astype('float32')

results = {}

# --- Sklearn baselines ---
print("Training sklearn baselines...")
lr = LogisticRegression(max_iter=1000, random_state=42).fit(X_train, y_train)
dt = DecisionTreeClassifier(random_state=42, max_depth=15).fit(X_train, y_train)
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=1).fit(X_train, y_train)
xgb_clf = joblib.load(MODELS / 'xgboost_model.pkl')

for name, model in [
    ('Logistic Regression', lr),
    ('Decision Tree', dt),
    ('Random Forest', rf),
    ('XGBoost', xgb_clf),
]:
    t0 = time.perf_counter()
    for _ in range(N_TRIALS):
        model.predict(X_test)
    ms = (time.perf_counter() - t0) / N_TRIALS * 1000

    if hasattr(model, 'n_estimators'):
        try:
            params = int(model.n_estimators)
            # RF: approx nodes across all trees
            total_nodes = sum(tree.tree_.node_count for tree in model.estimators_)
            params_str = str(total_nodes) + ' nodes'
        except:
            params_str = str(model.n_estimators) + ' trees'
    elif hasattr(model, 'coef_'):
        p = model.coef_.size + model.intercept_.size
        params_str = str(p)
    elif hasattr(model, 'tree_'):
        params_str = str(model.tree_.node_count) + ' nodes'
    else:
        params_str = 'N/A'

    results[name] = {'inference_ms': round(ms, 3), 'params': params_str}
    print(f"{name}: {ms:.3f}ms  params={params_str}")

# --- FT-Transformer ---
print("Loading FT-Transformer...")
ft = FTTransformer()
ckpt = torch.load(MODELS / 'best_model.pth', map_location='cpu')
ft.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
ft.eval()
Xt = torch.tensor(X_test, dtype=torch.float32)
with torch.no_grad():
    ft(Xt)  # warmup
    t0 = time.perf_counter()
    for _ in range(N_TRIALS):
        ft(Xt)
    ms = (time.perf_counter() - t0) / N_TRIALS * 1000
ft_params = sum(p.numel() for p in ft.parameters())
results['FT-Transformer'] = {'inference_ms': round(ms, 3), 'params': str(ft_params)}
print(f"FT-Transformer: {ms:.3f}ms  params={ft_params:,}")

# --- KG-Hybrid ---
print("Loading KG-Hybrid...")
booster = xgb_clf.get_booster()
NUM_TREES = 4400; MAX_L = 27; TOTAL = NUM_TREES * MAX_L
TREE_OFF  = np.arange(NUM_TREES, dtype=np.int64) * MAX_L
cfg       = FTConfig()
embedder  = XGBLeafEmbedder(num_unique_leaves=TOTAL, num_trees=NUM_TREES, embedding_dim=128)
ft2       = FTTransformer()
ft2.load_state_dict(ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt)
from src.knowledge.authoritative_matrix import load_authoritative_knowledge_matrix
km, _, _  = load_authoritative_knowledge_matrix(DATA)
fusion    = KnowledgeGuidedFusion(dim_xgb=128, dim_ft=cfg.EMBEDDING_DIM,
                                   num_crops=22, num_features=7, knowledge_matrix=km)
kg        = KGHybrid(embedder, ft2, fusion, num_classes=22)
kg_ckpt_path = MODELS / 'kg_hybrid_corrected.pth' if (MODELS / 'kg_hybrid_corrected.pth').exists() else MODELS / 'kg_hybrid_best.pth'
kg.load_state_dict(torch.load(kg_ckpt_path, map_location='cpu'), strict=True)
kg.eval()

df_X  = pd.DataFrame(X_test, columns=FEATURES)
dmat  = xgb_lib.DMatrix(df_X)
raw   = booster.predict(dmat, pred_leaf=True).astype(np.int64) + TREE_OFF
raw   = np.clip(raw, 0, TOTAL - 1)
leaf_t = torch.tensor(raw, dtype=torch.long)
x_t   = torch.tensor(X_test, dtype=torch.float32)

with torch.no_grad():
    kg(leaf_t, x_t)  # warmup
    t0 = time.perf_counter()
    for _ in range(N_TRIALS):
        kg(leaf_t, x_t)
    ms_nn = (time.perf_counter() - t0) / N_TRIALS * 1000

t0 = time.perf_counter()
for _ in range(N_TRIALS):
    booster.predict(dmat, pred_leaf=True)
ms_xgb = (time.perf_counter() - t0) / N_TRIALS * 1000
ms_total = ms_nn + ms_xgb

kg_params = sum(p.numel() for p in kg.parameters())
results['KG-Hybrid'] = {'inference_ms': round(ms_total, 3), 'params': str(kg_params)}
print(f"KG-Hybrid: {ms_total:.3f}ms (nn={ms_nn:.3f}+xgb={ms_xgb:.3f})  params={kg_params:,}")

print("\n=== SUMMARY ===")
for m, v in results.items():
    print(f"  {m:<25}: {v['inference_ms']:>8.3f}ms  params={v['params']}")

out = Path('outputs/tables/inference_params.json')
with open(out, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nSaved -> {out}")
