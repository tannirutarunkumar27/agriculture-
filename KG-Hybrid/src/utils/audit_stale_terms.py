import os
import json
from pathlib import Path

search_terms = [
    '4,042,120', '4042120', '88,400', '88400', '88,554', '88554',
    '26.76', '1.94', '84.24', '89.21', '96.80', '99.70', '90.00',
    'hard-coded', 'earlier manuscript', 'should be rerun', 'final audit',
    'not retained', 'calibration split'
]

results = {t: [] for t in search_terms}
base_dir = Path(r"d:\Final Project\implementation_for _crop recommendation")

targets = [base_dir / "KG-Hybrid", base_dir / "manuscript"]

for target in targets:
    for root, dirs, files in os.walk(target):
        if any(p in root for p in ['.venv', '.venv-1', '.git', '__pycache__', '.system_generated']):
            continue
        for f in files:
            if f.endswith(('.py', '.tex', '.json', '.csv', '.md', '.txt')):
                fpath = os.path.join(root, f)
                try:
                    with open(fpath, 'r', encoding='utf-8', errors='ignore') as fp:
                        for line_no, line in enumerate(fp, start=1):
                            for t in search_terms:
                                if t.lower() in line.lower():
                                    results[t].append({
                                        'file': os.path.relpath(fpath, base_dir),
                                        'line': line_no,
                                        'text': line.strip()[:120]
                                    })
                except Exception:
                    pass

out_path = base_dir / "KG-Hybrid" / "outputs" / "tables" / "stale_terms_audit.json"
with open(out_path, "w", encoding='utf-8') as fp:
    json.dump(results, fp, indent=2)

print("[OK] Stale terms audit complete!")
