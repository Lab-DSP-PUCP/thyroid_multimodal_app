import json, os
from pathlib import Path

SENTINELS = {"", "-", "none", "null", "n/a"}
def count_present(item):
    keys = ["composition","echogenicity","margins","calcifications","sex","age"]
    def ok(v): 
        s = str(v).strip().lower()
        return s not in SENTINELS
    return sum(1 for k in keys if ok(item.get(k)))

p = Path("uploads/_history.json")
data = json.loads(p.read_text(encoding="utf-8"))
for it in data:
    k = count_present(it)
    it["k_present"] = k
    it["meta_coverage"] = round(k/6.0, 3)
    it["meta_used"] = k > 0
p.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
print("Listo ✓")