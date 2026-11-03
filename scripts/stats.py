"""Collect every system's per-trial predictions on the 144 held-out trials, then report
accuracy, 95% bootstrap interval, LEFT-answer rate, and McNemar / bootstrap-gain tests.
Writes repro/predictions.csv and repro/results.json."""
import json, os
import numpy as np
from scipy.stats import binomtest

W = __file__.rsplit("/repro/", 1)[0]
R = f"{W}/repro"
ids = [json.loads(l)["id"] for l in open(f"{R}/test.jsonl")]
y = np.array([json.loads(l)["messages"][2]["content"] for l in open(f"{R}/test.jsonl")])
c = json.load(open(f"{R}/classic_results.json"))
assert c["ids"] == ids
sys = {"csp_lda": np.where(np.array(c["csp_lda"]["pred"]) == 1, "RIGHT", "LEFT")}
for name in ("base", "ft", "harness_prompt", "harness_tool"):
    p = f"{R}/llm_{name}.json"
    if os.path.exists(p):
        rows = {r["id"]: r["guess"] for r in json.load(open(p))["rows"]}
        sys[{"base": "glm53_base", "ft": "glm53_lora", "harness_prompt": "glm53_score_in_prompt", "harness_tool": "glm53_tool_harness"}[name]] = np.array([rows[i] for i in ids])

rng = np.random.default_rng(0)
B = rng.integers(0, len(y), (10000, len(y)))
out = {}
for k, p in sys.items():
    ok = p == y
    bs = ok[B].mean(1)
    out[k] = dict(acc=ok.mean(), correct=int(ok.sum()), n=len(y), ci95=np.percentile(bs, [2.5, 97.5]).tolist(),
                  left_rate=float((p == "LEFT").mean()), p_vs_chance=binomtest(int(ok.sum()), len(y), 0.5, alternative="greater").pvalue)


def compare(a, b):
    oa, ob = sys[a] == y, sys[b] == y
    n01, n10 = int((~oa & ob).sum()), int((oa & ~ob).sum())  # b right & a wrong, a right & b wrong
    p = binomtest(n01, n01 + n10, 0.5).pvalue if n01 + n10 else 1.0
    gain = (ob[B].mean(1) - oa[B].mean(1)) * 100
    return dict(b_wins=n01, a_wins=n10, mcnemar_exact_p=p, gain_pts=float((ob.mean() - oa.mean()) * 100),
                gain_ci95=np.percentile(gain, [2.5, 97.5]).tolist())


pairs = [("glm53_base", "csp_lda"), ("glm53_base", "glm53_lora"), ("csp_lda", "glm53_lora"), ("glm53_base", "glm53_tool_harness"), ("glm53_base", "glm53_score_in_prompt"), ("csp_lda", "glm53_tool_harness")]
out["compare"] = {f"{a}->{b}": compare(a, b) for a, b in pairs if a in sys and b in sys}
json.dump(out, open(f"{R}/results.json", "w"), indent=1, default=float)
with open(f"{R}/predictions.csv", "w") as f:
    f.write("id,truth," + ",".join(sys) + "\n")
    for i, t in enumerate(ids):
        f.write(f"{t},{y[i]}," + ",".join(sys[k][i] for k in sys) + "\n")
for k, v in out.items():
    if k != "compare":
        print(f"{k:12s} {v['acc']:.3f} ({v['correct']}/{v['n']})  CI {v['ci95'][0]:.3f}-{v['ci95'][1]:.3f}  LEFT {v['left_rate']:.0%}  p>chance {v['p_vs_chance']:.3g}")
for k, v in out["compare"].items():
    print(f"{k:26s} gain {v['gain_pts']:+.1f} pts  CI [{v['gain_ci95'][0]:+.1f},{v['gain_ci95'][1]:+.1f}]  wins {v['b_wins']} vs {v['a_wins']}  McNemar p {v['mcnemar_exact_p']:.3g}")
