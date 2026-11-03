"""Score a Fireworks model on the 144 held-out trials. Usage: eval_llm.py <model id> <out name>

Reads FIREWORKS_API_KEY from the environment. The system prompt is the same on every call and
comes first, so Fireworks' automatic prefix caching applies to it.
"""
import json, os, sys, time, concurrent.futures as cf
import requests

W = __file__.rsplit("/repro/", 1)[0]
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
model, name = sys.argv[1], sys.argv[2]
key = os.environ["FIREWORKS_API_KEY"]
rows = [json.loads(l) for l in open(f"{W}/repro/" + os.environ.get("TESTFILE", "test.jsonl"))]


def ask(r):
    body = {"model": model, "messages": r["messages"][:2], "max_tokens": 3000, "temperature": 0, "reasoning_effort": "low"}
    for attempt in range(5):
        resp = requests.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=60)
        if resp.status_code == 429:
            time.sleep(5 * (attempt + 1))
            continue
        if resp.status_code == 200:
            j = resp.json()
            out = (j["choices"][0]["message"].get("content") or "").strip().upper()
            words = [w.strip(".*:\n ") for w in out.split()]
            last = next((w for w in reversed(words) if w in ("LEFT", "RIGHT")), "NONE")
            guess = last
            return dict(id=r["id"], truth=r["messages"][2]["content"], guess=guess, raw=out, usage=j.get("usage", {}))
        time.sleep(2 ** attempt)
    raise RuntimeError(resp.text[:300])


with cf.ThreadPoolExecutor(2) as ex:
    res = list(ex.map(ask, rows))
acc = sum(r["truth"] == r["guess"] for r in res) / len(res)
json.dump(dict(model=model, acc=acc, rows=res), open(f"{W}/repro/llm_{name}.json", "w"), indent=1)
print(f"{name}: {acc:.3f} ({sum(r['truth'] == r['guess'] for r in res)}/{len(res)}), "
      f"no-answer {sum(r['guess'] == 'NONE' for r in res)}, "
      f"cached prompt tokens {sum(r['usage'].get('prompt_tokens_details', {}).get('cached_tokens', 0) or 0 for r in res)}")
