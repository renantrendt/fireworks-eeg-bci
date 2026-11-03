"""Harness: the LLM does not decode EEG. It calls the classic decoder (CSP+LDA) as a tool,
reads the score, and gives the final command. Scored on the same 144 held-out trials.

Usage: eval_harness.py [model id]   (default: base GLM 5.3 Flash)
Writes repro/llm_harness_tool.json in the same shape as llm_base.json / llm_ft.json.
"""
import json, os, sys, time, concurrent.futures as cf
import requests

W = __file__.rsplit("/repro/", 1)[0]
URL = "https://api.fireworks.ai/inference/v1/chat/completions"
model = sys.argv[1] if len(sys.argv) > 1 else "accounts/fireworks/models/glm-5p3-flash"
key = os.environ["FIREWORKS_API_KEY"]
rule = json.load(open(f"{W}/repro/lda_rule.json"))
w, b = rule["weights"], rule["bias"]
rows = [json.loads(l) for l in open(f"{W}/repro/test.jsonl")]

SYSTEM = ("You are the controller of an assistive device for a person with paralysis. "
          "A classic EEG decoder (CSP+LDA) has already read the person's brain signal and returns a score. "
          "Positive score means the person imagined the RIGHT hand, negative means LEFT. "
          "Scores near zero are uncertain. Your job: output the command for the device. "
          "Answer with one word: LEFT or RIGHT.")

TOOL = {"type": "function", "function": {
    "name": "eeg_decoder", "description": "Run the CSP+LDA decoder on the current 4 s EEG window.",
    "parameters": {"type": "object", "properties": {}}}}


def decoder_score(user_text):
    comps = [float(l.split(":")[1]) for l in user_text.strip().splitlines()]
    return sum(wi * ci for wi, ci in zip(w, comps)) + b


def ask(r):
    s = decoder_score(r["messages"][1]["content"])
    tool_result = json.dumps({"score": round(s, 2), "sign_rule": "positive=RIGHT, negative=LEFT",
                              "confidence": "low" if abs(s) < 0.3 else "ok"})
    msgs = [{"role": "system", "content": SYSTEM},
            {"role": "user", "content": "New EEG window ready. Decide the command."},
            {"role": "assistant", "content": None, "tool_calls": [{"id": "call_1", "type": "function",
                                                                   "function": {"name": "eeg_decoder", "arguments": "{}"}}]},
            {"role": "tool", "tool_call_id": "call_1", "content": tool_result}]
    body = {"model": model, "messages": msgs, "tools": [TOOL], "max_tokens": 2000, "temperature": 0,
            "reasoning_effort": "low"}
    for attempt in range(6):
        resp = requests.post(URL, json=body, headers={"Authorization": f"Bearer {key}"}, timeout=90)
        if resp.status_code == 429:
            time.sleep(5 * (attempt + 1)); continue
        if resp.status_code == 200:
            j = resp.json()
            out = (j["choices"][0]["message"].get("content") or "").strip().upper()
            words = [x.strip(".*:\n ") for x in out.split()]
            guess = next((x for x in reversed(words) if x in ("LEFT", "RIGHT")), "NONE")
            return dict(id=r["id"], truth=r["messages"][2]["content"], guess=guess, raw=out,
                        score=s, usage=j.get("usage", {}))
        time.sleep(2 ** attempt)
    raise RuntimeError(resp.text[:300])


with cf.ThreadPoolExecutor(2) as ex:
    res = list(ex.map(ask, rows))
acc = sum(r["truth"] == r["guess"] for r in res) / len(res)
lda_acc = sum(("RIGHT" if r["score"] > 0 else "LEFT") == r["truth"] for r in res) / len(res)
agree = sum(("RIGHT" if r["score"] > 0 else "LEFT") == r["guess"] for r in res) / len(res)
json.dump(dict(model=model, acc=acc, lda_acc=lda_acc, agree_with_lda=agree, rows=res),
          open(f"{W}/repro/llm_harness_tool.json", "w"), indent=1)
print(f"harness: {acc:.3f} ({sum(r['truth'] == r['guess'] for r in res)}/{len(res)}), "
      f"LDA alone {lda_acc:.3f}, LLM agreed with decoder {agree:.1%}, "
      f"no-answer {sum(r['guess'] == 'NONE' for r in res)}")
