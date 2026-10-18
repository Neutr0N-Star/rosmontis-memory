import json, urllib.request, concurrent.futures

URL = "http://127.0.0.1:11434/api/chat"
MODEL = "qwen38-4b-long:latest"
BASE = "你是小深，和用户长期相处的伙伴。像朋友一样自然地聊天，不必刻意安慰。"

def chat(system, user):
    body = {"model": MODEL,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
            "stream": False, "think": False,
            "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 300}}
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=600) as r:
        d = json.loads(r.read().decode())
    m = d["message"]
    return (m.get("content") or m.get("thinking") or "").strip()

Q1 = "今天有点累，早点睡吧"
Q2 = "今天下班路上看到一家新开的面馆，人挺多的"
Q3 = "帮我想想周末去哪儿散步比较好"
B = BASE + "\n\n"

CASES = []
for i in range(5):
    CASES += [("E1","A 人设层", B+"【关于他】他怕黑。晚上不要关灯。", Q1),
              ("E1","B 检索层", B+"【想起来的片段】- 他之前说过他怕黑", Q1),
              ("E1","C 无记忆", BASE, Q1)]
for i in range(5):
    CASES += [("E2","A 人设层", B+"【关于他】他母亲上个月去世了。", Q2),
              ("E2","B 检索层", B+"【想起来的片段】- 他之前说过他母亲上个月去世了", Q2)]
for i in range(5):
    CASES += [("E3","标签=关于他/措辞=前提", B+"【关于他】他怕黑。晚上不要关灯。", Q1),
              ("E3","标签=关于他/措辞=事件", B+"【关于他】他之前说过他怕黑", Q1),
              ("E3","标签=想起来的/措辞=前提", B+"【想起来的片段】- 他怕黑。晚上不要关灯。", Q1),
              ("E3","标签=想起来的/措辞=事件", B+"【想起来的片段】- 他之前说过他怕黑", Q1)]
for i in range(5):
    CASES += [("E4","常驻+无关问题", B+"【常驻·珍视】他母亲上个月去世了。这是他最不愿意被反复提起的事。", Q3)]

def run(c):
    exp, cond, s, q = c
    try: out = chat(s, q)
    except Exception as e: out = "ERROR: " + str(e)
    return {"exp": exp, "cond": cond, "system": s, "user": q, "out": out}

with concurrent.futures.ThreadPoolExecutor(max_workers=3) as ex:
    res = list(ex.map(run, CASES))
json.dump({"model": MODEL, "runs": res}, open("d:/MyHub/06-Projects/alaya-memory/tests/_ab_position_raw2.json","w",encoding="utf-8"), ensure_ascii=False, indent=1)
with open("d:/MyHub/06-Projects/alaya-memory/tests/_ab_position_raw2.md","w",encoding="utf-8") as f:
    for exp in ["E1","E2","E3","E4"]:
        f.write(f"\n# {exp}\n")
        for r in [x for x in res if x["exp"]==exp]:
            f.write(f"\n## {r['cond']}\nUSER: {r['user']}\nOUT: {r['out']}\n")
print("done", len(res))
