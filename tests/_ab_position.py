import json, urllib.request, concurrent.futures, sys

URL = "http://127.0.0.1:11434/api/chat"
MODEL = "minicpm5-2b:q8-64k"
BASE = "你是小深，用户的 AI 伙伴。用中文口语回答，2-3 句，不要长篇。"

def chat(system, user, tag):
    body = {
        "model": MODEL,
        "messages": ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}],
        "stream": False,
        "think": False,
        "options": {"temperature": 0.7, "top_p": 0.9, "num_predict": 160},
    }
    req = urllib.request.Request(URL, data=json.dumps(body).encode(), headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        d = json.loads(r.read().decode())
    return {"tag": tag, "system": system, "user": user, "out": d["message"]["content"].strip()}

Q1 = "今天有点累，早点睡吧"
Q2 = "今天下班路上看到一家新开的面馆，人挺多的"
Q3 = "帮我想想周末去哪儿散步比较好"

CASES = []
# 实验一：怕黑
sA = BASE + "\n\n【关于他】他怕黑。晚上不要关灯。"
sB = BASE + "\n\n【想起来的片段】- 他之前说过他怕黑"
sC = BASE
for i in range(5):
    CASES += [(("E1","A"), sA, Q1), (("E1","B"), sB, Q1), (("E1","C"), sC, Q1)]
# 实验二：母亲去世
s2A = BASE + "\n\n【关于他】他母亲上个月去世了。"
s2B = BASE + "\n\n【想起来的片段】- 他之前说过他母亲上个月去世了"
for i in range(5):
    CASES += [(("E2","A"), s2A, Q2), (("E2","B"), s2B, Q2)]
# 实验三：2x2 标签 x 措辞（怕黑）
LAB_P = "【关于他】"; LAB_R = "【想起来的片段】- "
TXT_P = "他怕黑。晚上不要关灯。"; TXT_E = "他之前说过他怕黑"
for i in range(5):
    CASES += [
        (("E3","标签=关于他 / 措辞=前提"), BASE + "\n\n" + LAB_P + TXT_P, Q1),
        (("E3","标签=关于他 / 措辞=事件"), BASE + "\n\n" + LAB_P + TXT_E, Q1),
        (("E3","标签=想起来的 / 措辞=前提"), BASE + "\n\n" + LAB_R + TXT_P, Q1),
        (("E3","标签=想起来的 / 措辞=事件"), BASE + "\n\n" + LAB_R + TXT_E, Q1),
    ]
# 额外验证1：常驻情绪信息 + 无关问题
sP = BASE + "\n\n【常驻·珍视】他母亲上个月去世了。这是他最不愿意被反复提起的事。"
for i in range(5):
    CASES += [(("E4","常驻+无关问题"), sP, Q3)]

def run(c):
    (exp, cond), s, q = c
    try:
        r = chat(s, q, cond)
    except Exception as e:
        r = {"tag": cond, "system": s, "user": q, "out": "ERROR: " + str(e)}
    r["exp"] = exp; r["cond"] = cond
    return r

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as ex:
    res = list(ex.map(run, CASES))

json.dump(res, open("d:/MyHub/06-Projects/alaya-memory/tests/_ab_position_raw.json", "w", encoding="utf-8"), ensure_ascii=False, indent=1)

with open("d:/MyHub/06-Projects/alaya-memory/tests/_ab_position_raw.md", "w", encoding="utf-8") as f:
    for exp in ["E1","E2","E3","E4"]:
        f.write(f"\n\n# {exp}\n")
        for r in [x for x in res if x["exp"] == exp]:
            f.write(f"\n## {r['cond']}\nSYS: {r['system'].replace(chr(10),' / ')}\nUSER: {r['user']}\nOUT: {r['out']}\n")
print("done", len(res))
