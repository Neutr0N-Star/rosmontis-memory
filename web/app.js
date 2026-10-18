/* 藏识 Alaya —— 对话前端 */

const $ = (s) => document.querySelector(s);

/* 一个角色 = 一个 session：人格和记忆都按 session 隔离，
 * 所以换角色不需要额外的数据结构，切 session 就够了。 */
const CARDS_KEY = 'alaya.cards';
const CUR_KEY = 'alaya.current';

function loadCards() {
  try {
    const raw = JSON.parse(localStorage.getItem(CARDS_KEY) || '[]');
    return Array.isArray(raw) && raw.length ? raw : [];
  } catch (e) {
    return [];
  }
}
function saveCards(list) {
  try {
    localStorage.setItem(CARDS_KEY, JSON.stringify(list));
  } catch (e) {}
}

function ensureDefaultCard() {
  let list = loadCards();
  if (!list.length) {
    let sid = localStorage.getItem('alaya.session');
    if (!sid) {
      sid = 's' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
      localStorage.setItem('alaya.session', sid);
    }
    list = [{ session: sid, name: '藏识', desc: '默认角色' }];
    saveCards(list);
    localStorage.setItem(CUR_KEY, sid);
  }
  return list;
}

let CARDS = ensureDefaultCard();
let SESSION = localStorage.getItem(CUR_KEY) || CARDS[0].session;

const state = { busy: false, closed: false, count: 0 };

const esc = (s) =>
  String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

async function api(path, opt) {
  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 60000);
  try {
    const r = await fetch(path, { ...(opt || {}), signal: ctrl.signal });
    const j = await r.json().catch(() => null);
    if (!j) throw new Error(`服务返回了看不懂的东西（${r.status}）`);
    if (!j.ok) {
      const err = new Error((j.error && j.error.message) || '请求失败');
      err.code = j.error && j.error.code;
      err.status = r.status;
      throw err;
    }
    return j.data;
  } finally {
    clearTimeout(timer);
  }
}

function toast(msg) {
  const el = $('#toast');
  el.textContent = msg;
  el.classList.add('on');
  clearTimeout(el._t);
  el._t = setTimeout(() => el.classList.remove('on'), 2600);
}

/* ── 消息 ────────────────────────────────────────────── */
function addMsg(role, text, meta) {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  if (role === 'sys') {
    el.textContent = text;
  } else {
    if (role === 'ai') {
      const who = document.createElement('div');
      who.className = 'who';
      who.textContent = '她';
      el.appendChild(who);
    }
    const body = document.createElement('div');
    body.className = 'body';
    body.textContent = text;
    el.appendChild(body);
    if (meta) {
      const c = document.createElement('div');
      c.className = 'cost';
      c.textContent = meta;
      el.appendChild(c);
    }
  }
  $('#chat').appendChild(el);
  $('#chat').scrollTop = $('#chat').scrollHeight;
  return el;
}

/** 等待气泡：三点跳动。静态「…」在 3 秒里看起来像卡死。 */
function addTyping() {
  const el = document.createElement('div');
  el.className = 'msg ai';
  el.innerHTML = `<div class="who">她</div>
    <div class="body typing"><i></i><i></i><i></i></div>
    <div class="cost elapsed">0.0s</div>`;
  $('#chat').appendChild(el);
  $('#chat').scrollTop = $('#chat').scrollHeight;

  const t0 = performance.now();
  el._timer = setInterval(() => {
    const d = el.querySelector('.elapsed');
    if (d) d.textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 100);
  return el;
}

/** 错误条：不要伪装成她的回复 */
function addError(text, retry) {
  const el = document.createElement('div');
  el.className = 'err-bar';
  el.innerHTML = `<span>${esc(text)}</span>`;
  if (retry) {
    const b = document.createElement('button');
    b.className = 'retry';
    b.textContent = '再说一次';
    b.addEventListener('click', () => {
      el.remove();
      retry();
    });
    el.appendChild(b);
  }
  $('#chat').appendChild(el);
  $('#chat').scrollTop = $('#chat').scrollHeight;
  return el;
}

function clearEmpty() {
  const e = $('#empty');
  if (e) e.remove();
}

function showEmpty() {
  const el = document.createElement('div');
  el.id = 'empty';
  el.className = 'empty-state';
  el.innerHTML = `
    <div class="dot"></div>
    <p class="lead">她记得你们说过的话</p>
    <p class="sub">聊两句，试试她能不能记住。</p>
    <div class="chips">
      <button data-say="我今天有点累">我今天有点累</button>
      <button data-say="我在准备考试，压力挺大">我在准备考试</button>
      <button data-say="随便聊聊吧">随便聊聊</button>
    </div>`;
  $('#chat').appendChild(el);
  el.querySelectorAll('[data-say]').forEach((b) =>
    b.addEventListener('click', () => {
      $('#input').value = b.dataset.say;
      autoGrow($('#input'));
      send();
    })
  );
}

/* ── 发送 ────────────────────────────────────────────── */
async function send(textOverride) {
  if (state.busy) return;
  if (state.closed) {
    toast('已经打烊了，点「开门」再聊');
    return;
  }
  const input = $('#input');
  const text = (textOverride != null ? textOverride : input.value).trim();
  if (!text) {
    input.focus();
    return;
  }

  state.busy = true;
  $('#send').disabled = true;
  if (textOverride == null) {
    input.value = '';
    autoGrow(input);
  }
  clearEmpty();

  addMsg('me', text);
  const ph = addTyping();

  try {
    const t0 = performance.now();
    const r = await api('/v1/turn', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, session_id: SESSION }),
    });
    const ms = Math.round(performance.now() - t0);

    clearInterval(ph._timer);
    ph.remove();

    const mu = r.memory_used || {};
    const bits = [];
    if (mu.memory_tokens) bits.push(`${mu.memory_tokens} token`);
    if (mu.memory_items) bits.push(`${mu.memory_items} 条记忆`);
    bits.push(`${(ms / 1000).toFixed(1)}s`);
    const el = addMsg('ai', r.reply || '……');
    const c = document.createElement('div');
    c.className = 'cost';
    c.textContent = bits.join(' · ');
    el.appendChild(c);

    state.count++;
    updateMeta();
  } catch (e) {
    clearInterval(ph._timer);
    ph.remove();
    const msg =
      e.name === 'AbortError'
        ? '等太久了，可能是模型卡住'
        : e.code === 'llm_unavailable'
        ? '她这边没接上话（模型没响应）'
        : e.message.includes('Failed to fetch')
        ? '连不上服务，看看后台还开着吗'
        : e.message;
    addError(msg, () => send(text));
  }

  state.busy = false;
  $('#send').disabled = false;
  input.focus();
}

/* ── 记忆抽屉 ────────────────────────────────────────── */
function openDrawer() {
  $('#drawer').hidden = false;
  $('#mask').classList.add('on');
  loadMemory();
}

function closeDrawer() {
  $('#drawer').hidden = true;
  $('#mask').classList.remove('on');
}

async function loadMemory() {
  const body = $('#drawerBody');
  body.innerHTML = '<div class="skeleton"><span></span><span></span><span></span></div>';
  try {
    const [eps, facts, ents, st, notes] = await Promise.all([
      api(`/v1/episodes?session_id=${SESSION}&limit=30`),
      api(`/v1/facts?session_id=${SESSION}&limit=40`),
      api(`/v1/entities?session_id=${SESSION}&limit=20`),
      api(`/v1/state/${encodeURIComponent('我')}`).catch(() => ({ slots: [] })),
      api(`/v1/notebook?session_id=${SESSION}&limit=20`).catch(() => []),
    ]);

    const total = (eps.items || []).length + (facts || []).length + (ents || []).length;
    if (!total && !(st.slots || []).length) {
      body.innerHTML = `<div class="empty-state small">
        <div class="dot"></div>
        <p class="lead">还什么都没记住</p>
        <p class="sub">聊两句，她就开始记了。</p></div>`;
      return;
    }

    const sec = (title, items, render, deletable) => {
      if (!items || !items.length) return '';
      return `<section class="sec"><h3>${title} <span class="cnt">${items.length}</span></h3>
        <ul>${items
          .map(
            (i) =>
              `<li>${render(i)}${
                deletable && i.id
                  ? `<button class="del" data-del="${esc(i.id)}" title="删掉这条">×</button>`
                  : ''
              }</li>`
          )
          .join('')}</ul></section>`;
    };

    const cn = { fact: '事实', preference: '偏好', emotion: '情绪', lesson: '教训', principle: '原则', tool_behavior: '工具' };
    const cnTier = { resident: '常驻', retrieval: '笔记', inbox: '速记', important: '重要' };

    body.innerHTML =
      sec('当前状态', st.slots, (s) => `<b>${esc(s.slot)}</b>：${esc(s.value)}<div class="row"><span>自 ${fmt(s.since)}</span></div>`) +
      sec('关于你', ents, (e) => `<b>${esc(e.name)}</b><div class="row"><span>${e.fact_count} 条事实</span><span>提及 ${e.mentions} 次</span></div>`) +
      sec('事实', facts, (f) => `${esc(f.subject)}${f.polarity < 0 ? '不' : ''}${esc(f.predicate)} ${esc(f.object)}<div class="row"><span>${cn[f.memory_type] || esc(f.memory_type || '')}</span><span>${fmt(f.valid_from)}</span></div>`) +
      sec('笔记', notes, (n) => `${esc(n.text || n.what || '')}<div class="row"><span>${cnTier[n.tier] || esc(n.tier)}</span>${n.next ? `<span>下次：${esc(n.next)}</span>` : ''}</div>`) +
      sec('说过的话', eps.items, (e) => `${esc(e.text)}<div class="row"><span>${cnTier[e.tier] || esc(e.tier)}</span><span>${fmt(e.t_event)}</span></div>`, true);

    // 删除：删了就真不回来了，二次确认
    body.querySelectorAll('[data-del]').forEach((b) => {
      b.addEventListener('click', async (ev) => {
        ev.stopPropagation();
        if (!confirm('删掉这条？删了就找不回来了。')) return;
        try {
          await api(`/v1/remove?item_id=${encodeURIComponent(b.dataset.del)}&reason=user`, { method: 'POST' });
          b.closest('li').remove();
          toast('删了');
        } catch (e) {
          toast(`没删成：${e.message}`);
        }
      });
    });
  } catch (e) {
    body.innerHTML = `<div class="empty-state small"><p class="lead">读不出来</p>
      <p class="sub">${esc(e.message)}</p></div>`;
  }
}

function fmt(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

/* ── 打烊 ────────────────────────────────────────────── */
async function sleep() {
  if (state.busy) return;
  if (state.closed) {
    state.closed = false;
    document.body.classList.remove('closed');
    $('#btnSleep').textContent = '打烊';
    addMsg('sys', '—— 开门 ——');
    return;
  }
  if (state.count === 0) {
    toast('还没聊呢，先聊两句');
    return;
  }

  state.busy = true;
  $('#btnSleep').disabled = true;
  try {
    const r = await api(`/v1/session/sleep?session_id=${SESSION}`, { method: 'POST' });
    const cells = [];
    if (r.close && r.close.touched) cells.push(['强化', r.close.touched]);
    if (r.pages) cells.push(['沉淀', r.pages]);
    if (r.dream) cells.push(['做梦', r.dream.replayed]);
    if (!cells.length) cells.push(['今天', '平静']);

    const el = document.createElement('div');
    el.className = 'settle';
    el.innerHTML = `<div class="settle-title">打烊了</div>
      <div class="settle-cells">${cells
        .map(([k, v]) => `<div><b>${v}</b><span>${k}</span></div>`)
        .join('')}</div>
      <p class="settle-note">${r.dream ? '她把今天过了一遍。' : r.dream_skipped ? '今天没什么可想的。' : ''}</p>`;
    $('#chat').appendChild(el);
    $('#chat').scrollTop = $('#chat').scrollHeight;

    state.closed = true;
    document.body.classList.add('closed');
    $('#btnSleep').textContent = '开门';
    $('#send').disabled = true;
  } catch (e) {
    addError(`打烊失败：${e.message}`, sleep);
  }
  state.busy = false;
  $('#btnSleep').disabled = false;
}

/* ── 状态栏 ──────────────────────────────────────────── */
function updateMeta() {
  $('#meta').textContent = state.count ? `已聊 ${state.count} 轮` : '';
}

/* ── 角色卡 ──────────────────────────────────────────── */
function renderCardPop() {
  const cur = CARDS.find((c) => c.session === SESSION);
  $('#cardName').textContent = (cur && cur.name) || '藏识';
  $('#cardList').innerHTML = CARDS.map(
    (c) => `<button class="pop-item ${c.session === SESSION ? 'on' : ''}" data-card="${esc(c.session)}">
      <b>${esc(c.name)}</b>${c.desc ? `<span>${esc(c.desc)}</span>` : ''}</button>`
  ).join('');
  $('#cardList')
    .querySelectorAll('[data-card]')
    .forEach((b) => b.addEventListener('click', () => switchCard(b.dataset.card)));
}

async function switchCard(sessionId) {
  if (sessionId === SESSION) {
    $('#cardPop').hidden = true;
    return;
  }
  SESSION = sessionId;
  localStorage.setItem(CUR_KEY, sessionId);
  $('#cardPop').hidden = true;
  renderCardPop();

  // 清空并重画这个角色的对话
  state.count = 0;
  state.closed = false;
  document.body.classList.remove('closed');
  $('#btnSleep').textContent = '打烊';
  $('#send').disabled = false;
  $('#chat').innerHTML = '';
  closeDrawer();
  await restoreHistory();

  try {
    await api(`/v1/session/wake?session_id=${SESSION}`, { method: 'POST' });
  } catch (e) {}
  $('#input').focus();
}

async function createCard() {
  const name = ($('#ncName').value || '').trim();
  if (!name) {
    toast('总得有个名字');
    return;
  }
  const desc = ($('#ncDesc').value || '').trim();
  const seed = ($('#ncSeed').value || '').trim();

  const session = 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
  try {
    // 种子写进人格：这是这个角色的记忆起点
    await api('/v1/persona', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: seed || `你是${name}。你记得来过的人，但不多问。`,
        session_id: session,
        section: 'identity',
        author: 'self',
        protected: true,
        priority: 100,
      }),
    });
  } catch (e) {
    toast(`种子没写进去：${e.message}`);
  }

  CARDS.push({ session, name, desc });
  saveCards(CARDS);
  $('#newModal').hidden = true;
  $('#ncName').value = '';
  $('#ncDesc').value = '';
  $('#ncSeed').value = '';
  await switchCard(session);
  toast(`建好了：${name}`);
}

/* ── 启动 ────────────────────────────────────────────── */
function autoGrow(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

async function boot() {
  $('#send').addEventListener('click', () => send());
  $('#input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  $('#input').addEventListener('input', (e) => autoGrow(e.target));
  $('#btnMemory').addEventListener('click', openDrawer);
  $('#closeDrawer').addEventListener('click', closeDrawer);
  $('#mask').addEventListener('click', closeDrawer);
  $('#btnSleep').addEventListener('click', sleep);
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeDrawer();
  });

  try {
    const w = await api(`/v1/session/wake?session_id=${SESSION}`, { method: 'POST' });
    if (w.persona_chars > 30) {
      // 有人格内容才算"她醒着"
      $('#meta').textContent = '已就绪';
    }
  } catch (e) {
    addError('连不上后台服务，先确认它开着', () => location.reload());
    return;
  }

  renderCardPop();

  // 事件：先绑再发请求，慢启动时不能整页是死的
  $('#brand').addEventListener('click', () => {
    $('#cardPop').hidden = !$('#cardPop').hidden;
  });
  $('#newCard').addEventListener('click', () => {
    $('#cardPop').hidden = true;
    $('#newModal').hidden = false;
    $('#ncName').focus();
  });
  $('#ncCancel').addEventListener('click', () => { $('#newModal').hidden = true; });
  $('#ncSave').addEventListener('click', createCard);
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#cardPop') && !e.target.closest('#brand')) $('#cardPop').hidden = true;
  });

  await restoreHistory();
}

async function restoreHistory() {
  try {
    const eps = await api(`/v1/episodes?session_id=${SESSION}&limit=50`);
    const dialog = (eps.items || [])
      .filter((e) => e.role === 'user' || e.role === 'assistant')
      .slice()
      .reverse(); // 接口按时间倒序给，渲染要正序

    if (!dialog.length) {
      showEmpty();
    } else {
      const first = new Date(dialog[0].t_event * 1000);
      addMsg('sys', `—— ${first.getMonth() + 1}月${first.getDate()}日 ——`);
      let lastDay = first.getDate();
      for (const e of dialog) {
        const d = new Date(e.t_event * 1000);
        if (d.getDate() !== lastDay) {
          lastDay = d.getDate();
          addMsg('sys', `—— ${d.getMonth() + 1}月${d.getDate()}日 ——`);
        }
        addMsg(e.role === 'user' ? 'me' : 'ai', e.text);
      }
      state.count = Math.ceil(dialog.length / 2);
      updateMeta();
      $('#chat').scrollTop = $('#chat').scrollHeight;
    }
  } catch (e) {
    showEmpty();
  }
  $('#input').focus();
}

boot();
