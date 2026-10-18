/* 藏识 Alaya —— 对话前端
 *
 * 单一状态源：所有可变状态都在 store 里，改完必须 commit()。
 * 事件绑定与网络请求分开：bindAll() 无条件先跑，网络请求放后面。
 */

const $ = (s) => document.querySelector(s);
const $$ = (s) => [...document.querySelectorAll(s)];
const esc = (s) => String(s == null ? '' : s).replace(/[&<>"]/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c]));

const K_CARDS = 'alaya.cards';
const K_CUR = 'alaya.current';
const K_BG = 'alaya.bg';

/* ── 状态：唯一真相 ─────────────────────────────────── */
const store = {
  cards: [],
  session: '',
  card: null,
  editing: null,
  pendingAvatar: '',
  mode: 'chat', // chat | wizard
  busy: false,
  closed: false,
  count: 0,
  debug: false, // 是否显示 token 等计量信息

  load() {
    try {
      this.cards = JSON.parse(localStorage.getItem(K_CARDS) || '[]') || [];
    } catch (e) {
      this.cards = [];
    }
    if (!this.cards.length) {
      this.cards = [{ session: 's' + Date.now().toString(36), name: '藏识', desc: '' }];
    }
    this.session = localStorage.getItem(K_CUR) || this.cards[0].session;
    this.syncCard();
    return this;
  },

  syncCard() {
    this.card = this.cards.find((c) => c.session === this.session) || this.cards[0] || { name: '藏识' };
  },

  commit() {
    try {
      localStorage.setItem(K_CARDS, JSON.stringify(this.cards));
      if (this.session) localStorage.setItem(K_CUR, this.session);
    } catch (e) {
      toast('存不下了（本地配额满），换张小点的图');
      return false;
    }
    this.syncCard();
    return true;
  },

  addCard(data) {
    const session = 'c' + Date.now().toString(36) + Math.random().toString(36).slice(2, 6);
    const card = { session, avatar: '', ...data };
    this.cards.push(card);
    this.commit();
    return card;
  },

  setSession(sid) {
    this.session = sid;
    this.commit();
    this.syncCard();
  },

  remove(sid) {
    this.cards = this.cards.filter((c) => c.session !== sid);
    if (!this.cards.length) {
      this.cards = [{ session: 's' + Date.now().toString(36), name: '藏识', desc: '' }];
    }
    if (this.session === sid) this.session = this.cards[0].session;
    this.commit();
  },
};

/* ── 网络 ───────────────────────────────────────────── */
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

/* ── 弹层栈：Esc 与遮罩统一处理 ─────────────────────── */
const layers = ['drawer', 'cardPop', 'newModal', 'setModal', 'wizModal'];
function openLayer(id) {
  $('#' + id).hidden = false;
  if (id === 'cardPop') $('#cardPop').classList.add('on');
}
function closeLayer(id) {
  const el = $('#' + id);
  if (el) el.hidden = true;
  if (id === 'cardPop') el.classList.remove('on');
}
function closeAllLayers() {
  layers.forEach(closeLayer);
  $('#mask').classList.remove('on');
}
/** 关最上层：从后往前找第一个开着的 */
function closeTopLayer() {
  for (let i = layers.length - 1; i >= 0; i--) {
    const el = $('#' + layers[i]);
    if (el && !el.hidden) {
      closeLayer(layers[i]);
      if (layers[i] === 'drawer') $('#mask').classList.remove('on');
      return true;
    }
  }
  return false;
}

/* ── 渲染：消息 ─────────────────────────────────────── */
let lastSpeaker = null;

function avatarEl(cls) {
  const el = document.createElement('div');
  el.className = cls || 'avatar';
  const c = store.card;
  if (c && c.avatar) {
    el.style.backgroundImage = `url(${c.avatar})`;
    el.style.backgroundSize = 'cover';
    el.style.backgroundPosition = 'center';
  } else {
    el.textContent = (c && c.name ? c.name : '她').charAt(0);
  }
  return el;
}

/** 署名只在换人时出现——每条都署名是聊天记录导出，不是对话 */
function addMsg(role, text, meta) {
  const el = document.createElement('div');
  el.className = `msg ${role}`;

  if (role === 'sys') {
    el.textContent = text;
  } else if (role === 'me') {
    const b = document.createElement('div');
    b.className = 'body';
    b.textContent = text;
    el.appendChild(b);
    lastSpeaker = 'me';
  } else {
    const row = document.createElement('div');
    row.className = 'ai-row';
    const col = document.createElement('div');
    col.className = 'ai-col';

    if (lastSpeaker !== 'ai') {
      const who = document.createElement('div');
      who.className = 'who';
      who.textContent = store.card ? store.card.name : '她';
      col.appendChild(who);
    }
    const b = document.createElement('div');
    b.className = 'body';
    b.textContent = text;
    col.appendChild(b);
    if (meta && store.debug) {
      const c = document.createElement('div');
      c.className = 'cost';
      c.textContent = meta;
      col.appendChild(c);
    }
    row.appendChild(avatarEl());
    row.appendChild(col);
    el.appendChild(row);
    lastSpeaker = 'ai';
  }

  $('#chat').appendChild(el);
  $('#chat').scrollTop = $('#chat').scrollHeight;
  return el;
}

function addTyping() {
  const el = document.createElement('div');
  el.className = 'msg ai';
  const row = document.createElement('div');
  row.className = 'ai-row';
  const col = document.createElement('div');
  col.className = 'ai-col';
  if (lastSpeaker !== 'ai') {
    const who = document.createElement('div');
    who.className = 'who';
    who.textContent = store.card ? store.card.name : '她';
    col.appendChild(who);
  }
  const t = document.createElement('div');
  t.className = 'body typing';
  t.innerHTML = '<i></i><i></i><i></i>';
  const e = document.createElement('div');
  e.className = 'cost elapsed';
  e.textContent = '0.0s';
  col.appendChild(t);
  if (store.debug) col.appendChild(e);
  row.appendChild(avatarEl());
  row.appendChild(col);
  el.appendChild(row);
  $('#chat').appendChild(el);
  $('#chat').scrollTop = $('#chat').scrollHeight;

  const t0 = performance.now();
  el._timer = setInterval(() => {
    if (e) e.textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 100);
  return el;
}

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
}

/* ── 空态 ───────────────────────────────────────────── */
function clearEmpty() {
  const e = $('#empty');
  if (e) e.remove();
}

function showEmpty() {
  clearEmpty();
  const nm = store.card ? store.card.name : '她';
  const el = document.createElement('div');
  el.id = 'empty';
  el.className = 'empty-state';
  el.innerHTML = `
    <div class="lead">和${esc(nm)}聊聊</div>
    <div class="sub">她会记住你们说过的话。试一句：</div>
    <div class="chips">
      <button data-say="我今天有点累">我今天有点累</button>
      <button data-say="我在准备考试，压力挺大">我在准备考试</button>
      <button data-say="随便聊聊吧">随便聊聊</button>
    </div>`;
  el.prepend(avatarEl('avatar big'));
  el.querySelectorAll('[data-say]').forEach((b) =>
    b.addEventListener('click', () => {
      $('#input').value = b.dataset.say;
      autoGrow($('#input'));
      send();
    })
  );
  $('#chat').appendChild(el);
}

/* ── 对话 ───────────────────────────────────────────── */
async function send(textOverride) {
  if (store.busy) return;
  if (store.closed) {
    toast('今天结束了，点「继续聊」开始新的一天');
    return;
  }
  const input = $('#input');
  const text = (textOverride != null ? textOverride : input.value).trim();
  if (!text) {
    input.focus();
    return;
  }

  store.busy = true;
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
      body: JSON.stringify({ text, session_id: store.session }),
    });
    const ms = Math.round(performance.now() - t0);

    clearInterval(ph._timer);
    ph.remove();

    const mu = r.memory_used || {};
    const bits = [];
    if (mu.memory_tokens) bits.push(`${mu.memory_tokens} token`);
    if (mu.memory_items) bits.push(`${mu.memory_items} 条记忆`);
    bits.push(`${(ms / 1000).toFixed(1)}s`);
    addMsg('ai', r.reply || '……', bits.join(' · '));

    store.count++;
    updateMeta();
  } catch (e) {
    clearInterval(ph._timer);
    ph.remove();
    lastSpeaker = null;
    const msg =
      e.name === 'AbortError'
        ? '等太久了，可能是模型卡住'
        : /401|invalid.*key|Unauthorized/i.test(e.message)
        ? '密钥好像不对，去「设置」里检查一下'
        : e.code === 'llm_unavailable'
        ? `她这边没接上话：${e.message.replace(/^模型没有返回内容[:：]?/, '')}`
        : e.message.includes('Failed to fetch')
        ? '连不上服务，看看后台还开着吗'
        : e.message;
    addError(msg, () => send(text));
  }

  store.busy = false;
  $('#send').disabled = false;
  input.focus();
}

/* ── 下一天 / 继续聊 ───────────────────────────────── */
async function nextDay() {
  if (store.busy) return;

  if (store.closed) {
    // 继续聊：恢复输入，不清历史
    store.closed = false;
    document.body.classList.remove('closed');
    $('#btnSleep').textContent = '下一天';
    $('#send').disabled = false;
    addMsg('sys', '—— 新的一天 ——');
    return;
  }

  if (store.count === 0) {
    toast('先聊两句，不然没东西可以结算');
    return;
  }

  store.busy = true;
  $('#btnSleep').disabled = true;
  try {
    const r = await api(`/v1/session/sleep?session_id=${store.session}`, { method: 'POST' });
    const cells = [];
    if (r.close && r.close.touched) cells.push([r.close.touched, '强化']);
    if (r.pages) cells.push([r.pages, '沉淀']);
    if (r.dream) cells.push([r.dream.replayed, '做梦']);
    if (!cells.length) cells.push(['—', '今天']);

    const el = document.createElement('div');
    el.className = 'settle';
    el.innerHTML = `<div class="settle-title">一天结束</div>
      <div class="settle-cells">${cells.map(([v, k]) => `<div><b>${esc(v)}</b><span>${esc(k)}</span></div>`).join('')}</div>
      <p class="settle-note">${r.dream ? '她把今天过了一遍。' : ''}</p>`;
    $('#chat').appendChild(el);
    $('#chat').scrollTop = $('#chat').scrollHeight;

    store.closed = true;
    document.body.classList.add('closed');
    $('#btnSleep').textContent = '继续聊';
    $('#send').disabled = true;
  } catch (e) {
    addError(`结算失败：${e.message}`, nextDay);
  }
  store.busy = false;
  $('#btnSleep').disabled = false;
}

/* ── 角色卡 ─────────────────────────────────────────── */
function renderCardPop() {
  // brand 区
  const nm = store.card ? store.card.name : '藏识';
  $('#cardName').textContent = nm;
  const bav = $('#brandAvatar');
  if (bav) {
    if (store.card && store.card.avatar) {
      bav.style.backgroundImage = `url(${store.card.avatar})`;
      bav.style.backgroundSize = 'cover';
      bav.style.backgroundPosition = 'center';
      bav.textContent = '';
    } else {
      bav.style.backgroundImage = '';
      bav.textContent = nm.charAt(0);
    }
  }

  $('#cardList').innerHTML = store.cards
    .map(
      (c) => `<div class="pop-item ${c.session === store.session ? 'on' : ''}" data-card="${esc(c.session)}">
      <b>${esc(c.name)}</b>${c.desc ? `<span>${esc(c.desc)}</span>` : ''}
      <div class="row-btns">
        <button data-edit="${esc(c.session)}">改</button>
        <button data-del="${esc(c.session)}">删</button>
      </div>
    </div>`
    )
    .join('');

  $$('#cardList [data-card]').forEach((el) =>
    el.addEventListener('click', (ev) => {
      if (ev.target.closest('[data-edit]') || ev.target.closest('[data-del]')) return;
      useCard(el.dataset.card);
    })
  );
  $$('#cardList [data-edit]').forEach((b) =>
    b.addEventListener('click', (ev) => {
      ev.stopPropagation();
      editCard(b.dataset.edit);
    })
  );
  $$('#cardList [data-del]').forEach((b) =>
    b.addEventListener('click', (ev) => {
      ev.stopPropagation();
      removeCard(b.dataset.del);
    })
  );
}

async function useCard(sid) {
  if (sid === store.session) {
    closeLayer('cardPop');
    return;
  }
  if (store.busy) {
    toast('她话还没说完，等一下再切');
    return;
  }
  closeLayer('cardPop');
  store.setSession(sid);
  store.count = 0;
  store.closed = false;
  document.body.classList.remove('closed');
  $('#btnSleep').textContent = '下一天';
  $('#send').disabled = false;
  $('#chat').innerHTML = '';
  lastSpeaker = null;
  renderCardPop();
  await restoreHistory();
}

/* ── 建卡 ───────────────────────────────────────────── */
async function createCard() {
  const name = $('#ncName').value.trim();
  if (!name) {
    toast('总得有个名字');
    return;
  }
  const desc = $('#ncDesc').value.trim();
  const seed = $('#ncSeed').value.trim();

  // 编辑已有卡
  if (store.editing) {
    const c = store.cards.find((x) => x.session === store.editing);
    if (c) {
      c.name = name;
      c.desc = desc;
      if (store.pendingAvatar) c.avatar = store.pendingAvatar;
      if (seed) {
        c.seed = seed;
        try {
          await api('/v1/persona', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
              text: `你是${name}。${seed}`,
              session_id: c.session,
              section: 'identity',
              author: 'self',
              protected: true,
              priority: 100,
            }),
          });
        } catch (e) {
          toast(`种子没写进去：${e.message}`);
        }
      }
    }
    store.commit();
    renderCardPop();
    closeLayer('newModal');
    resetCardForm();
    toast('改好了');
    return;
  }

  const card = store.addCard({ name, desc, seed, avatar: store.pendingAvatar });
  // 种子写进人格：她的记忆从这句话开始
  try {
    await api('/v1/persona', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: `你是${name}。${seed || '你记得来过的人，但不多问。'}`,
        session_id: card.session,
        section: 'identity',
        author: 'self',
        protected: true,
        priority: 100,
      }),
    });
  } catch (e) {
    toast(`种子没写进去：${e.message}`);
  }

  closeLayer('newModal');
  resetCardForm();
  renderCardPop();
  await useCard(card.session);
  toast(`「${name}」建好了，TA 应该已经先开口了`);
}

function editCard(sid) {
  const c = store.cards.find((x) => x.session === sid);
  if (!c) return;
  store.editing = sid;
  $('#modalTitle').textContent = '编辑角色';
  $('#ncName').value = c.name;
  $('#ncDesc').value = c.desc || '';
  $('#ncSeed').value = c.seed || '';
  const pv = $('#ncAvatarPrev');
  pv.style.backgroundImage = c.avatar ? `url(${c.avatar})` : '';
  pv.style.backgroundSize = 'cover';
  pv.textContent = c.avatar ? '' : c.name.charAt(0);
  closeLayer('cardPop');
  openLayer('newModal');
  $('#ncName').focus();
}

function resetCardForm() {
  store.editing = null;
  store.pendingAvatar = '';
  $('#modalTitle').textContent = '新角色';
  $('#ncName').value = '';
  $('#ncDesc').value = '';
  $('#ncSeed').value = '';
  const pv = $('#ncAvatarPrev');
  pv.style.backgroundImage = '';
  pv.textContent = '?';
}

async function removeCard(sid) {
  const c = store.cards.find((x) => x.session === sid);
  if (!confirm(`删掉「${c ? c.name : sid}」？TA 的全部记忆也会一起删掉，找不回来。`)) return;
  if (store.cards.length === 1) {
    toast('最后一个角色不能删');
    return;
  }
  try {
    await api(`/v1/wipe?session_id=${encodeURIComponent(sid)}`, { method: 'POST' });
  } catch (e) {
    /* 记忆清理失败不阻断 */
  }
  store.remove(sid);
  closeLayer('cardPop');
  renderCardPop();
  $('#chat').innerHTML = '';
  lastSpeaker = null;
  store.count = 0;
  store.closed = false;
  document.body.classList.remove('closed');
  $('#btnSleep').textContent = '下一天';
  $('#send').disabled = false;
  await restoreHistory();
  toast('删了');
}

/* ── 建卡向导：独立模式，不挂记忆 ─────────────────────── */
const WIZ = [
  { key: 'name', q: '先给这个角色起个名字？', ph: '比如：深夜便利店' },
  { key: 'desc', q: '用一句话介绍 TA（显示在角色列表里）。', ph: '话不多，记性很好' },
  { key: 'seed', q: 'TA 的来历是什么？这会成为 TA 记忆的起点。', ph: '你在这里待了很多年…' },
  { key: 'firstMes', q: 'TA 开口第一句说什么？', ph: '……' },
];

const wiz = { step: 0, data: {}, active: false };

function addWiz(role, text) {
  const el = document.createElement('div');
  el.className = `msg ${role}`;
  const b = document.createElement('div');
  b.className = 'body';
  b.textContent = text;
  el.appendChild(b);
  $('#wizChat').appendChild(el);
  $('#wizChat').scrollTop = $('#wizChat').scrollHeight;
  return el;
}

function wizStart() {
  openLayer('wizModal');
  wiz.step = 0;
  wiz.data = {};
  wiz.active = true;
  $('#wizChat').innerHTML = '';
  $('#wizActions').hidden = true;
  $('#wizInput').disabled = false;
  $('#wizSend').disabled = false;
  $('#wizInput').placeholder = WIZ[0].ph;
  addWiz('sys', '四个小问题，答完就有了一个新角色');
  addWiz('ai', WIZ[0].q);
  $('#wizInput').focus();
}

function wizAnswer() {
  if (!wiz.active || wiz.step >= WIZ.length) return;
  const input = $('#wizInput');
  const val = input.value.trim();
  if (!val) return;
  input.value = '';
  addWiz('me', val);
  wiz.data[WIZ[wiz.step].key] = val;
  wiz.step++;

  if (wiz.step < WIZ.length) {
    setTimeout(() => {
      addWiz('ai', WIZ[wiz.step].q);
      input.placeholder = WIZ[wiz.step].ph;
    }, 240);
  } else {
    wizConfirm();
  }
}

function wizConfirm() {
  wiz.active = false;
  $('#wizInput').disabled = true;
  $('#wizSend').disabled = true;
  $('#wizInput').placeholder = '确认后创建';
  addWiz('ai', '都齐了。确认一下：');
  addWiz(
    'sys',
    `名字：${wiz.data.name}\n一句话：${wiz.data.desc || '（无）'}\n来历：${wiz.data.seed || '（空白开始）'}\n开场：${wiz.data.firstMes || '……'}`
  );
  $('#wizActions').hidden = false;
}

async function wizDone() {
  const card = store.addCard({
    name: wiz.data.name,
    desc: wiz.data.desc,
    seed: wiz.data.seed,
  });
  try {
    await api('/v1/persona', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        text: `你是${card.name}。${wiz.data.seed || '你记得来过的人，但不多问。'}`,
        session_id: card.session,
        section: 'identity',
        author: 'self',
        protected: true,
        priority: 100,
      }),
    });
  } catch (e) {
    toast(`种子没写进去：${e.message}`);
  }
  closeLayer('wizModal');
  wiz.active = false;
  renderCardPop();
  await useCard(card.session);
  toast(`「${card.name}」建好了，TA 应该已经先开口了`);
}

function wizClose() {
  closeLayer('wizModal');
  wiz.active = false;
  $('#wizInput').disabled = false;
  $('#wizSend').disabled = false;
}

/* ── 设置 ───────────────────────────────────────────── */
let settingsCache = null;

async function openSettings() {
  openLayer('setModal');
  $('#setNote').textContent = '读中…';
  try {
    const s = await api('/v1/settings');
    settingsCache = s;
    const cloud = s.llm_hot === 'openai';
    $('#selProv').value = cloud ? 'openai' : 'ollama';
    $('#setBase').value = s.openai_base || '';
    $('#setKey').value = '';
    $('#setModel').value = s.openai_model || '';
    $('#setLocal').value = s.llm_hot_model || '';
    $('#dbgInfo').checked = store.debug;
    toggleCloud(cloud);
    $('#setNote').innerHTML = cloud
      ? `当前：<b>${esc(s.openai_model || '未填')}</b> · 密钥${s.openai_key_set ? '已配置' : '<span class="warn">未填</span>'}`
      : `当前：本地 <b>${esc(s.llm_hot_model)}</b>`;
  } catch (e) {
    $('#setNote').textContent = `读不到设置：${e.message}`;
  }
}

function toggleCloud(cloud) {
  $('#cloudBox').hidden = !cloud;
  $('#localBox').hidden = cloud;
}

async function saveSettings() {
  const cloud = $('#selProv').value === 'openai';
  const body = {};
  if (cloud) {
    body.llm_hot = 'openai';
    body.llm_deep = 'openai';
    body.openai_base = $('#setBase').value.trim();
    body.openai_model = $('#setModel').value.trim() || 'deepseek-chat';
    const key = $('#setKey').value.trim();
    if (key) body.openai_key = key;
    if (!body.openai_base) return toast('云端要填接口地址');
    if (!key && !(settingsCache && settingsCache.openai_key_set)) return toast('云端要填密钥');
  } else {
    body.llm_hot = 'ollama';
    body.llm_deep = 'ollama';
    body.llm_hot_model = $('#setLocal').value.trim() || 'minicpm5-2b:q8-64k';
    body.llm_deep_model = body.llm_hot_model;
  }
  try {
    const r = await api('/v1/settings', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    closeLayer('setModal');
    toast(r.reloaded ? '换好了' : '写进去了，但重载失败，重启服务生效');
  } catch (e) {
    toast(`没存成：${e.message}`);
  }
}

/* ── 背景 ───────────────────────────────────────────── */
function applyBg() {
  const bg = localStorage.getItem(K_BG);
  const p = $('#bgPrev');
  if (bg) {
    document.body.style.backgroundImage = `linear-gradient(rgba(17,18,20,.72),rgba(17,18,20,.72)),url(${bg})`;
    document.body.classList.add('has-bg');
    if (p) {
      p.style.backgroundImage = `url(${bg})`;
      p.textContent = '';
    }
  } else {
    document.body.style.backgroundImage = '';
    document.body.classList.remove('has-bg');
    if (p) {
      p.style.backgroundImage = '';
      p.textContent = '无';
    }
  }
}

/** 图片转 dataURL，带大小上限与压缩 */
function readImage(file, maxKB, cb) {
  if (!file) return;
  if (file.size > maxKB * 1024 * 2) {
    toast(`图太大了（${Math.round(file.size / 1024)}KB），换张小的`);
    return;
  }
  const rd = new FileReader();
  rd.onload = () => {
    const img = new Image();
    img.onload = () => {
      // 压到最长边 512，够头像/背景用
      const scale = Math.min(1, 512 / Math.max(img.width, img.height));
      const c = document.createElement('canvas');
      c.width = Math.round(img.width * scale);
      c.height = Math.round(img.height * scale);
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      cb(c.toDataURL('image/jpeg', 0.8));
    };
    img.onerror = () => toast('这张图读不出来');
    img.src = rd.result;
  };
  rd.readAsDataURL(file);
}

/* ── 记忆抽屉 ───────────────────────────────────────── */
async function loadMemory() {
  const body = $('#drawerBody');
  body.innerHTML = '<div class="skeleton"><span></span><span></span><span></span></div>';
  try {
    const [eps, facts, ents, st, notes] = await Promise.all([
      api(`/v1/episodes?session_id=${store.session}&limit=30`),
      api(`/v1/facts?session_id=${store.session}&limit=40`),
      api(`/v1/entities?session_id=${store.session}&limit=20`),
      api(`/v1/state/${encodeURIComponent('我')}?session_id=${store.session}`).catch(() => ({ slots: [] })),
      api(`/v1/notebook?session_id=${store.session}&limit=20`).catch(() => []),
    ]);

    const has = (eps.items || []).length + (facts || []).length + (ents || []).length + (st.slots || []).length;
    if (!has) {
      body.innerHTML = `<div class="empty-state small">${''}
        <div class="lead">还什么都没记住</div>
        <div class="sub">聊两句，她就开始记了。</div></div>`;
      const e = body.querySelector('.empty-state');
      if (e) e.prepend(avatarEl('avatar big'));
      return;
    }

    const sec = (title, items, render) => {
      if (!items || !items.length) return '';
      return `<section class="sec"><h3>${title} <span class="cnt">${items.length}</span></h3>
        <ul>${items.map((i) => `<li>${render(i)}</li>`).join('')}</ul></section>`;
    };

    const cnT = { resident: '常驻', retrieval: '笔记', inbox: '速记', important: '重要' };

    body.innerHTML =
      sec('当前状态', st.slots, (s) => `<b>${esc(s.slot)}</b>：${esc(s.value)}<div class="row"><span>${fmt(s.since)}</span></div>`) +
      sec('关于你', ents, (e) => `<b>${esc(e.name)}</b><div class="row"><span>${e.fact_count} 条事实</span></div>`) +
      sec(
        '事实',
        facts,
        (f) =>
          `${esc(f.subject)}${f.polarity < 0 ? '不' : ''}${esc(f.predicate)} ${esc(f.object)}<div class="row"><span>${fmt(f.valid_from)}</span></div>`
      ) +
      sec(
        '笔记',
        notes,
        (n) =>
          `${esc(n.text || n.what || '')}<div class="row"><span>${cnT[n.tier] || esc(n.tier)}</span>${
            n.next ? `<span>下次：${esc(n.next)}</span>` : ''
          }`
      ) +
      sec(
        '说过的话',
        eps.items,
        (e) => `${esc(e.text)}<div class="row"><span>${cnT[e.tier] || esc(e.tier)}</span><span>${fmt(e.t_event)}</span></div>`
      );
  } catch (e) {
    body.innerHTML = `<div class="empty-state small"><div class="lead">读不出来</div>
      <div class="sub">${esc(e.message)}</div></div>`;
  }
}

function fmt(ts) {
  if (!ts) return '';
  const d = new Date(ts * 1000);
  return `${d.getMonth() + 1}/${d.getDate()} ${String(d.getHours()).padStart(2, '0')}:${String(d.getMinutes()).padStart(2, '0')}`;
}

function updateMeta() {
  $('#meta').textContent = store.count ? `已聊 ${store.count} 轮` : '';
}

/* ── 历史恢复 ───────────────────────────────────────── */
async function restoreHistory() {
  try {
    const eps = await api(`/v1/episodes?session_id=${store.session}&limit=50`);
    const dialog = (eps.items || []).filter((e) => e.role === 'user' || e.role === 'assistant').reverse();

    if (!dialog.length) {
      showEmpty();
      return;
    }
    let lastDay = null;
    for (const e of dialog) {
      const d = new Date(e.t_event * 1000);
      const key = `${d.getMonth()}-${d.getDate()}`;
      if (key !== lastDay) {
        lastDay = key;
        addMsg('sys', `—— ${d.getMonth() + 1}月${d.getDate()}日 ——`);
      }
      addMsg(e.role === 'user' ? 'me' : 'ai', e.text);
    }
    store.count = Math.ceil(dialog.length / 2);
    updateMeta();
    $('#chat').scrollTop = $('#chat').scrollHeight;
  } catch (e) {
    showEmpty();
  }
}

function autoGrow(el) {
  el.style.height = 'auto';
  el.style.height = Math.min(el.scrollHeight, 160) + 'px';
}

/* ── 绑定：无条件先跑，不依赖网络 ─────────────────────── */
function bindAll() {
  // 对话
  $('#send').addEventListener('click', () => send());
  $('#input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      send();
    }
  });
  $('#input').addEventListener('input', (e) => autoGrow(e.target));

  // 顶栏
  $('#btnMemory').addEventListener('click', () => {
    openLayer('drawer');
    $('#mask').classList.add('on');
    loadMemory();
  });
  $('#closeDrawer').addEventListener('click', closeAllLayers);
  $('#mask').addEventListener('click', closeAllLayers);
  $('#btnSleep').addEventListener('click', nextDay);
  $('#brand').addEventListener('click', () => {
    const el = $('#cardPop');
    if (el.hidden) {
      renderCardPop();
      openLayer('cardPop');
    } else {
      closeLayer('cardPop');
    }
  });
  $('#btnSettings').addEventListener('click', openSettings);
  $('#btnWizard').addEventListener('click', wizStart);

  // 角色卡与新建
  $('#newCard').addEventListener('click', () => {
    resetCardForm();
    closeLayer('cardPop');
    openLayer('newModal');
    $('#ncName').focus();
  });
  $('#ncCancel').addEventListener('click', () => {
    resetCardForm();
    closeLayer('newModal');
  });
  $('#ncSave').addEventListener('click', createCard);
  $('#ncAvatar').addEventListener('change', (e) => {
    readImage(e.target.files[0], 300, (d) => {
      store.pendingAvatar = d;
      const pv = $('#ncAvatarPrev');
      pv.style.backgroundImage = `url(${d})`;
      pv.style.backgroundSize = 'cover';
      pv.style.backgroundPosition = 'center';
      pv.textContent = '';
    });
  });

  // 设置
  $('#setCancel').addEventListener('click', () => closeLayer('setModal'));
  $('#setSave').addEventListener('click', saveSettings);
  $('#selProv').addEventListener('change', (e) => toggleCloud(e.target.value === 'openai'));
  $('#dbgInfo').addEventListener('change', (e) => {
    store.debug = e.target.checked;
    try {
      localStorage.setItem('alaya.debug', store.debug ? '1' : '');
    } catch (err) {}
  });
  $('#bgFile').addEventListener('change', (e) => {
    readImage(e.target.files[0], 800, (d) => {
      try {
        localStorage.setItem(K_BG, d);
        applyBg();
        toast('背景换好了');
      } catch (err) {
        toast('这张图存不下，换张小点的');
      }
    });
  });
  $('#bgClear').addEventListener('click', () => {
    localStorage.removeItem(K_BG);
    applyBg();
  });

  // 向导
  $('#wizSend').addEventListener('click', wizAnswer);
  $('#wizInput').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
      e.preventDefault();
      wizAnswer();
    }
  });
  $('#wizRestart').addEventListener('click', wizStart);
  $('#wizDone').addEventListener('click', wizDone);
  $('#wizClose').addEventListener('click', wizClose);

  // Esc 统一关最上层
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeTopLayer();
  });

  // 点空白关角色浮层
  document.addEventListener('click', (e) => {
    if (!e.target.closest('#cardPop') && !e.target.closest('#brand')) closeLayer('cardPop');
  });
}

/* ── 启动 ───────────────────────────────────────────── */
async function boot() {
  store.load();
  try {
    store.debug = localStorage.getItem('alaya.debug') === '1';
  } catch (e) {}

  // 先绑，再发请求：这两件事的顺序不能反
  bindAll();
  renderCardPop();
  applyBg();

  try {
    await api(`/v1/session/wake?session_id=${store.session}`, { method: 'POST' });
  } catch (e) {
    addError('连不上后台服务，先确认它开着', () => location.reload());
    return;
  }

  await restoreHistory();
  $('#input').focus();
}

boot().catch((e) => {
  console.error('boot failed', e);
  const el = document.createElement('div');
  el.className = 'err-bar';
  el.style.cssText = 'position:fixed;top:0;left:0;right:0;z-index:60';
  el.innerHTML = `<span>启动失败：${esc(e && e.message)}</span>`;
  document.body.appendChild(el);
});
