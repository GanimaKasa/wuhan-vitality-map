const LABEL_COLS = [
  "工作日_深夜", "工作日_早高峰", "工作日_日间", "工作日_晚高峰", "工作日_夜间",
  "休息日_深夜", "休息日_早间", "休息日_日间", "休息日_傍晚", "休息日_夜间",
];
const GATE_NAMES = [
  ["poi", "POI"], ["road", "路网"], ["landuse", "土地利用"], ["nightlight", "夜光"],
  ["sentinel2", "Sentinel-2"], ["weibo", "微博"], ["streetview", "街景CLIP"], ["gnn", "GNN路网图"],
];
const COLOR_RAMP = ["#4393c3", "#74c476", "#fee08b", "#fd8d3c", "#d6604d"];

// 原始格网centroid是按统一经纬度间隔生成的，经纬度用了相同的度数步长，
// 不是严格按物理米数换算的正方形（经度方向按250m/cos(纬度)算会比实际间隔更宽，导致相邻格网重叠）。
// 用实测的精确间隔(对全部centroid最近邻距离取中位数)取一半，做到严丝合缝拼接，不留人为缝隙。
const GRID_SPACING_DEG = 0.00224578;
const HALF_LAT = GRID_SPACING_DEG / 2;
const HALF_LNG = GRID_SPACING_DEG / 2;

const NORMAL_FILL_OPACITY = 0.45;
const NORMAL_BORDER_WEIGHT = 0.5;

let geojsonData = null;
let markerByGridId = new Map();
let markerLayer = null;
let weiboMarkerLayer = null;
let lastWeiboData = null; // {total_relevant, returned, posts}——切换排序时对这份数据纯前端重排，不重新请求后端
let currentPeriod = LABEL_COLS[2]; // 默认工作日_日间
let fillHidden = false;   // true时不填充颜色
let borderHidden = false; // true时连边框也不显示

function currentStyle() {
  return { fillOpacity: fillHidden ? 0 : NORMAL_FILL_OPACITY, weight: borderHidden ? 0 : NORMAL_BORDER_WEIGHT };
}

const map = L.map("mapContainer", { preferCanvas: true }).setView([30.58, 114.28], 11.5);
L.tileLayer("https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png", {
  attribution: "&copy; OpenStreetMap contributors",
  maxZoom: 18,
}).addTo(map);

function escapeHtml(str) {
  const div = document.createElement("div");
  div.textContent = str ?? "";
  return div.innerHTML;
}

function valueToColor(value, min, max) {
  if (max === min) return COLOR_RAMP[0];
  const t = Math.max(0, Math.min(1, (value - min) / (max - min)));
  const idx = Math.min(COLOR_RAMP.length - 2, Math.floor(t * (COLOR_RAMP.length - 1)));
  return COLOR_RAMP[idx];
}

function gateBarHtml(props) {
  return GATE_NAMES.map(([key, zh]) => {
    const v = props[`gate_${key}`] ?? 0;
    const pct = Math.round(v * 100);
    return `<div class="gate-bar-row">
      <span class="gate-bar-label">${zh}</span>
      <span class="gate-bar-track"><span class="gate-bar-fill" style="width:${pct}%"></span></span>
      <span>${v.toFixed(3)}</span>
    </div>`;
  }).join("");
}

function popupHtml(props) {
  const missBadges = [];
  if (props.missing_weibo) missBadges.push('<span class="missing-badge">微博缺失</span>');
  if (props.missing_streetview) missBadges.push('<span class="missing-badge">街景缺失</span>');

  const periodRows = LABEL_COLS.map(
    (col) => `<div>${col}：${props[`pred_${col}`].toFixed(2)}</div>`
  ).join("");

  return `<div class="grid-popup">
    <h3>格网 #${props.grid_id}（${props.district}）</h3>
    ${missBadges.join("")}
    <div style="margin:6px 0;"><b>10个时段预测活力值</b>${periodRows}</div>
    <div style="margin-top:6px;"><b>8模态门控权重</b>${gateBarHtml(props)}</div>
    <button class="weibo-activity-btn" data-grid-id="${props.grid_id}">查看该地区微博动态</button>
    <div class="activity-container" id="activity-container-${props.grid_id}"></div>
  </div>`;
}

function buildActivityHtml(data) {
  if (!data.count) {
    return '<div class="activity-summary">该地区暂无可用的微博文本样本。</div>';
  }
  const samples = data.posts.slice(0, 3).map((p) => {
    const text = escapeHtml((p.text || "").trim().slice(0, 50));
    const placeType = escapeHtml(p.place_type || "");
    return `<div class="activity-sample">「${text}」${placeType ? `（${placeType}）` : ""}</div>`;
  }).join("");
  return `<div class="activity-summary"><b>活动解读（共${data.count}条样本）</b><br>${escapeHtml(data.summary)}</div>${samples}`;
}

document.addEventListener("click", async (e) => {
  const btn = e.target.closest(".weibo-activity-btn");
  if (!btn) return;
  const gridId = btn.dataset.gridId;
  const container = document.getElementById(`activity-container-${gridId}`);
  if (!container) return;
  container.textContent = "加载中...";
  try {
    const res = await fetch(`/api/weibo/grid/${gridId}`);
    const data = await res.json();
    container.innerHTML = res.ok ? buildActivityHtml(data) : escapeHtml(data.detail || "加载失败，请稍后再试。");
  } catch {
    container.textContent = "加载失败，请稍后再试。";
  }
});

// 颜色映射统一用"全部10个时段"合在一起算的分位数范围（而不是每次切换时段各自
// 用当前时段的min/max归一化），这样颜色深浅才能真实反映跨时段的绝对活力高低——
// 否则同一个格网哪怕日间活力远高于夜间，切换时段后颜色可能看起来差不多甚至更淡。
let GLOBAL_COLOR_MIN = 0;
let GLOBAL_COLOR_MAX = 1;

function computeGlobalColorRange() {
  const all = [];
  for (const feat of geojsonData.features) {
    for (const col of LABEL_COLS) {
      all.push(feat.properties[`pred_${col}`]);
    }
  }
  all.sort((a, b) => a - b);
  const pct = (p) => all[Math.min(all.length - 1, Math.floor(p * (all.length - 1)))];
  GLOBAL_COLOR_MIN = pct(0.02);
  GLOBAL_COLOR_MAX = pct(0.98);
}

function renderMarkers() {
  if (markerLayer) map.removeLayer(markerLayer);
  markerByGridId.clear();

  markerLayer = L.layerGroup();
  for (const feat of geojsonData.features) {
    const [lng, lat] = feat.geometry.coordinates;
    const props = feat.properties;
    const value = props[`pred_${currentPeriod}`];
    const color = valueToColor(value, GLOBAL_COLOR_MIN, GLOBAL_COLOR_MAX);
    const bounds = [
      [lat - HALF_LAT, lng - HALF_LNG],
      [lat + HALF_LAT, lng + HALF_LNG],
    ];
    const marker = L.rectangle(bounds, {
      color: color,
      fillColor: color,
      ...currentStyle(),
    });
    marker.bindPopup(() => popupHtml(props));
    marker.addTo(markerLayer);
    markerByGridId.set(props.grid_id, marker);
  }
  markerLayer.addTo(map);
}

function toggleFill(hidden) {
  fillHidden = hidden;
  markerByGridId.forEach((m) => m.setStyle({ color: m.options.fillColor, ...currentStyle() }));
}

function toggleBorder(hidden) {
  borderHidden = hidden;
  markerByGridId.forEach((m) => m.setStyle({ color: m.options.fillColor, ...currentStyle() }));
}

function highlightGridIds(gridIds) {
  markerByGridId.forEach((m) => m.setStyle({ color: m.options.fillColor, ...currentStyle() }));
  let firstMarker = null;
  for (const gid of gridIds) {
    const m = markerByGridId.get(gid);
    if (m) {
      m.setStyle({ weight: 2.5, color: "#ffffff", fillOpacity: 0.85 });
      m.bringToFront();
      if (!firstMarker) firstMarker = m;
    }
  }
  if (firstMarker) {
    map.flyTo(firstMarker.getBounds().getCenter(), 14);
    firstMarker.openPopup();
  }
}

function initPeriodSelect() {
  const sel = document.getElementById("periodSelect");
  for (const col of LABEL_COLS) {
    const opt = document.createElement("option");
    opt.value = col;
    opt.textContent = col.replace("_", " · ");
    sel.appendChild(opt);
  }
  sel.value = currentPeriod;
  sel.addEventListener("change", () => {
    currentPeriod = sel.value;
    renderMarkers();
  });
}

function clearWeiboResults() {
  if (weiboMarkerLayer) map.removeLayer(weiboMarkerLayer);
  weiboMarkerLayer = null;
  lastWeiboData = null;
  document.getElementById("weiboResultList").innerHTML = "";
  document.getElementById("weiboSearchInfo").textContent = "";
}

function getSortedWeiboPosts() {
  if (!lastWeiboData) return [];
  const sortByLikes = document.getElementById("weiboSortByLikesToggle").checked;
  return sortByLikes
    ? [...lastWeiboData.posts].sort((a, b) => b.like_count - a.like_count)
    : lastWeiboData.posts;
}

function renderWeiboResults() {
  if (weiboMarkerLayer) map.removeLayer(weiboMarkerLayer);
  weiboMarkerLayer = L.layerGroup();
  const listEl = document.getElementById("weiboResultList");
  listEl.innerHTML = "";

  for (const post of getSortedWeiboPosts()) {
    const marker = L.circleMarker([post.lat, post.lng], {
      radius: 5,
      color: "#ffffff",
      weight: 1,
      fillColor: "#f2a541",
      fillOpacity: 0.9,
    });
    const text = escapeHtml((post.text || "").trim().slice(0, 80));
    const placeType = escapeHtml(post.place_type || "");
    marker.bindPopup(
      `<div class="grid-popup"><b>格网 #${post.grid_id}</b>${placeType ? `（${placeType}）` : ""} · ❤${post.like_count}<br>` +
      `<span style="color:#888;font-size:11px;">${escapeHtml(post.post_time || "")}</span><br>${text}</div>`
    );
    marker.addTo(weiboMarkerLayer);

    const item = document.createElement("div");
    item.className = "weibo-item";
    item.innerHTML =
      `<div class="weibo-meta">格网#${post.grid_id} · ${placeType || "未知类型"} · ❤${post.like_count} · ${escapeHtml(post.post_time || "")}</div>` +
      `<div class="weibo-text">${text}</div>`;
    item.addEventListener("click", () => {
      map.flyTo([post.lat, post.lng], 15);
      marker.openPopup();
    });
    listEl.appendChild(item);
  }
  weiboMarkerLayer.addTo(map);

  const sortByLikes = document.getElementById("weiboSortByLikesToggle").checked;
  const sortLabel = sortByLikes ? "点赞数从高到低" : "相关度+热度综合";
  document.getElementById("weiboSearchInfo").textContent =
    `语义检索到 ${lastWeiboData.total_relevant} 条相关微博，按${sortLabel}排序展示前 ${lastWeiboData.returned} 条`;
}

async function weiboSearch() {
  const keyword = document.getElementById("weiboSearchInput").value.trim();
  if (!keyword) return;
  const topN = document.getElementById("weiboTopNInput").value || 150;
  document.getElementById("weiboSearchInfo").textContent = "搜索中...";
  try {
    const res = await fetch(`/api/weibo/search?keyword=${encodeURIComponent(keyword)}&top_n=${topN}`);
    const data = await res.json();
    if (!res.ok) {
      document.getElementById("weiboSearchInfo").textContent = data.detail || "搜索失败，请稍后再试。";
      return;
    }
    lastWeiboData = data;
    renderWeiboResults();
  } catch {
    document.getElementById("weiboSearchInfo").textContent = "搜索失败，请稍后再试。";
  }
}

async function initDistrictList() {
  const res = await fetch("/api/districts");
  const { districts } = await res.json();
  const list = document.getElementById("districtList");
  for (const d of districts) {
    const opt = document.createElement("option");
    opt.value = d;
    list.appendChild(opt);
  }
}

async function runTopSearch(order) {
  const district = document.getElementById("districtInput").value.trim();
  const params = new URLSearchParams({ period: currentPeriod, order, topn: "10" });
  if (district) params.set("district", district);
  const res = await fetch(`/api/search?${params.toString()}`);
  const data = await res.json();
  const ids = data.results.map((r) => r.grid_id);
  highlightGridIds(ids);
  document.getElementById("searchResultInfo").textContent =
    `匹配到 ${data.count} 个格网，已在地图上高亮`;
}

function appendMessage(role, text) {
  const box = document.getElementById("chatMessages");
  const div = document.createElement("div");
  div.className = `msg ${role}`;
  div.textContent = text;
  box.appendChild(div);
  box.scrollTop = box.scrollHeight;
}

async function sendChat() {
  const input = document.getElementById("chatInput");
  const question = input.value.trim();
  if (!question) return;
  appendMessage("user", question);
  input.value = "";

  const res = await fetch("/api/chat", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const data = await res.json();
  if (!res.ok) {
    appendMessage("bot", data.detail || "请求失败，请稍后再试。");
    return;
  }
  appendMessage("bot", data.answer);
  if (data.highlight_grid_ids && data.highlight_grid_ids.length) {
    highlightGridIds(data.highlight_grid_ids);
  }
}

async function main() {
  initPeriodSelect();
  await initDistrictList();

  const res = await fetch("/api/geojson");
  geojsonData = await res.json();
  computeGlobalColorRange();
  renderMarkers();

  document.getElementById("topHighBtn").addEventListener("click", () => runTopSearch("desc"));
  document.getElementById("topLowBtn").addEventListener("click", () => runTopSearch("asc"));
  document.getElementById("clearFilterBtn").addEventListener("click", () => {
    document.getElementById("districtInput").value = "";
    document.getElementById("searchResultInfo").textContent = "";
    renderMarkers();
  });
  document.getElementById("fillToggle").addEventListener("change", (e) => toggleFill(e.target.checked));
  document.getElementById("borderToggle").addEventListener("change", (e) => toggleBorder(e.target.checked));
  document.getElementById("weiboSearchBtn").addEventListener("click", weiboSearch);
  document.getElementById("weiboSearchInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") weiboSearch();
  });
  document.getElementById("weiboClearBtn").addEventListener("click", clearWeiboResults);
  document.getElementById("weiboSortByLikesToggle").addEventListener("change", () => {
    if (lastWeiboData) renderWeiboResults();
  });
  document.getElementById("chatSendBtn").addEventListener("click", sendChat);
  document.getElementById("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
  });

  appendMessage("bot", "你好，我是活力地图问答助手，已接入DeepSeek。可以问我“武昌区晚上活力怎么样”这类问题，也可以在上面的微博搜索框里搜关键词看真实网友动态。");
}

main();
