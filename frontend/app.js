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
let studyAreaBoundaryData = null; // 三环整体轮廓GeoJSON，首次开启时拉取一次并缓存
let studyAreaBoundaryLayer = null;
let weiboMarkerLayer = null;
let lastWeiboData = null; // {total_relevant, returned, posts}——切换排序时对这份数据纯前端重排，不重新请求后端
let allPoiData = null; // {count, posts}——全部微博POI轻量数据，只在首次开启开关时拉取一次并缓存
let allPoiClusterLayer = null;
let gridIdToLatLng = new Map(); // grid_id -> [lng, lat]，3D柱状图给微博数据按格网聚合定位用
let maplibreMap = null;
let deckOverlay = null;
let vitality3DOn = false;
let weibo3DOn = false;
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

// 2D地图本身不支持旋转（格网是按正北朝上画的矩形），指南针在这里是固定指北的
// 静态图标，不是实时指向控件——主要是跟3D视图的罗盘控件保持视觉一致，用户
// 一眼能确认"上=北"，两种模式切换时不会有"这个东西怎么不见了"的落差感。
const CompassControl = L.Control.extend({
  options: { position: "topright" },
  onAdd: function () {
    const div = L.DomUtil.create("div", "compass-badge");
    div.title = "正北朝上（2D地图不支持旋转）";
    div.innerHTML =
      '<svg width="34" height="34" viewBox="0 0 34 34">' +
      '<circle cx="17" cy="17" r="16" fill="rgba(20,24,32,0.85)" stroke="rgba(255,255,255,0.25)"/>' +
      '<polygon points="17,6 21,17 17,14 13,17" fill="#d6604d"/>' +
      '<polygon points="17,28 13,17 17,20 21,17" fill="#cfd4dd"/>' +
      '<text x="17" y="10" text-anchor="middle" font-size="7" fill="#ffffff" font-weight="bold">N</text>' +
      "</svg>";
    return div;
  },
});
map.addControl(new CompassControl());

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

function hexToRgb(hex) {
  const n = parseInt(hex.slice(1), 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
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
  const allPosts = data.posts.map((p) => {
    const text = escapeHtml((p.text || "").trim());
    const placeType = escapeHtml(p.place_type || "");
    const postTime = escapeHtml(p.post_time || "");
    return `<div class="activity-post-item">
      <div class="activity-post-meta">${placeType || "未知类型"}${postTime ? ` · ${postTime}` : ""}</div>
      <div class="activity-post-text">${text}</div>
    </div>`;
  }).join("");
  return `<div class="activity-summary"><b>活动解读（共${data.count}条样本）</b><br>${escapeHtml(data.summary)}</div>${samples}
    <div class="activity-post-list">${allPosts}</div>`;
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

// ==============================================================
// 3D柱状图：deck.gl的Leaflet桥接库(deck.gl-leaflet)已停止维护且有已知bug
// （LeafletLayer undefined），所以不在现有Leaflet画布上叠加3D图层，而是用
// deck.gl官方支持、CDN可直接用的独立模式——开关打开时切到一个deck.gl自带
// 底图的3D画布（可倾斜/旋转/缩放），关闭时切回Leaflet 2D地图，两者共享同一
// 经纬度坐标系统。
//
// 视觉效果参考了deck.gl官方文档+可视化配色的通用做法：
// 1. 高度用平方根拉伸（cartography里经典的Flannery面积补偿类思路）而不是
//    纯线性映射到固定的[MIN_BAR_HEIGHT, MAX_BAR_HEIGHT]米数区间——线性映射
//    在数据大部分集中在低值、只有少数极端高值时，会让大多数柱子被压成一样
//    矮的"地毯"，平方根让中低值也能拉开明显的高度差，视觉冲击力更强。
// 2. 颜色用plasma色系（感知均匀、对比强烈，比默认蓝-绿-黄柱冲击力更强，
//    是数据可视化里"要视觉冲击力"场景的推荐色系）而不是继续用2D地图那套
//    偏柔和的色阶。
// 3. flatShading+方向光源，让柱子之间的高度差通过明暗对比更直观。
// ==============================================================

const MIN_BAR_HEIGHT = 80; // 最矮的柱子也保留这么高，不会被极端值压成看不见的薄片
const MAX_BAR_HEIGHT = 3000;

// matplotlib plasma色系的关键取样点（感知均匀、高对比，数据可视化里公认比
// 默认蓝绿黄红更有视觉冲击力），用线性插值在取样点之间过渡。
const PLASMA_STOPS = ["#0d0887", "#7e03a8", "#cc4778", "#f89441", "#f0f921"].map(hexToRgb);

function plasmaColor(t) {
  t = Math.max(0, Math.min(1, t));
  const n = PLASMA_STOPS.length - 1;
  const scaled = t * n;
  const i = Math.min(n - 1, Math.floor(scaled));
  const frac = scaled - i;
  const [r0, g0, b0] = PLASMA_STOPS[i];
  const [r1, g1, b1] = PLASMA_STOPS[i + 1];
  return [
    Math.round(r0 + (r1 - r0) * frac),
    Math.round(g0 + (g1 - g0) * frac),
    Math.round(b0 + (b1 - b0) * frac),
    255,
  ];
}

// 平方根拉伸：把value在[domainMin, domainMax]里的相对位置t，用sqrt(t)而不是
// t本身去插值到目标高度区间——低值段的高度差会被放大，不会被高值一家独大压平。
function scaleElevation(value, domainMin, domainMax) {
  if (domainMax <= domainMin) return MIN_BAR_HEIGHT;
  const t = Math.max(0, Math.min(1, (value - domainMin) / (domainMax - domainMin)));
  return MIN_BAR_HEIGHT + Math.sqrt(t) * (MAX_BAR_HEIGHT - MIN_BAR_HEIGHT);
}

function buildVitalityColumnLayer() {
  const data = geojsonData.features.map((feat) => ({
    position: feat.geometry.coordinates,
    value: feat.properties[`pred_${currentPeriod}`],
  }));
  return new deck.ColumnLayer({
    id: "vitality-3d",
    data,
    diskResolution: 6,
    radius: 100,
    extruded: true,
    flatShading: true,
    material: { ambient: 0.35, diffuse: 0.6, shininess: 32, specularColor: [255, 255, 255] },
    getPosition: (d) => d.position,
    getElevation: (d) => scaleElevation(d.value, GLOBAL_COLOR_MIN, GLOBAL_COLOR_MAX),
    getFillColor: (d) => {
      const t = (d.value - GLOBAL_COLOR_MIN) / (GLOBAL_COLOR_MAX - GLOBAL_COLOR_MIN || 1);
      return plasmaColor(t);
    },
    pickable: true,
  });
}

// 微博3D柱状图的数据源是"全部微博POI"（受当前类别/点赞数筛选影响），不是
// 语义搜索结果——语义搜索结果条数太少太稀疏（通常几十到一两百条分摊到几十
// 个格网），柱子普遍只有1~3条，视觉上没有区分度；全量POI按格网聚合密度
// 分布差异大得多，才有3D热力对比的意义。
function buildWeiboColumnLayer() {
  const filtered = getFilteredAllPois();
  if (!filtered.length) return null;

  const counts = new Map();
  for (const post of filtered) {
    counts.set(post.grid_id, (counts.get(post.grid_id) || 0) + 1);
  }
  const data = [];
  for (const [gridId, count] of counts) {
    const pos = gridIdToLatLng.get(gridId);
    if (pos) data.push({ position: pos, count, gridId });
  }
  if (!data.length) return null;

  const allCounts = data.map((d) => d.count).sort((a, b) => a - b);
  // 用2%~98%分位数做domain而不是绝对min/max——避免个别扎堆格网的极端计数
  // 把其余大多数格网的高度差压得看不出来（跟2D地图颜色映射用的分位数思路一致）。
  const pct = (p) => allCounts[Math.min(allCounts.length - 1, Math.floor(p * (allCounts.length - 1)))];
  const domainMin = pct(0.02);
  const domainMax = pct(0.98);

  return new deck.ColumnLayer({
    id: "weibo-3d",
    data,
    diskResolution: 6,
    radius: 100,
    extruded: true,
    flatShading: true,
    material: { ambient: 0.35, diffuse: 0.6, shininess: 32, specularColor: [255, 255, 255] },
    getPosition: (d) => d.position,
    getElevation: (d) => scaleElevation(d.count, domainMin, domainMax),
    getFillColor: (d) => {
      const t = (d.count - domainMin) / (domainMax - domainMin || 1);
      return plasmaColor(t);
    },
    pickable: true,
    onClick: (info) => handleWeibo3DBarClick(info),
  });
}

async function handleWeibo3DBarClick(info) {
  if (!info.object || !maplibreMap) return;
  const { gridId, count } = info.object;
  const popup = new maplibregl.Popup({ closeButton: true, maxWidth: "300px" })
    .setLngLat(info.coordinate)
    .setHTML('<div class="grid-popup">加载中...</div>')
    .addTo(maplibreMap);
  try {
    const res = await fetch(`/api/weibo/grid/${gridId}`);
    const data = await res.json();
    const html = res.ok
      ? `<div class="grid-popup"><h3>格网 #${gridId}</h3><div class="hint">该柱统计的POI数：${count}</div>${buildActivityHtml(data)}</div>`
      : `<div class="grid-popup">加载失败：${escapeHtml(data.detail || "请稍后再试")}</div>`;
    popup.setHTML(html);
  } catch {
    popup.setHTML('<div class="grid-popup">加载失败，请稍后再试。</div>');
  }
}

function updateDeck3DLayers() {
  if (!deckOverlay) return;
  const layers = [];
  if (vitality3DOn && geojsonData) layers.push(buildVitalityColumnLayer());
  if (weibo3DOn) {
    const weiboLayer = buildWeiboColumnLayer();
    if (weiboLayer) layers.push(weiboLayer);
  }
  deckOverlay.setProps({ layers });
}

// 3D视图默认机位，reset按钮和初始化都用这份，保持一致。
const MAP3D_DEFAULT_VIEW = { center: [114.28, 30.58], zoom: 11.5, pitch: 50, bearing: -20 };

function ensureDeckInstance() {
  if (deckOverlay) return;
  // deck.gl自身不带底图渲染能力（官方文档明确说standalone的Deck类不处理底图），
  // 之前直接给deck.DeckGL传mapStyle却没加载maplibre-gl.js，导致canvas量不出
  // 尺寸卡在默认300x150——踩过这个坑，正确做法是显式建一个MapLibre地图实例，
  // 再用deck.gl官方支持的MapboxOverlay（兼容MapLibre）把3D图层叠加上去。
  // MapLibre默认就支持左键拖拽平移、右键(或Ctrl+左键)拖拽旋转俯仰、滚轮缩放，
  // 不用额外写交互逻辑；只是这些手势对普通用户不够直观，所以另外加了可见的
  // 罗盘控件+复位视角按钮+操作提示文字，让"全方位查看"这件事显性化。
  maplibreMap = new maplibregl.Map({
    container: "map3dContainer",
    style: "https://basemaps.cartocdn.com/gl/dark-matter-nolabels-gl-style/style.json",
    ...MAP3D_DEFAULT_VIEW,
  });
  maplibreMap.addControl(new maplibregl.NavigationControl({ visualizePitch: true }), "top-right");
  // 方向光+环境光：柱子之间靠明暗对比强化高度差的直观感受，纯顶光下拉伸再高
  // 的柱子看起来也会显得"扁"。
  const lightingEffect = new deck.LightingEffect({
    ambientLight: new deck.AmbientLight({ color: [255, 255, 255], intensity: 1.0 }),
    directionalLight: new deck.DirectionalLight({
      color: [255, 255, 255],
      intensity: 2.5,
      direction: [-2, -3, -1],
    }),
  });
  deckOverlay = new deck.MapboxOverlay({ interleaved: true, layers: [], effects: [lightingEffect] });
  maplibreMap.addControl(deckOverlay);
}

function updateMap3DVisibility() {
  const anyOn = vitality3DOn || weibo3DOn;
  const mapEl = document.getElementById("mapContainer");
  const map3dEl = document.getElementById("map3dContainer");
  if (anyOn) {
    // 容器必须先变可见拿到真实宽高，MapLibre才能测出正确尺寸——顺序反了的话
    // canvas会卡在初始化时的默认尺寸（父容器display:none时量不出尺寸）。
    mapEl.classList.add("hidden");
    map3dEl.classList.remove("hidden");
    ensureDeckInstance();
    if (maplibreMap) maplibreMap.resize(); // 容器从display:none变可见后手动触发一次重新量尺寸
    updateDeck3DLayers();
  } else {
    map3dEl.classList.add("hidden");
    mapEl.classList.remove("hidden");
    map.invalidateSize(); // Leaflet隐藏期间尺寸缓存失效，切回来要手动触发重新计算
  }
}

function toggleFill(hidden) {
  fillHidden = hidden;
  markerByGridId.forEach((m) => m.setStyle({ color: m.options.fillColor, ...currentStyle() }));
}

function toggleBorder(hidden) {
  borderHidden = hidden;
  markerByGridId.forEach((m) => m.setStyle({ color: m.options.fillColor, ...currentStyle() }));
}

// 只画研究区域的整体轮廓（单个多边形描边），不是1万+个格网各自的边框——两种
// 是完全不同的图层，开启时把逐格网图层整个从地图上摘掉，避免叠在一起分不清。
async function toggleStudyAreaBoundary(show) {
  if (show) {
    if (markerLayer) map.removeLayer(markerLayer);
    if (!studyAreaBoundaryLayer) {
      if (!studyAreaBoundaryData) {
        const res = await fetch("/api/study_area_boundary");
        studyAreaBoundaryData = await res.json();
      }
      studyAreaBoundaryLayer = L.geoJSON(studyAreaBoundaryData, {
        style: { color: "#d6604d", weight: 4, fillOpacity: 0, dashArray: "8 6" },
      });
    }
    studyAreaBoundaryLayer.addTo(map);
  } else {
    if (studyAreaBoundaryLayer) map.removeLayer(studyAreaBoundaryLayer);
    if (markerLayer) markerLayer.addTo(map);
  }
}

function highlightGridIds(gridIds, flyToFirst = true) {
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
  if (firstMarker && flyToFirst) {
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
    if (vitality3DOn) updateDeck3DLayers();
  });
}

function todayDateStr() {
  const d = new Date();
  const yyyy = d.getFullYear();
  const mm = String(d.getMonth() + 1).padStart(2, "0");
  const dd = String(d.getDate()).padStart(2, "0");
  return `${yyyy}-${mm}-${dd}`;
}

// 按查询日期的工作日/休息日属性，自动把时段选择器切到对应的组（保留原来选的
// 时段档位，如深夜/日间/夜间；工作日和休息日各自独有的档位如"早高峰"vs"早间"
// 没有直接对应，找不到就退回该组第一档）。
function switchPeriodPrefix(isWorkday) {
  const targetPrefix = isWorkday ? "工作日" : "休息日";
  const currentPrefix = currentPeriod.startsWith("工作日") ? "工作日" : "休息日";
  if (currentPrefix === targetPrefix) return;
  const suffix = currentPeriod.slice(currentPrefix.length);
  const candidate = targetPrefix + suffix;
  currentPeriod = LABEL_COLS.includes(candidate) ? candidate : LABEL_COLS.find((c) => c.startsWith(targetPrefix));
  document.getElementById("periodSelect").value = currentPeriod;
  renderMarkers();
  if (vitality3DOn) updateDeck3DLayers();
}

async function queryCalendarAndWeather() {
  const dateStr = document.getElementById("calendarDateInput").value;
  if (!dateStr) return;
  const resultEl = document.getElementById("calendarResult");
  resultEl.textContent = "查询中...";

  let dayTypeHtml = "";
  let weatherHtml = "";

  try {
    const res = await fetch(`/api/calendar/day_type?date_str=${dateStr}`);
    const data = await res.json();
    if (res.ok) {
      const cls = data.is_workday ? "workday" : "restday";
      dayTypeHtml = `<span class="calendar-badge ${cls}">${data.label}</span>` +
        (data.note ? `<div class="calendar-note">${escapeHtml(data.note)}</div>` : "");
      switchPeriodPrefix(data.is_workday);
    } else {
      dayTypeHtml = `<div class="calendar-note">${escapeHtml(data.detail || "查询失败")}</div>`;
    }
  } catch {
    dayTypeHtml = '<div class="calendar-note">日期类型查询失败，请稍后再试。</div>';
  }

  try {
    const res = await fetch(`/api/weather?date_str=${dateStr}`);
    const data = await res.json();
    if (res.ok) {
      weatherHtml =
        `<div class="calendar-weather">白天${escapeHtml(data.text_day)} · 夜间${escapeHtml(data.text_night)} · ` +
        `${data.temp_min}~${data.temp_max}℃</div>`;
    } else {
      weatherHtml = `<div class="calendar-note">${escapeHtml(data.detail || "天气查询失败")}</div>`;
    }
  } catch {
    weatherHtml = '<div class="calendar-note">天气查询失败，请稍后再试。</div>';
  }

  resultEl.innerHTML = dayTypeHtml + weatherHtml;
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

function poiPopupHtml(post) {
  const placeType = escapeHtml(post.place_type || "未知类型");
  return `<div class="grid-popup"><b>格网 #${post.grid_id}</b>（${placeType}） · ❤${post.like_count}
    <button class="weibo-activity-btn" data-grid-id="${post.grid_id}">查看该地区微博动态</button>
    <div class="activity-container" id="activity-container-${post.grid_id}"></div>
  </div>`;
}

function populatePoiCategorySelect(posts) {
  const counts = new Map();
  for (const p of posts) {
    const key = p.place_type || "（未知类型）";
    // 极少数帖子的place_type是未映射到中文名的原始数字编码（如"88"“247”），
    // 只有几十条，属于上游数据脏值，不作为可选类别展示（不影响这些帖子本身
    // 在地图上显示，只是筛选下拉框里不单独列出这个噪声类别）。
    if (/^\d+$/.test(key)) continue;
    counts.set(key, (counts.get(key) || 0) + 1);
  }
  const sorted = [...counts.entries()].sort((a, b) => b[1] - a[1]);
  const sel = document.getElementById("poiCategorySelect");
  sel.innerHTML = "";
  for (const [name, count] of sorted) {
    const opt = document.createElement("option");
    opt.value = name;
    opt.textContent = `${name} (${count})`;
    sel.appendChild(opt);
  }
}

function getSelectedPoiCategories() {
  const sel = document.getElementById("poiCategorySelect");
  return [...sel.selectedOptions].map((o) => o.value);
}

function getFilteredAllPois() {
  if (!allPoiData) return [];
  const selectedCategories = getSelectedPoiCategories();
  const minLikes = Number(document.getElementById("poiLikeThresholdInput").value) || 0;
  return allPoiData.posts.filter((post) => {
    if (post.like_count < minLikes) return false;
    const category = post.place_type || "（未知类型）";
    if (selectedCategories.length && !selectedCategories.includes(category)) return false;
    return true;
  });
}

function renderAllPoiLayer() {
  if (!allPoiData) return;
  if (allPoiClusterLayer) map.removeLayer(allPoiClusterLayer);

  const filtered = getFilteredAllPois();
  allPoiClusterLayer = L.markerClusterGroup({ chunkedLoading: true });
  for (const post of filtered) {
    const marker = L.circleMarker([post.lat, post.lng], {
      radius: 4,
      color: "#ffffff",
      weight: 1,
      fillColor: "#4393c3",
      fillOpacity: 0.85,
    });
    marker.bindPopup(() => poiPopupHtml(post));
    allPoiClusterLayer.addLayer(marker);
  }
  allPoiClusterLayer.addTo(map);

  document.getElementById("allPoiInfo").textContent =
    `共 ${allPoiData.count} 条POI，当前筛选展示 ${filtered.length} 条`;

  if (weibo3DOn) updateDeck3DLayers();
}

async function loadAllPois() {
  document.getElementById("allPoiInfo").textContent = "加载中（首次加载约9MB，请稍候）...";
  try {
    const res = await fetch("/api/weibo/all_pois");
    const data = await res.json();
    if (!res.ok) {
      document.getElementById("allPoiInfo").textContent = data.detail || "加载失败，请稍后再试。";
      return;
    }
    allPoiData = data;
    populatePoiCategorySelect(data.posts);

    let maxLikes = 0;
    for (const p of data.posts) if (p.like_count > maxLikes) maxLikes = p.like_count;
    const slider = document.getElementById("poiLikeThresholdInput");
    slider.max = maxLikes;
    slider.value = 0;
    document.getElementById("poiLikeThresholdValue").textContent = "0";

    renderAllPoiLayer();
  } catch {
    document.getElementById("allPoiInfo").textContent = "加载失败，请稍后再试。";
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

// 问答框合并前，"快问快答"(/api/chat)和"智能推荐"(/api/chat/agent)是两个独立
// 输入框，用户分不清该用哪个。现在统一成一个/api/chat端点+一个输入框，后端按
// 问题内容判断走哪条内部路径，前端不关心走的是哪条——都当同一套SSE事件流处理：
// 简单问题只收到一个"final"事件（跟原来agent路径的最终事件同格式），复杂问题
// 会先收到若干"tool_call"/"tool_result"中间步骤事件，渲染成聊天流里的一条
// 步骤气泡，再收到"final"事件。
function handleChatEvent(event, stepsBox) {
  const box = document.getElementById("chatMessages");
  if (event.type === "tool_call") {
    stepsBox.classList.remove("hidden");
    const item = document.createElement("div");
    item.className = "agent-step";
    item.dataset.tool = event.tool;
    item.textContent = `🔍 正在${event.label}...`;
    stepsBox.appendChild(item);
    box.scrollTop = box.scrollHeight;
  } else if (event.type === "tool_result") {
    const steps = stepsBox.querySelectorAll(`.agent-step[data-tool="${event.tool}"]`);
    const item = steps[steps.length - 1];
    if (item) {
      item.textContent = `✅ ${event.label}完成`;
      item.classList.add("done");
    }
  } else if (event.type === "final") {
    if (stepsBox.classList.contains("hidden")) stepsBox.remove();
    appendMessage("bot", event.answer);
    const gridIds = event.highlight_grid_ids || [];
    const hasGrids = gridIds.length > 0;
    const hasRoute = (event.markers && event.markers.length) || (event.polylines && event.polylines.length);

    // 两种信号都有时，谁都不单独飞镜头，最后统一算一个同时框住两者的视野；
    // 只有一种信号时，各自按原来的方式飞镜头（不改变已有的单一场景行为）。
    if (hasGrids) highlightGridIds(gridIds, !hasRoute);
    renderAgentRoute(event.markers, event.polylines, !hasGrids);
    if (hasGrids && hasRoute) {
      fitBoundsToGridsAndRoute(gridIds, event.markers, event.polylines);
    }
  }
}

async function sendChat() {
  const input = document.getElementById("chatInput");
  const question = input.value.trim();
  if (!question) return;
  appendMessage("user", question);
  input.value = "";

  const box = document.getElementById("chatMessages");
  const sendBtn = document.getElementById("chatSendBtn");
  sendBtn.disabled = true;
  sendBtn.textContent = "思考中...";
  if (agentRouteLayer) {
    map.removeLayer(agentRouteLayer);
    agentRouteLayer = null;
  }

  // 中间步骤气泡先建好但保持隐藏——只有走多步骤路径时后端才会发tool_call事件，
  // 那时才显示出来；简单问题走完全程这个气泡都是空的，最终答案来了直接移除。
  const stepsBox = document.createElement("div");
  stepsBox.className = "msg bot agent-steps-msg hidden";
  box.appendChild(stepsBox);

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) {
      const data = await res.json().catch(() => ({}));
      stepsBox.remove();
      appendMessage("bot", data.detail || "请求失败，请稍后再试。");
      return;
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";
    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });
      const parts = buffer.split("\n\n");
      buffer = parts.pop(); // 最后一段可能不完整（还没收全一个事件），留到下次拼接
      for (const part of parts) {
        if (!part.startsWith("data: ")) continue;
        handleChatEvent(JSON.parse(part.slice(6)), stepsBox);
      }
    }
  } catch {
    stepsBox.remove();
    appendMessage("bot", "请求失败，请稍后再试。");
  } finally {
    sendBtn.disabled = false;
    sendBtn.textContent = "发送";
    box.scrollTop = box.scrollHeight;
  }
}

let agentRouteLayer = null;

// agent规划路线时，把geocode查到的点（编号标记）和route_between查到的真实道路
// 坐标串（不是直线近似）画在地图上。markers的编号顺序是工具调用顺序（geocode
// 一般在plan_route_order排完最终顺序之前调用），不严格等于最终访问顺序，
// 但足够让用户看清"路线大致覆盖了哪些点"。
function renderAgentRoute(markers, polylines, autoFit = true) {
  if (agentRouteLayer) map.removeLayer(agentRouteLayer);
  if ((!markers || !markers.length) && (!polylines || !polylines.length)) return;

  agentRouteLayer = L.layerGroup();
  for (const line of polylines || []) {
    const latlngs = line.map(([lng, lat]) => [lat, lng]);
    L.polyline(latlngs, { color: "#4393c3", weight: 4, opacity: 0.85 }).addTo(agentRouteLayer);
  }

  const bounds = [];
  (markers || []).forEach((m, i) => {
    const marker = L.marker([m.lat, m.lng], {
      icon: L.divIcon({
        className: "agent-route-marker",
        html: `<div class="agent-route-marker-badge">${i + 1}</div>`,
        iconSize: [24, 24],
        iconAnchor: [12, 12],
      }),
    });
    marker.bindPopup(escapeHtml(m.name || ""));
    marker.addTo(agentRouteLayer);
    bounds.push([m.lat, m.lng]);
  });

  agentRouteLayer.addTo(map);
  if (autoFit && bounds.length) {
    map.invalidateSize();
    map.fitBounds(bounds, { padding: [40, 40] });
  }
}

// 高亮格网和路线标记/连线都会争抢"地图该看向哪"这个镜头控制权，两种信号同时
// 出现时（比如"规划下周末游玩路线"这种问法，既查了活力数据又geocode了地点），
// 不能各自飞一次镜头互相打架——后调用的会覆盖先调用的，先高亮的格网还没看清
// 镜头就被路线标记抢走了。统一算一个能同时框住格网+路线的视野，只飞一次镜头。
function fitBoundsToGridsAndRoute(gridIds, markers, polylines) {
  const bounds = L.latLngBounds([]);
  for (const gid of gridIds || []) {
    const pos = gridIdToLatLng.get(gid); // [lng, lat]
    if (pos) bounds.extend([pos[1], pos[0]]);
  }
  for (const m of markers || []) bounds.extend([m.lat, m.lng]);
  for (const line of polylines || []) {
    for (const [lng, lat] of line) bounds.extend([lat, lng]);
  }
  if (bounds.isValid()) {
    map.invalidateSize(); // 防止Leaflet缓存的容器尺寸过期导致fitBounds算出的缩放级别不对
    map.fitBounds(bounds, { padding: [40, 40] });
  }
}

async function main() {
  initPeriodSelect();
  await initDistrictList();

  const res = await fetch("/api/geojson");
  geojsonData = await res.json();
  computeGlobalColorRange();
  renderMarkers();
  for (const feat of geojsonData.features) {
    gridIdToLatLng.set(feat.properties.grid_id, feat.geometry.coordinates);
  }

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
  document.getElementById("allPoiToggle").addEventListener("change", async (e) => {
    const controls = document.getElementById("allPoiControls");
    if (e.target.checked) {
      controls.classList.remove("hidden");
      if (allPoiData) {
        renderAllPoiLayer();
      } else {
        await loadAllPois();
      }
    } else {
      controls.classList.add("hidden");
      if (allPoiClusterLayer) {
        map.removeLayer(allPoiClusterLayer);
        allPoiClusterLayer = null;
      }
    }
  });
  document.getElementById("poiCategorySelect").addEventListener("change", renderAllPoiLayer);
  document.getElementById("poiLikeThresholdInput").addEventListener("input", (e) => {
    document.getElementById("poiLikeThresholdValue").textContent = e.target.value;
    renderAllPoiLayer();
  });
  document.getElementById("calendarDateInput").value = todayDateStr();
  document.getElementById("calendarDateInput").addEventListener("change", queryCalendarAndWeather);
  queryCalendarAndWeather(); // 页面加载时默认查一次今天，不用等用户手动改日期
  document.getElementById("vitality3DToggle").addEventListener("change", (e) => {
    vitality3DOn = e.target.checked;
    // 两个3D柱状图不共存——同时叠两层柱子在同一批格网上会互相遮挡，看不清楚
    // 到底哪根柱子代表什么，开一个自动关掉另一个。
    if (vitality3DOn && weibo3DOn) {
      weibo3DOn = false;
      document.getElementById("weibo3DToggle").checked = false;
    }
    updateMap3DVisibility();
  });
  document.getElementById("studyAreaBoundaryToggle").addEventListener("change", (e) => {
    toggleStudyAreaBoundary(e.target.checked);
  });
  document.getElementById("weibo3DToggle").addEventListener("change", async (e) => {
    weibo3DOn = e.target.checked;
    if (weibo3DOn && vitality3DOn) {
      vitality3DOn = false;
      document.getElementById("vitality3DToggle").checked = false;
    }
    // 3D柱状图数据源是全部POI，开关打开时如果还没拉取过全量数据就顺便拉一次，
    // 不用非要先手动开"显示全部微博POI"那个开关才能看3D。
    if (weibo3DOn && !allPoiData) await loadAllPois();
    updateMap3DVisibility();
  });
  document.getElementById("reset3DViewBtn").addEventListener("click", () => {
    if (maplibreMap) maplibreMap.easeTo({ ...MAP3D_DEFAULT_VIEW, duration: 600 });
  });
  // 面板折叠后，里面还开着的地图图层类开关（3D柱状图、全部POI聚合展示）要跟着
  // 自动关掉——不然面板收起来了，地图上的东西还留着，用户也没法在侧边栏看到
  // 状态、没法关。复用checkbox已有的change监听逻辑（模拟一次用户手动取消勾选），
  // 不用另外写一份清理逻辑。
  function autoOffOnCollapse(detailsId, checkboxIds) {
    const details = document.getElementById(detailsId);
    details.addEventListener("toggle", () => {
      if (details.open) return;
      for (const id of checkboxIds) {
        const cb = document.getElementById(id);
        if (cb.checked) {
          cb.checked = false;
          cb.dispatchEvent(new Event("change"));
        }
      }
    });
  }
  autoOffOnCollapse("vitalityPanel", ["vitality3DToggle", "studyAreaBoundaryToggle"]);
  autoOffOnCollapse("allPoiPanel", ["allPoiToggle", "weibo3DToggle"]);

  document.getElementById("chatSendBtn").addEventListener("click", sendChat);
  document.getElementById("chatInput").addEventListener("keydown", (e) => {
    if (e.key === "Enter") sendChat();
  });

  appendMessage("bot", "你好，我是活力地图问答助手，已接入DeepSeek。可以问我“武昌区晚上活力怎么样”这类问题，也可以问“这个周末去哪玩”这类需要规划路线的开放性问题，也可以在上面的微博搜索框里搜关键词看真实网友动态。");
}

main();
