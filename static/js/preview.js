// Leer diccionarios embebidos
function readJsonScript(id){
  const el=document.getElementById(id);
  if(!el) return {};
  try{ return JSON.parse(el.textContent||'{}'); }catch{ return {}; }
}

const UI  = readJsonScript('ui-json');
const VAL = readJsonScript('val-json');

// Helpers de formato
const norm = (s)=> String(s??'').trim()
  .toLowerCase()
  .normalize('NFD').replace(/[\u0300-\u036f]/g,'')
  .replace(/\s+/g,'_');

function toTitle(s){ return String(s||'').replace(/^\p{L}/u, c=>c.toUpperCase()); }

// Alias (clave XML -> clave estandar del diccionario)
const ALIAS = {
  echogenicity: {
    isoechogenicity: 'isoechoic',
    isoechogenic:    'isoechoic',
    hypoechogenicity:'hypoechoic',
    hyperechogenicity:'hyperechoic',
  },
  calcifications: {
    microcalcifications: 'microcalcification',
    macrocalcifications: 'macrocalcification',
    peripheral: 'rim',
    peripheral_calcifications: 'rim',
  },
  margins: {
    well_defined: 'smooth',
    ill_defined:  'ill_defined',
  }
};

// Traduccion de respaldo directa
const DIRECT = {
  composition: {
    spongiform: 'Espongiforme',
    solid:      'Sólido',
    mixed:      'Mixto',
    cystic:     'Quístico',
  },
  echogenicity: {
    isoechogenicity:   'Isoecoico',
    isoechogenic:      'Isoecoico',
    isoechoic:         'Isoecoico',
    hypoechogenicity:  'Hipoecoico',
    hypoechoic:        'Hipoecoico',
    hyperechogenicity: 'Hiperecoico',
    hyperechoic:       'Hiperecoico',
    anechoic:          'Anecoico',
  },
  margins: {
    well_defined: 'Bien definidos',
    smooth:       'Lisos',
    ill_defined:  'Mal definidos',
    lobulated:    'Lobulados',
    irregular:    'Irregulares',
  },
  calcifications: {
    microcalcifications: 'Microcalcificaciones',
    microcalcification:  'Microcalcificaciones',
    macrocalcifications: 'Macrocalcificaciones',
    macrocalcification:  'Macrocalcificaciones',
    peripheral:          'Periféricas',
    rim:                 'Periféricas',
    none:                'Ninguna',
  },
  sex: {
    f: 'Femenino',
    m: 'Masculino',
  }
};

// Busca en VAL, si no hay, usa alias. Si no, DIRECT, si no, deja tal cual.
function fmt(k, v){
  const raw = String(v ?? '').trim();
  const vLower = raw.toLowerCase();
  const dict = (VAL && VAL[k]) ? VAL[k] : null;

  // match directo en VAL
  if (dict && raw in dict) return dict[raw];

  // alias -> VAL
  const aliasMap = ALIAS[k] || {};
  if (aliasMap[vLower] && dict && aliasMap[vLower] in dict) {
    return dict[ aliasMap[vLower] ];
  }

  // normalizados -> VAL
  if (dict){
    const candidates = [ norm(raw), vLower.replace(/\s+/g,'_') ];
    for(const c of candidates){ if (c in dict) return dict[c]; }
  }

  // respaldo directo (ES) por categoria
  if (DIRECT[k] && (vLower in DIRECT[k])) return DIRECT[k][vLower];

  // ultimo recurso
  return toTitle(raw || '-').replace(/_/g,' ');
}

const titleize = s => (s||'').replaceAll('_',' ').replace(/^\w/,c=>c.toUpperCase());

function toTitle(s){ return String(s||'').replace(/^\p{L}/u, c=>c.toUpperCase()); }

// Estado global
let currentExplainFile = null;
let currentClinPayload = null;
let currentImageURL = null;
let defaultHintHTML = "";
let lastPreview = { type: null, url: null };

// refs
const pane = () => document.getElementById('preview-pane');
const slot = () => document.getElementById('preview-content');

// Oculta TODO del panel (imagen CAM, grid XML y original)
function collapsePreview() {
  wipePreviewImages(); // limpia el slot superior (CAM / XML)
  const orig = document.getElementById('original-view');
  if (orig) orig.classList.add('d-none'); // oculta la imagen original
  toggleCamControls(false);  // oculta botones CAM
  lastPreview = { type: null, url: null }; // resetea estado
}

// Funcion para limpiar/resetear la vista de Grad-CAM
function resetGradcamView() {
  const gradcamView = document.getElementById('gradcam-view');
  if (gradcamView) {
    gradcamView.src = '';
    gradcamView.classList.add('d-none');
  }
  const btnExplain = document.getElementById('btn-explain');
  if(btnExplain) btnExplain.disabled = true;
}

// Bloqueo de altura (evita saltos)
let panePrevHeight = null;
function lockPaneHeight(){
  const p = pane();
  if (!p) return;
  panePrevHeight = p.offsetHeight;
  const h = Math.max(panePrevHeight, 260);
  p.style.height = h + 'px';
}

function unlockPaneHeight(){
  const p = pane();
  if (!p) return;
  p.style.height = '';
  panePrevHeight = null;
}

// Limpieza estricta (evita duplicados)
function wipePreviewImages(){
  const s = slot();
  if (!s) return;
  // Elimina unicamente lo que se dibuja dentro del slot (overlay CAM, spinner, etc.)
  s.querySelectorAll('img, .cam-result').forEach(n => n.remove());
  s.innerHTML = '';
}

function ensureCamOutput() {
  // Devuelve elemento <img> unico donde se pintara el resultado
  const s = slot();
  let out = s.querySelector('#cam-output');
  if (!out) {
    out = document.createElement('img');
    out.id = 'cam-output';
    out.className = 'img-fluid rounded cam-result';
    s.appendChild(out);
  }
  return out;
}

// Encabezado/etiquetas (para XML)
function headerBadge(container, item){
  const wrap=document.createElement('div');
  const badgeClass=(String(item.label).toLowerCase()==='maligno')?'badge-soft-danger':'badge-soft-success';
  wrap.innerHTML=`
    <div class="d-flex align-items-center gap-2 mb-2">
      <span class="${badgeClass}">${item.label ?? '-'}</span>
      ${Number.isFinite(Number(item.prob)) ? `<span class="text-muted">Prob: ${Number(item.prob).toFixed(2)}</span>` : ''}
      ${item.patient_id && item.patient_id!=='-' ? `<span class="ms-auto"><strong>Paciente:</strong> ${item.patient_id}</span>` : ''}
    </div>`;
  container.appendChild(wrap);
}

// Render de solo la imagen (cuando se da clic en "ver")
function renderImageOnly(url){
  wipePreviewImages();
  const out = ensureCamOutput();
  out.alt = 'Imagen';
  out.src = url;
}

// Render XML amigable visualmente
function parseXmlToSections(xmlText){
  const pairs=[], extras=[];
  try{
    const parser=new DOMParser();
    const xmlDoc=parser.parseFromString(xmlText,'application/xml');
    if(xmlDoc.getElementsByTagName('parsererror').length) throw new Error('XML');

    const caseNode=xmlDoc.getElementsByTagName('case')[0] || xmlDoc.documentElement;
    Array.from(caseNode.children).forEach(node=>{
      const key=node.tagName;
      let raw=(node.textContent||'').trim();
      const label = UI[key] || titleize(key);

      const isHeavy = /^(mark|svg)$/i.test(key) || /"points"\s*:/.test(raw) || /^\s*\d+\s*\[/.test(raw);
      if (isHeavy){
        const clean = raw.replace(/^\s*\d+\s*(?=\[)/,'');
        let count = 0, pretty = clean;
        try{
          const obj = JSON.parse(clean);
          if (Array.isArray(obj)) obj.forEach(o=>{ if(o && Array.isArray(o.points)) count += o.points.length; });
          else if (obj && Array.isArray(obj.points)) count = obj.points.length;
          pretty = JSON.stringify(obj,null,2);
        }catch{ pretty = (raw.length>200? raw.slice(0,200)+'…' : raw); }

        pairs.push([label, `${count>0? count+' puntos' : 'Detalle'}`]);

        const details=document.createElement('details');
        details.className='preview-details';
        const summary=document.createElement('summary'); summary.textContent='Ver detalle de puntos';
        const pre=document.createElement('pre'); pre.className='preview-pre'; pre.textContent=pretty;
        details.appendChild(summary); details.appendChild(pre);
        extras.push(details);
        return;
      }

      const keyLower = key.toLowerCase();
      let value = raw || '-';
      if (['composition','echogenicity','margins','calcifications','sex'].includes(keyLower)) {
        value = fmt(keyLower, raw);
      }
      pairs.push([label, value]);
    });
  }catch{
    const pre=document.createElement('pre'); pre.className='preview-pre'; pre.textContent=xmlText.replace(/></g,'>\n<');
    pairs.push(['XML', pre]);
  }
  return {pairs, extras};
}

async function renderXmlOnly(url, item){
  wipePreviewImages();
  const s = slot();
  headerBadge(s, item);
  try{
    const res = await fetch(url);
    const text = await res.text();
    const {pairs, extras} = parseXmlToSections(text);
    const grid=document.createElement('div'); grid.className='preview-grid';
    pairs.forEach(([label,value])=>{
      const l=document.createElement('div'); l.className='label'; l.textContent=label;
      const v=document.createElement('div'); v.className='value';
      if (value instanceof Node) {
        v.appendChild(value);
      } else {
        const txt = String(value ?? "-");
        v.textContent = toTitle(txt);
        if (txt === "-") v.classList.add('text-muted','fst-italic');
      }
      grid.appendChild(l); grid.appendChild(v);
    });
    s.appendChild(grid); extras.forEach(el=>s.appendChild(el));
  }catch{
    const err=document.createElement('div');
    err.className='text-danger mt-2';
    err.textContent='No se pudo cargar o parsear el XML.';
    s.appendChild(err);
  }
}

function toggleCamControls(show){
  const btnExplain = document.getElementById('btn-explain');
  const btnFull    = document.getElementById('btn-fullscreen');
  // contenedor del grupo de radios "Modelo / Completa"
  const camGroup   = document.querySelector('[aria-label="Vista Grad-CAM"]');

  [btnExplain, btnFull, camGroup].forEach(el=>{
    if(!el) return;
    el.classList.toggle('d-none', !show);
  });
  if(btnExplain) btnExplain.disabled = !show;
}

// Seleccion desde la tabla (ver y xml)
document.addEventListener('click', async (e)=>{
  const a = e.target.closest('a.preview-link');
  if(!a) return;
  e.preventDefault();

  const type = a.getAttribute('data-type');
  const url  = a.getAttribute('data-file');

  // TOGGLE: si es el mismo (tipo+url), colapsa
  if (lastPreview.type === type && lastPreview.url === url) {
    collapsePreview();
    return;
  }
  lastPreview = { type, url };

  // Limpia el slot superior SIEMPRE
  wipePreviewImages();

  // Oculta la imagen original previa
  const orig = document.getElementById('original-view');
  if (orig) orig.classList.add('d-none');

  // Manten contexto para Grad-CAM
  const item = JSON.parse(a.getAttribute('data-json')||'{}');
  currentExplainFile = item?.file || null;
  currentImageURL = (type==='image') ? url : currentImageURL;
  currentClinPayload = {
    composition:    item?.composition    ?? "",
    echogenicity:   item?.echogenicity   ?? "",
    margins:        item?.margins        ?? "",
    calcifications: item?.calcifications ?? "",
    sex:            item?.sex            ?? "",
    age:            item?.age            ?? ""
  };

  if (type === 'image') {
    // Mostrar SOLO la imagen original abajo (sin crear otra en el slot)
    toggleCamControls(true);
    if (orig) {
      orig.src = url;
      orig.alt = 'Ecografía original (sin Grad-CAM)';
      orig.classList.remove('d-none');
    }
  } else {
    // XML: oculta controles CAM y renderiza la tabla bonita arriba
    toggleCamControls(false);
    await renderXmlOnly(url, item);
  }
});

// Grad-CAM
async function requestGradcam(){
  if (!currentExplainFile) {
    alert('Primero haz clic en "ver" para seleccionar el caso.');
    return;
  }
  lockPaneHeight();
  wipePreviewImages();

  const s = slot();
  const spinner = document.createElement('div');
  spinner.className = 'd-flex justify-content-center align-items-center w-100 py-4 cam-result';
  spinner.innerHTML = '<div class="spinner-border" role="status" aria-label="Cargando Grad-CAM"></div>';
  s.appendChild(spinner);

  const fd = new FormData();
  fd.append('filename', currentExplainFile);

  const view = document.querySelector('input[name="cam-view"]:checked')?.value || 'crop';
  fd.append('view', view);

  // siempre enviar clinicos del item (si existen)
  const p = currentClinPayload || {};
  fd.append('composition',    p.composition    ?? '');
  fd.append('echogenicity',   p.echogenicity   ?? '');
  fd.append('margins',        p.margins        ?? '');
  fd.append('calcifications', p.calcifications ?? '');
  fd.append('sex',            p.sex            ?? '');
  fd.append('age',            p.age            ?? '');

  fd.append('alpha', '0.35');

  try{
    const res = await fetch('/explain', { method: 'POST', body: fd });
    const data = await res.json();

    wipePreviewImages(); // quita el spinner
    if (data.overlay_png_base64) {
      const out = ensureCamOutput();
      out.alt = 'Grad-CAM';
      out.src = data.overlay_png_base64;
      // Mostrar ecografia original debajo (sin Grad-CAM)
      const orig = document.getElementById('original-view');
      if (orig) {
        const fallback = currentExplainFile ? ('/uploads/' + encodeURIComponent(currentExplainFile)) : null;
        const urlOrig = currentImageURL || fallback;
        if (urlOrig) {
          orig.src = urlOrig;
          orig.classList.remove('d-none');
        }
      }
    } else {
      s.innerHTML = '<div class="text-danger cam-result">No se pudo generar Grad-CAM.</div>';
    }
  }catch{
    wipePreviewImages();
    s.innerHTML = '<div class="text-danger cam-result">Error al solicitar Grad-CAM.</div>';
  }finally{
    unlockPaneHeight();
  }
}

function initGradcamControls(){
  document.getElementById('btn-explain')?.addEventListener('click', requestGradcam);
  document.querySelectorAll('input[name="cam-view"]').forEach(r=>{
    r.addEventListener('change', () => { if (currentExplainFile) requestGradcam(); });
  });
}

function clearPreviewPane() {
  wipePreviewImages();
  // oculta imagen original y controles CAM
  document.getElementById('original-view')?.classList.add('d-none');
  toggleCamControls(false);
  // restaura el hint inicial
  const s = slot();
  if (s) s.innerHTML = defaultHintHTML || '<div class="text-muted">Previsualización cerrada.</div>';
  // limpia estado actual
  const p = pane();
  if (p) { delete p.dataset.viewType; delete p.dataset.viewUrl; }
  currentExplainFile = null;
}

// Tema y toggle de metadatos (solo UI, no CAM)
function initThemeToggle(){
  const btn  = document.getElementById('theme-toggle');
  const root = document.documentElement;

  function apply(theme){
    root.classList.add('theme-animating');
    root.setAttribute('data-bs-theme', theme);
    localStorage.setItem('theme', theme);
    const sun = document.querySelector('.icon-sun');
    const moon = document.querySelector('.icon-moon');
    if (sun && moon) {
      const dark = theme === 'dark';
      sun.style.display  = dark ? 'none' : 'inline';
      moon.style.display = dark ? 'inline' : 'none';
    }
    setTimeout(()=> root.classList.remove('theme-animating'), 600);
  }

  let saved = localStorage.getItem('theme');
  if(!saved) saved = window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light';
  apply(saved);

  btn?.addEventListener('click', () => {
    const next = (root.getAttribute('data-bs-theme') === 'dark') ? 'light' : 'dark';
    apply(next); btn.blur();
  });
}

function initMetadataToggle() {
  const sw = document.getElementById('use-metadata-switch');
  if (!sw) return;
  const section = document.getElementById('metadata-section');
  const controls = Array.from(section?.querySelectorAll('select, input') || []);
  const setEnabled = on => { section?.classList.toggle('d-none', !on); controls.forEach(el => el.disabled = !on); };
  setEnabled(false);
  sw.addEventListener('change', () => { setEnabled(sw.checked); sw.blur(); });
}

// Boot
document.addEventListener('DOMContentLoaded', () => {
  if (!defaultHintHTML) defaultHintHTML = slot()?.innerHTML || "";
  initGradcamControls();
  initThemeToggle();
  initMetadataToggle();
  // Asegurar "Completa" como vista CAM por defecto
  const full = document.getElementById('camViewFull');
  const crop = document.getElementById('camViewCrop');
  if (full) full.checked = true;
  if (crop) crop.checked = false;
});

// Abrir visor a pantalla completa con la imagen visible (CAM u original)
document.addEventListener('click', (e) => {
  const btn = e.target.closest('#btn-fullscreen');
  if (!btn) return;

  const pane = document.getElementById('preview-pane');
  if (!pane) return;

  // 1) ¿hay una imagen CAM (base64) visible en el preview?
  let targetImg = pane.querySelector('img:not(.d-none)');
  // 2) si no, intenta la primera imagen dentro del preview-content
  if (!targetImg) targetImg = pane.querySelector('#preview-content img');
  // 3) ultimo recurso: imagen original (por id)
  if (!targetImg) targetImg = document.getElementById('original-view');

  const src = targetImg?.getAttribute('src');
  if (src && window.FSViewer && typeof window.FSViewer.open === 'function') {
    window.FSViewer.open(src);
  }
});

// Tutorial video: autoplay + overlay
(() => {
  const video   = document.getElementById('tutorial-video');
  const overlay = document.querySelector('.video-overlay-play');
  const modal   = document.getElementById('tutorialModal');
  if (!video || !overlay || !modal) return;

  // Muestra/oculta overlay segun estado del video
  const syncOverlay = () => {
    if (video.paused || video.ended) {
      overlay.classList.remove('hidden');
    } else {
      overlay.classList.add('hidden');
    }
  };

  // Al abrir el modal: intenta reproducir (requiere muted, ya esta en el HTML)
  modal.addEventListener('shown.bs.modal', async () => {
    try {
      video.currentTime = 0;
      await video.play();
    } catch (e) {
      // si el navegador bloquea, el usuario vera el overlay y podra clickear
    } finally {
      syncOverlay();
    }
  });

  // Al cerrar el modal: pausar y resetear estado
  modal.addEventListener('hidden.bs.modal', () => {
    try { video.pause(); } catch (_) {}
    video.currentTime = 0;
    syncOverlay();
  });

  // Click en el overlay: toggle play/pause
  overlay.addEventListener('click', async () => {
    if (video.paused || video.ended) {
      try { await video.play(); } catch (_) {}
    } else {
      video.pause();
    }
    syncOverlay();
  });

  // Eventos del video: mantener overlay correcto
  ['play','playing','pause','ended','waiting','seeking'].forEach(ev => {
    video.addEventListener(ev, syncOverlay);
  });
})();
