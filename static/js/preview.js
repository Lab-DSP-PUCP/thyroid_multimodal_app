// Leer diccionarios embebidos
function readJsonScript(id){
  const el=document.getElementById(id);
  if(!el) return {};
  try{ return JSON.parse(el.textContent||'{}'); }catch{ return {}; }
}
const UI  = readJsonScript('ui-json');
const VAL = readJsonScript('val-json');

// Helpers de formato
const fmt = (k,v)=> (k in VAL && v in VAL[k]) ? VAL[k][v] : (v ?? "-");
const titleize = s => (s||'').replaceAll('_',' ').replace(/^\w/,c=>c.toUpperCase());

// Estado global/explain 
let currentExplainFile = null;
let currentClinPayload = null;
let currentImageURL = null;

// refs
const pane = () => document.getElementById('preview-pane');
const slot = () => document.getElementById('preview-content');

// Funcion para limpiar/resetear la vista de Grad-CAM
function resetGradcamView() {
  const gradcamView = document.getElementById('gradcam-view');
  if (gradcamView) {
    gradcamView.src = '';
    gradcamView.classList.add('d-none');
  }
  // Tambien es buena idea deshabilitar el botón "Explicar" hasta tener un nuevo contexto
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
  // Devuelve (elemento <img> unico donde se pintara el resultado)
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

// Render de solo la imagen (cuando das clic en "ver")
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
      if (value instanceof Node) v.appendChild(value); else v.textContent=String(value ?? "-");
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

// Selección desde la tabla (ver y xml)
document.addEventListener('click', async (e)=>{
  const a = e.target.closest('a.preview-link');
  if(!a) return;
  e.preventDefault();

  wipePreviewImages();

  // Oculta la vista de la imagen original anterior.
  const orig = document.getElementById('original-view');
  if (orig) orig.classList.add('d-none');

  // Habilita el boton "Explicar" solo si se hace clic en una imagen.
  const btnExplain = document.getElementById('btn-explain');
  if (btnExplain) {
     const type = a.getAttribute('data-type');
     btnExplain.disabled = (type !== 'image');
  }

  const type = a.getAttribute('data-type'); // 'image' | 'xml'
  const url  = a.getAttribute('data-file');
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
    currentImageURL = url; // guarda la URL para Grad-CAM
    const orig = document.getElementById('original-view');
    if (orig) {
      orig.src = url;
      orig.alt = 'Ecografía original (sin Grad-CAM)';
      orig.classList.remove('d-none');
    }
  }
  if (type === 'xml')     await renderXmlOnly(url, item);
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
  initGradcamControls();
  initThemeToggle();
  initMetadataToggle();
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
  // 3) último recurso: imagen original (por id)
  if (!targetImg) targetImg = document.getElementById('original-view');

  const src = targetImg?.getAttribute('src');
  if (src && window.FSViewer && typeof window.FSViewer.open === 'function') {
    window.FSViewer.open(src);
  }
});

// ======= Tutorial video: autoplay + overlay =======
(() => {
  const video   = document.getElementById('tutorial-video');
  const overlay = document.querySelector('.video-overlay-play');
  const modal   = document.getElementById('tutorialModal');
  if (!video || !overlay || !modal) return;

  // Muestra/oculta overlay según estado del video
  const syncOverlay = () => {
    if (video.paused || video.ended) {
      overlay.classList.remove('hidden');
    } else {
      overlay.classList.add('hidden');
    }
  };

  // Al abrir el modal: intenta reproducir (requiere muted, ya está en el HTML)
  modal.addEventListener('shown.bs.modal', async () => {
    try {
      video.currentTime = 0;
      await video.play();
    } catch (e) {
      // si el navegador bloquea, el usuario verá el overlay y podrá clickear
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
