document.addEventListener('DOMContentLoaded', () => {
  console.log('[crud.js] DOM listo');

  const qs = (sel, root=document)=>root.querySelector(sel);
  const form = qs('#edit-history-form');
  const modalEl = qs('#editHistoryModal');
  
  if (!form || !modalEl) {
    console.warn('[crud.js] No se encontró el modal de edición; continúo para registro…');
  }
  const modal = modalEl ? new bootstrap.Modal(modalEl) : null;
  // value_labels para mapear etiquetas de la tabla
  const valJsonEl = document.getElementById('val-json');
  const roiJsonEl = document.getElementById('roi-json');

  const uiKeysEl = document.getElementById('ui-keys-json');
  let UI_KEYS = {};
  try { UI_KEYS = JSON.parse(uiKeysEl?.textContent || '{}'); } catch { UI_KEYS = {}; }

  // reconstruye un <select> con placeholder unico "—" (con normalizacion y dedupe)
  function rebuildOptions(selectEl, cat) {
    if (!selectEl) return;

    // normalizador de claves por campo
    const normKey = (c, k) => {
      const kl = String(k || '').trim().toLowerCase();
      if (c === 'calcifications') {
        if (kl === 'non' || kl === 'none') return 'none';
        if (kl.startsWith('microcalcification')) return 'microcalcification';
        if (kl.startsWith('macrocalcification')) return 'macrocalcification';
        if (kl === 'rim' || kl === 'peripheral_calcifications') return 'peripheral';
        return k;
      }
      if (c === 'margins') {
        if (kl === 'ill- defined' || kl === 'ill_defined') return 'ill defined';
        if (kl === 'well_defined') return 'well defined';
        if (kl === 'smooth') return 'well defined smooth';
        return k;
      }
      return k;
    };

    const keys = (UI_KEYS?.[cat] || []).map(k => normKey(cat, k));

    // limpia y coloca placeholder
    selectEl.innerHTML = '';
    const opt0 = document.createElement('option');
    opt0.value = '';
    opt0.textContent = '—';
    selectEl.appendChild(opt0);

    // dedupe por clave y por etiqueta renderizada
    const seenKeys = new Set();
    const seenLabels = new Set();

    for (let k of keys) {
      if (seenKeys.has(k)) continue;

      const label = (value_labels?.[cat]?.[k]) || (k.charAt(0).toUpperCase() + k.slice(1));
      if (seenLabels.has(label)) continue;

      const o = document.createElement('option');
      o.value = k;
      o.textContent = label;
      selectEl.appendChild(o);

      seenKeys.add(k);
      seenLabels.add(label);
    }
  }

  let roi_labels = {};
  try { roi_labels = JSON.parse(roiJsonEl?.textContent || '{}'); } catch {}
  let value_labels = {};
  try { value_labels = JSON.parse(valJsonEl?.textContent || '{}'); } catch {}
  // i18n helpers para el modal (mostrar ES, enviar EN)
  const norm = s => String(s??'').trim().toLowerCase()
    .normalize('NFD').replace(/[\u0300-\u036f]/g,'').replace(/\s+/g,'_');

  const ALIAS = {
    echogenicity: {
      isoechogenicity:'isoechoic', isoechogenic:'isoechoic',
      hypoechogenicity:'hypoechoic',
      hyperechogenicity:'hyperechoic',
      'marked hypoechogenicity':'very_hypoechoic',
    },
    margins: {
      'well_defined': 'well defined',
      'ill_defined':  'ill defined',
      'ill- defined': 'ill defined',
      'smooth':       'well defined smooth'
    },
    calcifications: {
      non:'none',
      microcalcifications:'micro',
      microcalcification:'micro',
      macrocalcifications:'macro',
      macrocalcification:'macro',
      rim:'peripheral',
      peripheral_calcifications:'peripheral',
    },
    composition: {
      'esponjiforme':'spongiform','espongiforme':'spongiform','spongiforme':'spongiform','spongiform':'spongiform',
      'solido':'solid','sólido':'solid',
      'quistico':'cystic','quístico':'cystic',
      'denso':'dense',
      'predominantly_solid':'predominantly solid',
      'predominantly_cystic':'predominantly cystic',
    }
  };

  const DIRECT = {
    composition: {
      solid:'Sólido',
      'predominantly solid':'Predominantemente sólido',
      spongiform:'Espongiforme',
      'predominantly cystic':'Predominantemente quístico',
      dense:'Denso',
      cystic:'Quístico',
      mixed:'Mixto',
    },
    echogenicity: {
      isoechoic:'Isoecoico',
      hypoechoic:'Hipoecoico',
      hyperechoic:'Hiperecoico',
      very_hypoechoic:'Muy hipoecoico',
    },
    margins: {
      smooth: 'Lisos',
      well_defined: 'Bien definidos',
      ill_defined: 'Mal definidos',
      spiculated: 'Espiculados',
      microlobulated: 'Microlobulados',
      macrolobulated: 'Macrolobulados',
    },
    calcifications: {
      none:'Ninguna', micro:'Microcalcificaciones', macro:'Macrocalcificaciones', peripheral:'Periféricas',
    },
    sex: { 'F':'Femenino', 'M':'Masculino'}
  };

  // Convierte "Male/Female" de XML a "M"/"F" crudo; y a etiqueta ES para mostrar
  function mapSexFromXml(v) {
    const s = String(v||'').trim().toLowerCase();
    if (s === 'f' || s.startsWith('fem')) return { raw: 'F', es: 'Femenino' };
    if (s === 'm' || s.startsWith('mas')) return { raw: 'M', es: 'Masculino' };
    return { raw: '', es: '' };
  }

  // Lee campos clinicos del XML
  async function readXmlFile(file) {
    const text = await file.text();
    const doc  = new DOMParser().parseFromString(text, 'application/xml');
    if (doc.getElementsByTagName('parsererror').length) throw new Error('XML inválido');

    const node = doc.getElementsByTagName('case')[0] || doc.documentElement;
    const pick = (tag) => (node.getElementsByTagName(tag)[0]?.textContent || '').trim();

    const raw = {
      composition:   pick('composition'),
      echogenicity:  pick('echogenicity'),
      margins:       pick('margins'),
      calcifications:pick('calcifications'),
      sex:           pick('sex'),
      age:           pick('age'),
    };
    return raw;
  }

  // Aplica el XML a los inputs del modal (solo visual). Si hay match, pone ES.
  // Aplica el XML a los inputs del modal (solo visual, sin forzar valores ya existentes)
  function fillModalFromXml(raw) {
    const sexMap = mapSexFromXml(raw.sex);

    document.querySelector('#eh-sex').value = mapSexFromXml(raw.sex).raw || '';
    document.querySelector('#eh-age').value = raw.age || '';

    setSelectValue(document.querySelector('#eh-composition'),   'composition',   raw.composition);
    setSelectValue(document.querySelector('#eh-echogenicity'),  'echogenicity',  raw.echogenicity);
    setSelectValue(document.querySelector('#eh-margins'),       'margins',       raw.margins);
    setSelectValue(document.querySelector('#eh-calcs'),         'calcifications',raw.calcifications);
  }

  // EN crudo -> ES (para mostrar)
  function toES(cat, raw){
    const v = String(raw ?? '').trim(); const vl = v.toLowerCase();
    const dict = value_labels?.[cat] || null;
    if (dict && v in dict) return dict[v];
    if (ALIAS[cat]?.[vl] && dict?.[ALIAS[cat][vl]]) return dict[ALIAS[cat][vl]];
    if (dict){ const cands=[norm(v), vl.replace(/\s+/g,'_')]; for (const c of cands){ if (c in dict) return dict[c]; } }
    if (DIRECT[cat]?.[vl]) return DIRECT[cat][vl];
    return v ? v.charAt(0).toUpperCase()+v.slice(1) : '';
  }

  // ES mostrado -> EN crudo (para enviar al backend/modelo)
  function toRAW(cat, label){
    const v = String(label ?? '').trim(); const n = norm(v);
    const dict = value_labels?.[cat] || {};
    for (const [k, es] of Object.entries(dict)){ if (es===v || norm(es)===n) return k; }
    const d = DIRECT[cat] || {};
    for (const [k, es] of Object.entries(d)){ if (es===v || norm(es)===n) return k; }
    return label; // si ya vino en EN o vacoo, se deja
  }
  const LBL = (field, value) => {
    if (value == null || value === '' || !value_labels[field]) return value ?? '-';
    const m = value_labels[field];
    return (m[value] !== undefined) ? m[value] : (value ?? '-');
  };

  // Deshabilitar manuales si se selecciona XML NUEVO
  const xmlInput = form.querySelector('#eh-xml');
  const xmlHint = form.querySelector('#eh-xml-hint');
  const manualMetaFields = ['#eh-composition','#eh-echogenicity','#eh-margins','#eh-calcs','#eh-sex','#eh-age'].map(id=>form.querySelector(id)).filter(Boolean);
  if (xmlInput) {
    xmlInput.addEventListener('change', async () => {
      if (removeCb?.checked) return;

      const hasFile = xmlInput.files && xmlInput.files.length > 0;
      if (hasFile) {
        try {
          const raw = await readXmlFile(xmlInput.files[0]);
          fillModalFromXml(raw);
          xmlHint?.classList.add('d-none');
        } catch {
          alert('No se pudo leer el XML. Verifica el archivo.');
        }
      }
      manualMetaFields.forEach(field => { if (field) field.disabled = hasFile; });
    });
  }

  // Checkbox "Quitar XML actual"
  const removeCb = form.querySelector('#eh-remove-xml');
  // listener para limpiar al marcar "Quitar XML"
  if (removeCb) {
    removeCb.addEventListener('change', () => {
      const hadXML = form.dataset.hadXml === '1';
      if (removeCb.checked) {
        if (xmlInput) xmlInput.value = '';
        manualMetaFields.forEach(f => { if (f) { f.disabled = false; f.value = ''; } });
        xmlHint?.classList.add('d-none');
        ['composition','echogenicity','margins','calcifications','sex'].forEach(cat => {
          const id = cat === 'calcifications' ? '#eh-calcs' : `#eh-${cat}`;
          rebuildOptions(form.querySelector(id), cat);
        });
      } else {
        // si el registro tenia XML -> vuelve a bloquear manuales
        if (hadXML) {
          manualMetaFields.forEach(f => { if (f) f.disabled = true; });
          xmlHint?.classList.remove('d-none');
        } else {
          manualMetaFields.forEach(f => { if (f) f.disabled = false; });
          xmlHint?.classList.add('d-none');
        }
      }
    });
  }

  // Admin login helpers
  const adminModalEl = document.getElementById('adminLoginModal');
  const adminModal = adminModalEl ? new bootstrap.Modal(adminModalEl) : null;
  const adminForm = document.getElementById('admin-login-form');
  const adminBadge = document.getElementById('admin-badge');

  async function getAdminStatus() {
    try { const r = await fetch('/admin/status'); const j = await r.json(); return !!j.is_admin; }
    catch { return false; }
  }

  function setAdminUI(isAdmin) {
    // toggles en el modal
    const pwdWrap = adminForm?.querySelector('.modal-body');
    const loginBtn = adminForm?.querySelector('button[type="submit"]');
    const logoutBtn = document.getElementById('admin-logout');

    pwdWrap?.classList.toggle('d-none', isAdmin);
    loginBtn?.classList.toggle('d-none', isAdmin);
    logoutBtn?.classList.toggle('d-none', !isAdmin);

    const title = adminModalEl?.querySelector('.modal-title');
    if (title) title.textContent = isAdmin ? 'Sesión de administrador' : 'Iniciar sesión de administrador';

    adminBadge?.classList.toggle('d-none', !isAdmin);
  }

  function toKEY(cat, val){
    const v = String(val ?? '').trim();
    const vl = v.toLowerCase();
    const dict   = value_labels?.[cat] || {};
    const uiList = UI_KEYS?.[cat] || [];

    // Alias primero (evita que entren keys antiguas como non/ill_defined/smooth)
    if (ALIAS[cat]?.[vl]) return ALIAS[cat][vl];

    // Si ya es una key canonica de UI, se usa
    if (uiList.includes(v)) return v;

    // Si es key conocida en value_labels pero NO esta en UI, mapear a su equivalente de UI
    if (v in dict && !uiList.includes(v)) {
      if (cat === 'calcifications') {
        if (vl === 'non') return 'none';
        if (vl === 'microcalcifications') return 'microcalcification';
        if (vl === 'macrocalcifications') return 'macrocalcification';
        if (vl === 'rim' || vl === 'peripheral_calcifications') return 'peripheral';
      }
      if (cat === 'margins') {
        if (vl === 'ill_defined' || vl === 'ill- defined') return 'ill defined';
        if (vl === 'well_defined') return 'well defined';
        if (vl === 'smooth') return 'well defined smooth';
      }
    }

    // Buscar por etiqueta ES que apunte a una key que si este en UI
    const n = norm(v);
    for (const [k, es] of Object.entries(dict)) {
      if ((es === v || norm(es) === n) && uiList.includes(k)) return k;
    }

    // se devolvera tal cual si nada aplica
    return v;
  }

  function setSelectValue(selectEl, cat, rawValue){
    if (!selectEl) return;
    // normaliza a clave del modelo (puede venir en ES/EN/alias)
    const key = toKEY(cat, rawValue);

    // si viene vacio / null / 'u' en sexo -> selecciona placeholder y sale
    if (!key || key === '-' || (cat === 'sex' && key.toLowerCase() === 'u')) {
      selectEl.value = ''; // asume que el placeholder tiene value=""
      return;
    }

    const exists = Array.from(selectEl.options).some(o => o.value === key);
    if (exists) {
      selectEl.value = key;
      return;
    }

    Array.from(selectEl.querySelectorAll('option[data-tmp="1"]'))
        .forEach(o => { if (o.value === key) o.remove(); });

    const opt = document.createElement('option');
    opt.value = key;
    const dict = (value_labels && value_labels[cat]) || {};
    opt.textContent = dict[key] || (key.charAt(0).toUpperCase() + key.slice(1));
    opt.dataset.tmp = '1';
    selectEl.appendChild(opt);
    selectEl.value = key;
  }

  // Al abrir el modal, decide que mostrar
  adminModalEl?.addEventListener('show.bs.modal', async () => {
    const isAdmin = await getAdminStatus();
    setAdminUI(isAdmin);
  });

  document.getElementById('btn-admin-login')?.addEventListener('click', async (ev)=>{
    const isAdmin = await getAdminStatus();
    setAdminUI(isAdmin);
    if (adminModal) adminModal.show();
  });

  adminForm?.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const fd = new FormData(adminForm);
    try{
      const r = await fetch('/admin/login', {method:'POST', body:fd});
      const j = await r.json();
      if(!r.ok || !j.ok) throw new Error(j.error || `HTTP ${r.status}`);
      setAdminUI(true);
      adminModal?.hide();
      const fn = __retryAction; __retryAction = null;
      if (typeof fn === 'function') await fn();
    }catch(err){
      alert(`No se pudo iniciar sesión: ${err.message}`);
    }
  });

  // Cerrar sesion
  document.getElementById('admin-logout')?.addEventListener('click', async ()=>{
    try { await fetch('/admin/logout', {method:'POST'}); } catch {}
    setAdminUI(false);
  });

  async function refreshAdminBadge() {
    try{
      const r = await fetch('/admin/status');
      const j = await r.json();
      const on = !!j.is_admin;
      if (adminBadge) adminBadge.classList.toggle('d-none', !on);
    }catch{}
  }
  refreshAdminBadge();

  // Reintento simple: guarda la ultima acción (fn async) para reejecutar tras login
  let __retryAction = null;
  function promptAdminAndRetry(actionFn){
    __retryAction = actionFn;
    if (adminModal) adminModal.show();
  }

  if (adminForm) {
    adminForm.addEventListener('submit', async (e)=>{
      e.preventDefault();
      const fd = new FormData(adminForm);
      try{
        const r = await fetch('/admin/login', {method:'POST', body:fd});
        const j = await r.json();
        if(!r.ok || !j.ok) throw new Error(j.error || `HTTP ${r.status}`);
        refreshAdminBadge();
        if (adminModal) adminModal.hide();
        const fn = __retryAction; __retryAction = null;
        if (typeof fn === 'function') await fn();
      }catch(err){
        alert(`No se pudo iniciar sesión: ${err.message}`);
      }
    });
  }

  const adminLogoutBtn = document.getElementById('admin-logout');
  if (adminLogoutBtn) {
    adminLogoutBtn.addEventListener('click', async ()=>{
      try { await fetch('/admin/logout', {method:'POST'}); } catch {}
      refreshAdminBadge();
    });
  }

  // Abrir modal EDITAR
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action="edit-h"]');
    if (!btn) return;

    const uid = btn.getAttribute('data-uid');
    if (!uid) return alert('UID no encontrado');

    // guarda la fila actual para refrescar luego
    window.__currentRow = btn.closest('tr');

    // precargar con los datos del boton
    let item = {};
    try { item = JSON.parse(btn.getAttribute('data-json') || '{}'); } catch {}

    rebuildOptions(qs('#eh-composition'),   'composition');
    rebuildOptions(qs('#eh-echogenicity'),  'echogenicity');
    rebuildOptions(qs('#eh-margins'),       'margins');
    rebuildOptions(qs('#eh-calcs'),         'calcifications');
    rebuildOptions(qs('#eh-sex'),           'sex');

    qs('#eh-id').value           = uid;
    qs('#eh-patient').value      = item.patient_id || '';
    qs('#eh-sex').value = (item.sex || '').toUpperCase();
    qs('#eh-age').value = item.age ?? '';

    setSelectValue(qs('#eh-composition'),   'composition',   item.composition);
    setSelectValue(qs('#eh-echogenicity'),  'echogenicity',  item.echogenicity);
    setSelectValue(qs('#eh-margins'),       'margins',       item.margins);
    setSelectValue(qs('#eh-calcs'),         'calcifications',item.calcifications);

    // reset UI del modal
    if (removeCb) removeCb.checked = false;
    if (xmlInput) xmlInput.value = '';

    // ¿El registro actual ya tiene XML?
    const hadXML = !!(item.xml && item.xml !== '-');
    // guardamos este estado en el form para usarlo en submit
    form.dataset.hadXml = hadXML ? '1' : '0';

    // Si tiene XML -> bloquear manuales y mostrar hint; si no, habilitar
    if (hadXML) {
      manualMetaFields.forEach(f => { if (f) f.disabled = true; });
      xmlHint?.classList.remove('d-none');
    } else {
      manualMetaFields.forEach(f => { if (f) f.disabled = false; });
      xmlHint?.classList.add('d-none');
    }

    modal.show();
  });

  // Guardar (con o sin XML)
  form.addEventListener('submit', async (e)=>{
    e.preventDefault();

    const uid = qs('#eh-id')?.value;
    if (!uid) return alert('UID vacío');

    const fd = new FormData(form);
    const xml = fd.get('xmlfile');
    const hasXML = xml && typeof xml === 'object' && xml.size > 0;
    const rmXML  = (() => {
      const v = (fd.get('remove_xml') || '').toString().toLowerCase();
      return v === 'on' || v === '1' || v === 'true' || v === 'yes';
    })();
    const hadXML = form.dataset.hadXml === '1';
    const manualTouched = ['composition','echogenicity','margins','calcifications','sex','age']
      .some(k => (fd.get(k) || '').toString().trim() !== '');

    // Si el registro YA tenia XML y NO se marco quitar ni se subio uno nuevo, y además se intento editar manual → bloquear con mensaje
    if (hadXML && !rmXML && !hasXML && manualTouched) {
      alert('Este registro tiene un XML guardado. Para editar manualmente, marca "Quitar XML actual" o sube un nuevo XML.');
      return;
    }
    const url = (hasXML || rmXML)
      ? `/api/history/${uid}/recompute`
      : `/api/history/${uid}`;

    // Si NO hay XML y NO se esta quitando -> mapea ES->EN y envia campos
    if (!hasXML && !rmXML) {
      fd.set('composition',    toRAW('composition',    form.querySelector('#eh-composition').value));
      fd.set('echogenicity',   toRAW('echogenicity',   form.querySelector('#eh-echogenicity').value));
      fd.set('margins',        toRAW('margins',        form.querySelector('#eh-margins').value));
      fd.set('calcifications', toRAW('calcifications', form.querySelector('#eh-calcs').value));
    }
    const doSubmit = async () => {
      const resp = await fetch(url, { method:'POST', body:fd });
      if (!resp.ok) {
        if (resp.status === 401) {
          return promptAdminAndRetry(doSubmit);
        }
        let j={}; try{ j=await resp.json(); }catch{}
        throw new Error(j.description || j.error || `HTTP ${resp.status}`);
      }
      const updated = await resp.json();

      // refrescar UI
      const row = document.querySelector(`tr[data-uid="${uid}"]`) || window.__currentRow;
      if (row && typeof updateTableRowUI === 'function') updateTableRowUI(row, updated);

      modal.hide();
    };
    try {
      await doSubmit();
    } catch (err) {
      console.error(err);
      alert(`No se pudo guardar: ${err.message}`);
    }
  });

  document.addEventListener('click', async (e)=>{
    const btn = e.target.closest('[data-action="del-h"]');
    if (!btn) return;

    const uid = btn.getAttribute('data-uid');
    if (!uid) return alert('UID no encontrado');

    const purge = e.shiftKey ? 1 : 0;
    const msg = purge
      ? '¿Eliminar PERMANENTEMENTE este registro y sus archivos?'
      : '¿Enviar este registro a papelera (borrado lógico)?';
    if (!confirm(msg)) return;

    const doDelete = async () => {
      const r = await fetch(`/api/history/${uid}?purge=${purge}`, { method:'DELETE' });
      if (!r.ok) {
        if (r.status === 401) {
          return promptAdminAndRetry(doDelete);
        }
        let j={}; try{ j=await r.json(); }catch{}
        throw new Error(j.description || `HTTP ${r.status}`);
      }
      btn.closest('tr')?.remove();
    };
    try{
      await doDelete();
    }catch(err){
      console.error(err);
      alert(`No se pudo eliminar: ${err.message}`);
    }
  });

  window.updateTableRowUI = window.updateTableRowUI || function(row, item){
    const labelCell = row.querySelector('[data-col="label"]');
    const probCell  = row.querySelector('[data-col="prob"]');
    if (labelCell) {
      const isMal = String(item.pred) === '1' || /malig/i.test(item.label||'');
      labelCell.innerHTML = `<span class="badge ${isMal?'bg-danger':'bg-success'}">${item.label||'-'}</span>`;
    }
    if (probCell) {
      probCell.textContent = (item.prob!=null) ? (Number(item.prob)*100).toFixed(2)+'%' : '-';
    }

    // links de XML (preview en panel, no fullscreen)
    const xmlCell = row.querySelector('[data-col="xml"]');
    if (xmlCell) {
      xmlCell.innerHTML = (item.xml && item.xml!=='-')
        ? `<a href="/uploads/${item.xml}" class="link-primary preview-link" data-type="xml" data-file="/uploads/${item.xml}" title="Ver XML">XML</a>`
        : '-';
    }

    // badge ROI con caso 'xml'
    const roiCell = row.querySelector('[data-col="roi"]');
    if (roiCell) {
      let c = 'secondary';
      if (item.roi_source === 'xml') c = 'info';
      else if (item.roi_source === 'meta') c = 'success';
      else if (item.roi_source === 'unet') c = 'warning';
      const nice = roi_labels?.[item.roi_source ?? '-'] || '-';
      roiCell.innerHTML = item.roi_source
        ? `<span class="badge bg-${c}">${nice}</span>`
        : '-';
    }

    // Pintar columnas clinicas
    const setTxt = (col, txt) => { const c = row.querySelector(`[data-col="${col}"]`); if (c) c.textContent = txt ?? '-'; };
    setTxt('composition',    LBL('composition',    item.composition));
    setTxt('echogenicity',   LBL('echogenicity',   item.echogenicity));
    setTxt('margins',        LBL('margins',        item.margins));
    setTxt('calcifications', LBL('calcifications', item.calcifications));
    setTxt('sex',            LBL('sex',            item.sex));
    setTxt('age',            item.age);

    row.dataset.sex = item.sex ?? '';
    row.dataset.age = item.age ?? '';
    row.dataset.composition  = item.composition ?? '';
    row.dataset.echogenicity = item.echogenicity ?? '';
    row.dataset.margins      = item.margins ?? '';
    row.dataset.calcifications = item.calcifications ?? '';

     const editBtn = row.querySelector('[data-action="edit-h"]');
     if (editBtn) {
        try { editBtn.setAttribute('data-json', JSON.stringify(item)); } catch {}
     }
  };

  // Registro (form principal /predict)
  (() => {
    const regForm = document.getElementById('predict-form') || document.querySelector('form[action="/predict"]');
    if (!regForm) return;

    const $c = regForm.querySelector('[name="composition"]');
    const $e = regForm.querySelector('[name="echogenicity"]');
    const $m = regForm.querySelector('[name="margins"]');
    const $k = regForm.querySelector('[name="calcifications"]');
    const $s = regForm.querySelector('[name="sex"]');
    const $a = regForm.querySelector('[name="age"]');

    // Misma lista/orden/ES que en editar
    [[$c,'composition'],[$e,'echogenicity'],[$m,'margins'],[$k,'calcifications'],[$s,'sex']]
      .forEach(([el,cat]) => rebuildOptions(el, cat));

    const xmlInput = regForm.querySelector('input[name="xmlfile"]');
    const manualFields = [$c,$e,$m,$k,$s,$a].filter(Boolean);

    async function applyXmlToRegister(file){
      const raw = await readXmlFile(file);
      setSelectValue($c,'composition',   raw.composition);
      setSelectValue($e,'echogenicity',  raw.echogenicity);
      setSelectValue($m,'margins',       raw.margins);
      setSelectValue($k,'calcifications',raw.calcifications);
      const sx = mapSexFromXml(raw.sex);
      if ($s) $s.value = sx.raw || '';
      if ($a && raw.age) $a.value = raw.age;
    }

    xmlInput?.addEventListener('change', async () => {
      const has = xmlInput.files && xmlInput.files.length > 0;
      if (has) {
        try { await applyXmlToRegister(xmlInput.files[0]); } catch {}
        manualFields.forEach(f => { if (f) f.disabled = true; });
      } else {
        manualFields.forEach(f => {
          if (!f) return;
          f.disabled = false;
          if (f.tagName === 'SELECT') f.value = '';
          else f.value = '';
        });
      }
    });
  })();
});
