document.addEventListener('DOMContentLoaded', () => {
  console.log('[crud.js] DOM listo');

  const qs = (sel, root=document)=>root.querySelector(sel);
  const form = qs('#edit-history-form');
  const modalEl = qs('#editHistoryModal');
  
  if (!form || !modalEl) {
    console.error('[crud.js] Falta #edit-history-form o #editHistoryModal');
    return;
  }
  const modal = new bootstrap.Modal(modalEl);
  // --- value_labels para mapear etiquetas de la tabla ---
  const valJsonEl = document.getElementById('val-json');
  const roiJsonEl = document.getElementById('roi-json');
  let roi_labels = {};
  try { roi_labels = JSON.parse(roiJsonEl?.textContent || '{}'); } catch {}
  let value_labels = {};
  try { value_labels = JSON.parse(valJsonEl?.textContent || '{}'); } catch {}
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
    xmlInput.addEventListener('change', () => {
      if (removeCb?.checked) return; // si quitar XML está activo, ignorar
      const hasFile = xmlInput.files.length > 0;
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
        // se quitará XML → habilita manuales
        if (xmlInput) xmlInput.value = '';
        manualMetaFields.forEach(f => { if (f) { f.disabled = false; f.value = ''; } });
        xmlHint?.classList.add('d-none');
      } else {
        // si el registro tenía XML → vuelve a bloquear manuales
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

  // ======= Admin login helpers =======
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

    // badge
    adminBadge?.classList.toggle('d-none', !isAdmin);
  }

  // Al abrir el modal, decide qué mostrar
  adminModalEl?.addEventListener('show.bs.modal', async () => {
    const isAdmin = await getAdminStatus();
    setAdminUI(isAdmin);
  });

  // Botón de la barra: si ya eres admin, muestra modal en modo logout
  document.getElementById('btn-admin-login')?.addEventListener('click', async (ev)=>{
    const isAdmin = await getAdminStatus();
    setAdminUI(isAdmin);
    if (adminModal) adminModal.show();
  });

  // Al loguear: ya lo tienes; solo refresca UI después
  adminForm?.addEventListener('submit', async (e)=>{
    e.preventDefault();
    const fd = new FormData(adminForm);
    try{
      const r = await fetch('/admin/login', {method:'POST', body:fd});
      const j = await r.json();
      if(!r.ok || !j.ok) throw new Error(j.error || `HTTP ${r.status}`);
      setAdminUI(true);
      adminModal?.hide();
      // reintento de acción pendiente (si lo usas)
      const fn = __retryAction; __retryAction = null;
      if (typeof fn === 'function') await fn();
    }catch(err){
      alert(`No se pudo iniciar sesión: ${err.message}`);
    }
  });

  // Cerrar sesión
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

  // Reintento simple: guarda la última acción (fn async) para reejecutar tras login
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
        // reintentar
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

  // --- Abrir modal EDITAR
  document.addEventListener('click', (e) => {
    const btn = e.target.closest('[data-action="edit-h"]');
    if (!btn) return;

    const uid = btn.getAttribute('data-uid');
    if (!uid) return alert('UID no encontrado');

    // guarda la fila actual para refrescar luego
    window.__currentRow = btn.closest('tr');

    // precargar con los datos del botón
    let item = {};
    try { item = JSON.parse(btn.getAttribute('data-json') || '{}'); } catch {}

    qs('#eh-id').value           = uid;
    qs('#eh-patient').value      = item.patient_id || '';
    qs('#eh-sex').value          = item.sex || '';
    qs('#eh-age').value          = item.age ?? '';
    qs('#eh-composition').value  = item.composition || '';
    qs('#eh-echogenicity').value = item.echogenicity || '';
    qs('#eh-margins').value      = item.margins || '';
    qs('#eh-calcs').value        = item.calcifications || '';

    // reset UI del modal
    if (removeCb) removeCb.checked = false;
    if (xmlInput) xmlInput.value = '';

    // ¿El registro actual ya tiene XML?
    const hadXML = !!(item.xml && item.xml !== '-');
    // guardamos este estado en el form para usarlo en submit
    form.dataset.hadXml = hadXML ? '1' : '0';

    // Si tiene XML → bloquear manuales y mostrar hint; si no, habilitar
    if (hadXML) {
      manualMetaFields.forEach(f => { if (f) f.disabled = true; });
      xmlHint?.classList.remove('d-none');
    } else {
      manualMetaFields.forEach(f => { if (f) f.disabled = false; });
      xmlHint?.classList.add('d-none');
    }

    modal.show();
  });

  // --- Guardar (con o sin XML) ---
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

    // ¿tocaron manuales?
    const manualTouched = ['composition','echogenicity','margins','calcifications','sex','age']
      .some(k => (fd.get(k) || '').toString().trim() !== '');

    // Si el registro YA tenía XML y NO marcaron quitar NI subieron uno nuevo,
    // y además intentan editar manual → bloquear con mensaje claro
    if (hadXML && !rmXML && !hasXML && manualTouched) {
      alert('Este registro tiene un XML guardado. Para editar manualmente, marca "Quitar XML actual" o sube un nuevo XML.');
      return;
    }
   const url = (hasXML || rmXML)
     ? `/api/history/${uid}/recompute`
     : `/api/history/${uid}`;
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

  // --- Eliminar (soft; Shift+click = purge) ---
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

  // --- Updater de fila (si no existe aún) ---
  window.updateTableRowUI = window.updateTableRowUI || function(row, item){
    // etiqueta / prob
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

    // --- Pintar columnas clínicas ---
    const setTxt = (col, txt) => { const c = row.querySelector(`[data-col="${col}"]`); if (c) c.textContent = txt ?? '-'; };
    setTxt('composition',    LBL('composition',    item.composition));
    setTxt('echogenicity',   LBL('echogenicity',   item.echogenicity));
    setTxt('margins',        LBL('margins',        item.margins));
    setTxt('calcifications', LBL('calcifications', item.calcifications));
    setTxt('sex',            LBL('sex',            item.sex));
    setTxt('age',            item.age);

    // datasets (por si reabres el modal)
    row.dataset.sex = item.sex ?? '';
    row.dataset.age = item.age ?? '';
    row.dataset.composition  = item.composition ?? '';
    row.dataset.echogenicity = item.echogenicity ?? '';
    row.dataset.margins      = item.margins ?? '';
    row.dataset.calcifications = item.calcifications ?? '';
  };
});
