(function(){
  let lastSelectedItem = null;
  // Se captura los clics en el historial para saber que filename usar
  document.addEventListener('click', (ev) => {
    const a = ev.target.closest('.preview-link');
    if (!a) return;
    try {
      const raw = a.getAttribute('data-json');
      if (raw) {
        lastSelectedItem = JSON.parse(raw);
      }
    } catch(e) { lastSelectedItem = null; }

    // Habilitar boton si hay imagen asociada
    const btn = document.getElementById('btn-explain');
    if (btn) {
      const hasHistImage = lastSelectedItem && lastSelectedItem.file && lastSelectedItem.file !== '-';
      const hasNewFile   = document.getElementById('file')?.files?.length > 0;
      btn.disabled = !(hasHistImage || hasNewFile);
    }
  });

  // Se habilita el boton cuando el usuario selecciona un archivo nuevo
  const fileInput = document.getElementById('file');
  if (fileInput) {
    fileInput.addEventListener('change', () => {
      const btn = document.getElementById('btn-explain');
      if (btn) btn.disabled = fileInput.files.length === 0;
    });
  }

  // Logica del boton "Explicar"
  const btnExplain = document.getElementById('btn-explain');
  const gradcamView = document.getElementById('gradcam-view');
  const useClinical = document.getElementById('use-clinical-checkbox');

  async function callExplainWithFilename(filename) {
    const fd = new FormData();
    fd.append('filename', filename);
    if (useClinical?.checked) appendClinicalFromRow(fd, lastSelectedItem);
    return fetch('/explain', { method: 'POST', body: fd });
  }

  async function callExplainWithFile(file) {
    const fd = new FormData();
    fd.append('file', file);
    if (useClinical?.checked) appendClinicalFromForm(fd);
    return fetch('/explain', { method: 'POST', body: fd });
  }

  function appendClinicalFromRow(fd, item){
    if (!item) return;
    fd.append('use_clinical','1');
    // Se toman los campos del item si es que existen
    ['composition','echogenicity','margins','calcifications','sex','age'].forEach(k=>{
      if (item[k] !== undefined && item[k] !== null && item[k] !== '') {
        fd.append(k, item[k]);
      }
    });
  }

  function appendClinicalFromForm(fd){
    fd.append('use_clinical','1');
    // Toma los valores del formulario (si se lleno manualmente)
    ['composition','echogenicity','margins','calcifications','sex','age'].forEach(k=>{
      const el = document.querySelector(`[name="${k}"]`);
      if (el && el.value) fd.append(k, el.value);
    });
  }

  btnExplain?.addEventListener('click', async () => {
    try {
      let res;
      const hasHistImage = lastSelectedItem && lastSelectedItem.file && lastSelectedItem.file !== '-';
      const file = document.getElementById('file')?.files?.[0];

      if (hasHistImage) {
        res = await callExplainWithFilename(lastSelectedItem.file);
      } else if (file) {
        res = await callExplainWithFile(file);
      } else {
        alert('Selecciona una imagen o elige una fila del historial.'); return;
      }

      const data = await res.json();
      if (data.overlay_png_base64) {
        gradcamView.src = data.overlay_png_base64;
        gradcamView.classList.remove('d-none');
        gradcamView.scrollIntoView({behavior:'smooth', block:'center'});
      } else {
        alert(data.error || 'No se pudo generar el Grad-CAM.');
      }
    } catch (e) {
      console.error(e);
      alert('Error al solicitar Grad-CAM.');
    }
  });
})();