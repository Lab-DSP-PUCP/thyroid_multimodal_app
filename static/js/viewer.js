// static/js/viewer.js
(() => {
  const overlay = document.getElementById('fs-viewer');
  if (!overlay) return;

  const canvas  = overlay.querySelector('.fsv-canvas');
  const img     = document.getElementById('fsv-img');
  const btns    = overlay.querySelectorAll('[data-fsv]');

  // Estado
  let scale = 1, minScale = 1, maxScale = 8;
  let tx = 0, ty = 0;           // translate X/Y
  let isPanning = false;
  let startX = 0, startY = 0;   // pointer down
  let startTx = 0, startTy = 0; // translate al inicio del pan

  // Util
  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const apply = () => {
    img.style.transform = `translate(${tx}px, ${ty}px) scale(${scale})`;
  };

  function fitToContainer() {
    const rect = canvas.getBoundingClientRect();
    const iw = img.naturalWidth;
    const ih = img.naturalHeight;
    if (!iw || !ih || rect.width <= 0 || rect.height <= 0) return;

    minScale = Math.min(rect.width / iw, rect.height / ih);
    minScale = Math.min(minScale, 1);  // no ampliar por defecto
    scale = minScale;

    // centrar
    tx = (rect.width  - iw * scale) / 2;
    ty = (rect.height - ih * scale) / 2;
    apply();
  }

  function open(src) {
    // mostrar overlay y cargar imagen
    overlay.classList.remove('d-none');
    img.src = src;

    // cuando cargue, ajusta a contenedor
    if (img.complete && img.naturalWidth) {
      fitToContainer();
    } else {
      img.onload = () => fitToContainer();
    }
  }

  function close() {
    overlay.classList.add('d-none');
    img.src = '';
    isPanning = false;
    canvas.style.cursor = 'grab';
  }

  function zoomAt(clientX, clientY, factor) {
    const rect = canvas.getBoundingClientRect();
    const cx = clientX - rect.left;
    const cy = clientY - rect.top;

    // punto de la imagen bajo el cursor (en coords imagen)
    const ix = (cx - tx) / scale;
    const iy = (cy - ty) / scale;

    const newScale = clamp(scale * factor, minScale, maxScale);
    // re-ubicar para mantener el punto bajo el cursor
    tx = cx - ix * newScale;
    ty = cy - iy * newScale;
    scale = newScale;
    apply();
  }

  // Eventos de toolbar
  btns.forEach(b => {
    b.addEventListener('click', () => {
      const action = b.getAttribute('data-fsv');
      if (action === 'close') return close();
      if (action === 'reset') return fitToContainer();
      if (action === 'zoom-in') return zoomAt(canvas.clientWidth/2, canvas.clientHeight/2, 1.2);
      if (action === 'zoom-out') return zoomAt(canvas.clientWidth/2, canvas.clientHeight/2, 1/1.2);
    });
  });

  // Cerrar con Esc
  document.addEventListener('keydown', (e) => {
    if (overlay.classList.contains('d-none')) return;
    if (e.key === 'Escape') close();
  });

  // Doble clic = zoom in (al cursor)
  canvas.addEventListener('dblclick', (e) => {
    e.preventDefault();
    zoomAt(e.clientX, e.clientY, 1.6);
  });

  // Rueda = zoom en cursor (sin Ctrl también)
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault();
    const dir = e.deltaY > 0 ? 1/1.15 : 1.15;
    zoomAt(e.clientX, e.clientY, dir);
  }, { passive: false });

  // Pan con Pointer Events + captura (soluciona “se sigue arrastrando”)
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0) return; // solo botón izquierdo
    isPanning = true;
    startX = e.clientX;
    startY = e.clientY;
    startTx = tx;
    startTy = ty;
    canvas.setPointerCapture(e.pointerId);
    canvas.style.cursor = 'grabbing';
    e.preventDefault();
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!isPanning) return;
    tx = startTx + (e.clientX - startX);
    ty = startTy + (e.clientY - startY);
    apply();
  });

  function endPan(e) {
    if (!isPanning) return;
    isPanning = false;
    try { canvas.releasePointerCapture(e.pointerId); } catch {}
    canvas.style.cursor = 'grab';
  }
  canvas.addEventListener('pointerup', endPan);
  canvas.addEventListener('pointercancel', endPan);
  // Por si acaso, si sale del overlay:
  overlay.addEventListener('pointerup', endPan);
  overlay.addEventListener('pointercancel', endPan);

  // Evita drag nativo del <img>
  img.addEventListener('dragstart', (e) => e.preventDefault());

  // Exponer API mínima para que lo abras desde otros scripts
  window.FSViewer = {
    open, close, reset: fitToContainer
  };
})();