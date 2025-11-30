from typing import Any, Callable, Optional

wrapper: Any = None
cat_maps: dict = {}
DEVICE: Any = None

_prepare_image_with_roi: Optional[Callable] = None
encode_clinical_with_mask: Optional[Callable] = None

def bind(*, wrapper_obj=None, cat_maps_obj=None, device_obj=None,
         prepare_image_fn=None, encode_clinical_fn=None) -> None:
    global wrapper, cat_maps, DEVICE, _prepare_image_with_roi, encode_clinical_with_mask
    if wrapper_obj is not None: wrapper = wrapper_obj
    if cat_maps_obj is not None: cat_maps = cat_maps_obj
    if device_obj is not None: DEVICE = device_obj
    if prepare_image_fn is not None: _prepare_image_with_roi = prepare_image_fn
    if encode_clinical_fn is not None: encode_clinical_with_mask = encode_clinical_fn