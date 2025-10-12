from utils import (
    allowed_file, preprocess_image,
    encode_clinical, encode_clinical_with_mask,
    load_cat_maps, _parse_xml_meta, canonicalize_meta,
    recompute_and_update_record,
)
from model import ThyroidAppWrapper, ModelNotReadyError
from gradcam_utils import GradCAM, overlay_cam_on_pil, overlay_cam_on_full
from detector_unet import UNetMini, unet_infer_box, overlay_mask_on_full
# Alias opcional para mantener un nombre “canónico” si se usa en otra parte:
unet_detect_nodule = unet_infer_box
import app_context as ctx

__all__ = [
    "allowed_file","preprocess_image","encode_clinical","encode_clinical_with_mask",
    "load_cat_maps","_parse_xml_meta","canonicalize_meta","recompute_and_update_record",
    "ThyroidAppWrapper","ModelNotReadyError",
    "GradCAM","overlay_cam_on_pil","overlay_cam_on_full",
    "UNetMini","unet_infer_box","overlay_mask_on_full","unet_detect_nodule",
    "ctx",
]