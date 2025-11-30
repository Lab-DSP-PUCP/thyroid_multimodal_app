from contextlib import contextmanager
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class ModelNotReadyError(Exception):
    pass

class MultimodalThyroidClassifier(nn.Module):
    def __init__(self, tabular_input_dim: int, num_classes: int = 1, dropout_p: float = 0.3):
        super().__init__()

        # ResNet-18 preentrenada
        cnn = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        feat_dim = cnn.fc.in_features  # 512

        # Congelar todo menos layer4
        for p in cnn.parameters():
            p.requires_grad = False
        for name, p in cnn.named_parameters():
            if name.startswith("layer4"):
                p.requires_grad = True

        cnn.fc = nn.Identity()
        self.cnn = cnn
        self.tabular_dim = int(tabular_input_dim)

        # Rama tabular
        self.tabular_net = nn.Sequential(
            nn.Linear(self.tabular_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(64, 32),
            nn.ReLU(),
        )

        # Clasificador final (imagen + tabular -> 1 logit)
        self.classifier = nn.Sequential(
            nn.Linear(feat_dim + 32, 128),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(128, num_classes),
        )

    def forward(self, imgs: torch.Tensor, clinical: torch.Tensor | None):
        """
        imgs:      [B, 3, 224, 224]
        clinical:  [B, tab_dim] o [tab_dim] o None
        - Si clinical es None → se usa vector-cero.
        - Si viene 1D → se convierte a batch de 1.
        - Si batch o nº de columnas no coincide → se ajusta (expand, recorte, padding).
        """

        # Features de imagen
        img_feat = self.cnn(imgs)           # [B, 512]
        B = img_feat.size(0)
        dev = img_feat.device
        dt = img_feat.dtype

        # Normalizacion de clinical
        if clinical is None:
            # Vector-cero [B, tabular_dim]
            clinical = imgs.new_zeros((B, self.tabular_dim), dtype=dt, device=dev)
        else:
            # Asegurar que sea tensor
            if not torch.is_tensor(clinical):
                clinical = torch.as_tensor(clinical, dtype=dt, device=dev)
            else:
                # Alinear dtype/device
                if clinical.device != dev or clinical.dtype != dt:
                    clinical = clinical.to(dev, dtype=dt)

            # Asegurar 2D [batch, features]
            if clinical.dim() == 1:
                clinical = clinical.unsqueeze(0)    # [1, F]

            # Ajustar batch:
            # Si solo hay 1 fila y B > 1 -> repetir
            if clinical.size(0) == 1 and B > 1:
                clinical = clinical.expand(B, -1)
            # Si el batch es distinto -> recortar a B
            elif clinical.size(0) != B:
                clinical = clinical[:B, ...]

            # Ajustar # de columnas a self.tabular_dim
            feat_dim = clinical.size(1)
            if feat_dim < self.tabular_dim:
                # Padding con ceros al final
                pad = self.tabular_dim - feat_dim
                padding = clinical.new_zeros((clinical.size(0), pad))
                clinical = torch.cat([clinical, padding], dim=1)
            elif feat_dim > self.tabular_dim:
                # Truncar columnas extra
                clinical = clinical[:, :self.tabular_dim]

        # Rama tabular
        clin_feat = self.tabular_net(clinical)    # [B, 32]

        # Fusion
        fused = torch.cat([img_feat, clin_feat], dim=1)  # [B, 512+32]

        # Clasificador final
        return self.classifier(fused)             # [B, num_classes]


class ThyroidAppWrapper:
    def __init__(self, weights_path: str, tabular_input_dim: int, device: str | None = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

        # construye el modelo base (tolerante a clinical = None)
        self.model = MultimodalThyroidClassifier(tabular_input_dim).to(self.device)

        # carga checkpoint
        try:
            ck = torch.load(weights_path, map_location="cpu")
        except FileNotFoundError as e:
            raise ModelNotReadyError(str(e))

        # soporta simple o ensamble
        state_dicts = ck.get("state_dicts")
        if state_dicts and isinstance(state_dicts, (list, tuple)) and len(state_dicts) > 0:
            self.model.load_state_dict(state_dicts[0], strict=False)  # 1er miembro del ensamble
        else:
            self.model.load_state_dict(ck["model_state"], strict=False)

        self.model.eval()

        # umbrales disponibles en el wrapper
        self.tau_mm  = float(ck.get("tau_opt", 0.5)) # con metadatos
        self.tau_img = float(ck.get("tau_opt_img", self.tau_mm))  # solo imagen (fallback)
        self.tau_by_k = {int(k): float(v) for k, v in ck.get("tau_by_k", {}).items()}
        self.ck_cat_maps = ck.get("cat_maps")

    @contextmanager
    def inference_mode(self):
        old = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(False)
            yield
        finally:
            torch.set_grad_enabled(old)

    def predict(self, img_tensor, clin_tensor=None):
        """
        img_tensor: [3,224,224] o [1,3,224,224]
        clin_tensor: [tab_dim] o [1,tab_dim] o None
        Devuelve (pred_por_defecto, prob). La app recalcula el pred con el umbral correcto.
        """
        img = img_tensor if img_tensor.dim() == 4 else img_tensor.unsqueeze(0)
        img = img.to(self.device)

        clin = None
        if clin_tensor is not None:
            if not torch.is_tensor(clin_tensor):
                clin_tensor = torch.tensor(clin_tensor, dtype=torch.float32)
            if clin_tensor.dim() == 1:
                clin_tensor = clin_tensor.unsqueeze(0)
            clin = clin_tensor.to(self.device)

        logits = self.model(img, clin) # el modelo maneja clin=None
        prob = torch.sigmoid(logits).reshape(-1)[0].item()
        # pred usando tau_mm; en la app se recalcula con tau_mm/tau_img
        pred = 1 if prob >= self.tau_mm else 0
        return pred, prob