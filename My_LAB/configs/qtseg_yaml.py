import yaml
from typing import Tuple, List, Union
from dataclasses import dataclass, field

@dataclass
class Config:
    epochs: int = 350
    val_epoch_freq: int = 1
    transfer_epochs: int = 50
    batch_size: int = 32
    log_freq: int = 40
    checkpoint_dir: str = "working/checkpoints"
    ckpt_save_fred: int = 5000
    use_amp: bool = False

    # --------------------------------- Optim settings
    # sgd, adamw
    optimizer: str = "adamw"
    momentum: float = 0.99
    betas: Tuple[float, float] = (0.9, 0.999)
    eps: float = 1e-08
    amsgard: bool = False
    nesterov: bool = True

    # --------------------------------- Scheduler settings
    # StepLR, MultiStepLR, ExponentialLR, CosineAnnealingLR, ReduceLROnPlateau, CosineAnnealingWarmRestarts, IdentityScheduler, PolyLR
    scheduler: str = "StepLR"
    learning_rate: float = 0.001
    learning_rate_min: float = 0.00001
    weight_decay: float = 3e-05
    scheduler_last_epoch: int = -1
    # StepLR
    lr_step_size: int = 50
    lr_step_gamma: float = 0.5
    # MultiStepLR
    lr_milestones: List[int] = [50, 100, 150, 200]
    lr_multistep_gamma: float = 0.1
    # ExponentialLR
    lr_exp_gamma: float = 0.99
    # CosineAnnealingLR
    lr_T_max: int = 50
    lr_eta_min: float = 0.00001
    # ReduceLROnPlateau
    lr_plateau_mode: str = "min"
    lr_plateau_factor: float = 0.1
    lr_plateau_patience: int = 10
    lr_plateau_threshold: float = 0.0001
    lr_plateau_threshold_mode: str = "rel"
    lr_plateau_cooldown: int = 0
    lr_plateau_min_lr: float = 0
    lr_plateau_eps: float = 1e-08
    # CosineAnnealingWarmRestarts
    lr_T_0: int = 50
    lr_T_mult: int = 2
    lr_eta_min: float = 1e-6
    # IdentityScheduler - No params, update every step

    # --------------------------------- Model settings
    model_type: str = "QTSeg"
    model_pretrained: str = ""
    img_size: int = 512
    image_embedding_size: Tuple[int, int] = (
        img_size // 16,
        img_size // 16,
    )

    # ----------------- Encoder settings
    encoder_model: str = "FPNEncoder"
    encoder_pretrained: str = "networks/pretrained/fpn-nano.pth"
    encoder_out_features: List[int] = [64, 128, 256]
    image_channel: int = 3
    n_channel: int = 16

    # ----------------- Bridge settings
    bridge_model: str = "MLFD"

    # ----------------- Decoder settings
    num_classes: int = 2  # Num classes
    decoder_model: str = "MaskDecoder"
    decoder_pretrained: str = ""
    mask_depths: List[int] = [1, 2, 3]
    mask_num_head: int = 8
    mask_mlp_dim: int = 2048

    # --------------------------------- Loss & Metric settings
    # Binary, MultiBinary
    metric = "Binary"
    # BCELoss, FocalLoss, CrossEntropyLoss, BinaryDiceLoss, CategoricalDiceLoss
    loss_type: List[str] = ["CrossEntropyLoss", "BinaryDiceLoss"]
    loss_weight: List[float] = [1.0, 1.0]

    focal_alpha: float = 0.25
    focal_gamma: float = 2
    lambda_value: float = 0.5
    dice_smooth: float = 1e-6

    # --------------------------------- Dataset
    # ISIC2016, BUSI, BKAI
    scale_value: float = 255.0
    cvtColor: Union[int, None] = None
    data_root: str = "working/dataset/ISIC2016"
    dataloader: str = "ISIC2016"
    valid_type: str = "test"
    num_workers: int = 8
    # Only used in BKAI for determining the location of the mask
    mask_type: str = ""

    # This SEED will be replaced at runtime and saved in the checkpoint
    SEED: int = 42

    @classmethod
    def from_yaml(cls, yaml_path: str):
        with open(yaml_path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)