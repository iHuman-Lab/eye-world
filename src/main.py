import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
from lightning.pytorch.loggers import TensorBoardLogger

from data.data_write import eye_gaze_to_webdataset
from dataset.pre_process_jepa import ComposePreprocessor, Resize, Stack
from dataset.torch_dataset import get_torch_dataloaders
from models.networks import ConvNet, UNet
from trainers.gaze_predict import GazeTraining
from trainers.jepa_main import JEPAWrapper, LightningVJEPA
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


with skip_run("skip", "data_cleaning") as check, check():
    for game in config["games"]:
        eye_gaze_to_webdataset(game, config)


with skip_run("skip", "torch_dataset") as check, check():
    game = config["games"][0]
    preprocessor = ComposePreprocessor([Resize(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )

    for x, y in train_test_dataloaders["train"]:
        print(x.shape)
        print(y.shape)


with skip_run("skip", "gaze_visualization") as check, check():
    game = config["games"][0]
    preprocessor = ComposePreprocessor([Resize(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )
    plt.ion()
    fig, ax = plt.subplots()
    for batch_idx, (imgs, labels) in enumerate(train_test_dataloaders["train"]):
        for i in range(len(imgs)):
            img = imgs[i]
            label = labels[i]
            img_np = img.permute(1, 2, 0).numpy()

            ax.imshow(img_np)
            ax.set_title(f"Frame {batch_idx},{i} Lable : {label}")
            plt.pause(0.1)
            ax.clear()
    plt.ioff
    plt.show()


with skip_run("skip", "gaze_prediction") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/gaze_prediction/")
    # gaze prediction network
    net = ConvNet(config=config)

    # Dataloader
    preprocessor = ComposePreprocessor([Resize(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )
    model = GazeTraining(config, net, train_test_dataloaders)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(model)


with skip_run("skip", "gaze_prediction_conv_deconv") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/gaze_prediction/")
    # Gaze prediction network
    net = UNet(config=config)

    # Dataloader
    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )
    model = GazeTraining(config, net, train_test_dataloaders)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=config["epochs"],
        logger=logger,
        devices=[0],
        accelerator="gpu",
        enable_progress_bar=True,
    )
    trainer.fit(model)


with skip_run("skip", "jepa_main") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/vjepa_world_model/")

    # -----------------------------
    # Preprocessing
    # -----------------------------
    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])

    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)
    train_loader = dataloaders["train"]  # grab the actual DataLoader
    jepa_dataset = JEPAWrapper(train_loader.dataset, config)

    train_dataloader = torch.utils.data.DataLoader(
        jepa_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=1,
        pin_memory=True,
    )

    # -----------------------------
    # Compute patch_dim
    # -----------------------------
    T = config["stack_size"]
    C = 1 if config.get("grey_scale_v", True) else 3
    a = config["patchx"]
    b = config["patchy"]

    patch_dim = T * C * a * b

    stack_len = 4
    C = 1
    patchx = 21
    patchy = 21

    student_patch_dim = C * patchx * patchy * (stack_len - 1)  # 1*21*21*3 = 1323
    teacher_patch_dim = C * patchx * patchy * 1  # 1*21*21*1 = 441

    model = LightningVJEPA(
        config=config,
        patch_dim=441,
        embed_dim=768,
        depth=12,
        predictor_depth=4,
        heads=12,
        mlp_dim=3072,
        mask_ratio=0.6,
        lr=1e-4,
        ema_decay=0.996,
    )

    """
    # ----------------------------- patch_dim=patch_dim,
    # Model
    # -----------------------------
    model = LightningVJEPA(
        config=config,
        student_patch_dim=student_patch_dim,
        teacher_patch_dim=teacher_patch_dim,
        embed_dim=768,
        depth=12,
        predictor_depth=4,
        heads=12,
        mlp_dim=3072,
        mask_ratio=0.6,
        lr=1e-4,
        ema_decay=0.996,
    )"""

    # -----------------------------
    # Trainer
    # -----------------------------
    trainer = pl.Trainer(
        accelerator="gpu",
        devices=1,
        precision=16,
        max_epochs=100,
        logger=logger,
        enable_progress_bar=True,
    )

    trainer.fit(model, train_dataloader)


with skip_run("run", "jepa_main") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/vjepa_world_model/")

    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])

    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)

    train_loader = dataloaders["train"]  # grab the actual DataLoader
    for x, y in train_loader:
        print(x.shape)
