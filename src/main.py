import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import webdataset as wds
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from torch.utils.data import DataLoader

from data.data_write import eye_gaze_to_webdataset
from dataset.pre_process import ComposePreprocessor, Resize, Stack
from dataset.torch_dataset import get_torch_dataloaders
from dataset.tri_stack import StackWithLabels
from models.networks import ConvNet, UNet
from models.vjepa import Predictor, TransformerEncoder, TubeletEmbedding, VJEPAEncoder
from trainers.gaze_predict import GazeTraining
from trainers.jepa import VJEPA, ActionConditionVJEPA
from utils import skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)


with skip_run("skip", "data_cleaning") as check, check():
    for game in config["games"]:
        eye_gaze_to_webdataset(game, config)


with skip_run("skip", "torch_dataset") as check, check():
    game = config["games"][0]
    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])
    train_test_dataloaders = get_torch_dataloaders(
        game, config, preprocessor=preprocessor
    )

    for x, y, z in train_test_dataloaders["train"]:
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
    plt.ioff()
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


with skip_run("skip", "jepa_training") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/vjepa_world_model/")

    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])

    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)
    train_loader = dataloaders["train"]

    for x, y, action in train_loader:
        print("Train batch shape:", x.shape)  # [32, 4, 84, 84]
        break

    patch_dim = 1 if config.get("grey_scale", True) else 3
    embed_dim = 768
    heads = 12
    mlp_dim = 3072

    tubelet_embed = TubeletEmbedding(
        config=config,
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        img_size=config.get("size_x", 84),
    )
    student = TransformerEncoder(embed_dim, depth=12, heads=heads, mlp_dim=mlp_dim)
    net = VJEPAEncoder(tubelet_embed=tubelet_embed, student=student)

    pred = Predictor(embed_dim, depth=4, heads=heads // 2, mlp_dim=mlp_dim)
    model = VJEPA(
        model=net,
        pred=pred,
        config=config,
        mask_ratio=0.6,
        lr=1e-4,
        ema_decay=0.996,
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader)
with skip_run("skip", "jepa_action_training") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/vjepa_world_model/")

    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])

    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)
    train_loader = dataloaders["train"]

    ckpt_path = "/home/cody/Documents/IHL/eye-world/tb_logs/ms_pacman/vjepa_world_model/version_2/checkpoints/epoch=49-step=138850.ckpt"

    ckpt = torch.load(ckpt_path, map_location="cpu")
    state_dict = ckpt["state_dict"]

    for x, y in train_loader:
        print("Train batch shape:", x.shape)  # [32, 4, 84, 84]
        break

    patch_dim = 1 if config.get("grey_scale_v", True) else 3
    embed_dim = 1024
    heads = 8
    mlp_dim = 2048

    tubelet_embed = TubeletEmbedding(
        config=config,
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        img_size=config.get("size_x", 84),
    )
    student = TransformerEncoder(embed_dim, depth=12, heads=heads, mlp_dim=mlp_dim)
    net = VJEPAEncoder(tubelet_embed=tubelet_embed, student=student)
    action_embed = ActionToken()
    predd = Predictor(embed_dim, depth=4, heads=heads // 2, mlp_dim=mlp_dim)
    model = ActionConditionVJEPA(
        model=net,
        action_embed=action_embed,
        config=config,
        lr=1e-4,
        ema_decay=0.996,
    )

    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, train_loader)

with skip_run("skip", "stack_creator") as check, check():
    # -----------------------------
    # Preprocessor
    # -----------------------------
    preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])

    # -----------------------------
    # Helper: create WebDataset DataLoader
    # -----------------------------
    def get_web_dataloader(config, preprocessor, split="train", batch_size=2):
        data_dir = config["processed_data_path"]  # e.g., "data/"
        file_pattern = f"{data_dir}/{split}-*.tar"  # automatically pick up shards

        dataset = (
            wds.WebDataset(
                file_pattern, shardshuffle=False, nodesplitter=wds.split_by_worker
            )
            .decode("pil")  # decode images to PIL
            .to_tuple("jpg", "json", "cls")  # tuple: (image, gaze, action)
        )

        # Lazy preprocessing
        def preprocess_dataset(ds):
            for sample in ds:
                yield preprocessor(sample)

        preprocessed_dataset = preprocess_dataset(dataset)

        # Collate function to batch tuples of tensors
        def collate_fn(batch):
            imgs, gazes, actions = zip(*batch)
            return (
                torch.stack(imgs),
                torch.stack(gazes),
                torch.stack(actions),
            )

        return DataLoader(
            preprocessed_dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # -----------------------------
    # Create DataLoader
    # -----------------------------
    train_loader = get_web_dataloader(config, preprocessor, split="train", batch_size=2)

    # -----------------------------
    # Iterate and print first batch
    # -----------------------------
    for batch_idx, (stacked_imgs, stacked_gaze, stacked_actions) in enumerate(
        train_loader
    ):
        print(f"Batch {batch_idx}:")
        print("Stacked images:", stacked_imgs.shape)  # e.g., [2, stack_len*C, H, W]
        print(
            "Stacked gaze:", stacked_gaze.shape
        )  # e.g., [2, stack_len, 2] or [2, stack_len*C, H, W]
        print("Stacked actions:", stacked_actions.shape)  # e.g., [2, stack_len]
        break  # just show first batch


with skip_run("run", "stack_creator") as check, check():
    # -----------------------------
    # Preprocessor
    # -----------------------------
    preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])
    game = config["games"][0]
    # -----------------------------
    # Helper: create WebDataset DataLoader
    # -----------------------------

    def get_web_dataloader(config, preprocessor, split="train", batch_size=2):
        data_dir = config["processed_data_path"]  # e.g., "data/"
        print(config["processed_data_path"])
        file_pattern = f"{data_dir}/{split}-*.tar"  # automatically pick up shards

        # WebDataset with preprocessing applied directly
        dataset = (
            wds.WebDataset(
                file_pattern, shardshuffle=False, nodesplitter=wds.split_by_worker
            )
            .decode("pil")  # decode images to PIL
            .to_tuple("jpg", "json", "cls")  # tuple: (image, gaze, action)
            .map(preprocessor)  # preprocess items directly
        )

        # Collate function to batch tuples of tensors
        def collate_fn(batch):
            imgs, gazes, actions = zip(*batch)
            return (
                torch.stack(imgs),
                torch.stack(gazes),
                torch.stack(actions),
            )

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            collate_fn=collate_fn,
        )

    # -----------------------------
    # Create DataLoader
    # -----------------------------
    # train_loader = get_torch_dataloaders(game, config, preprocessor=preprocessor)
    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)
    train_loader = dataloaders["train"]

    # train_loader = get_web_dataloader(config, preprocessor, split="train", batch_size=2)
    print(config["processed_data_path"])
    # -----------------------------
    # Iterate and print first batch
    # -----------------------------
    # train_loader = DataLoader["train"]  # access the actual DataLoader

    # Iterate over the first batch safely
    for batch_idx, batch in enumerate(train_loader):
        # Inspect the batch type
        if isinstance(batch, tuple) and len(batch) == 3:
            stacked_imgs, stacked_gaze, stacked_actions = batch
            print(f"Batch {batch_idx}:")
            print("Stacked images:", stacked_imgs.shape)
            print("Stacked gaze:", stacked_gaze.shape)
            print("Stacked actions:", stacked_actions.shape)
        else:
            # Preprocessor returned a single object (dict, list, or custom structure)
            print(f"Batch {batch_idx} has unexpected structure:")
            print(batch)
        break  # only show first batch


"""    for batch_idx, (stacked_imgs, stacked_gaze, stacked_actions) in enumerate(
        train_loader
    ):
        print(f"Batch {batch_idx}:")
        print("Stacked images:", stacked_imgs.shape)  # e.g., [2, stack_len*C, H, W]
        print("Stacked gaze:", stacked_gaze.shape)  # e.g., [2, stack_len, 2]
        print("Stacked actions:", stacked_actions.shape)  # e.g., [2, stack_len]
        break  # just show first batch
"""
