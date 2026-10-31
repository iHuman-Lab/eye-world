from pathlib import Path

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
import yaml
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.strategies import DDPStrategy
from torch.utils.data import DataLoader

from data.data_write import create_webdataset
from dataset.pre_process import ComposePreprocessor, Resize, Stack, StackWithLabels
from dataset.torch_dataset import get_torch_dataloaders
from dataset.utils import compute_action_class_weights, get_train_test_files
from evaluate.gym_eval import GymManager, evaluate_policy
from evaluate.utils import ALE_ACTION_NAMES
from models.action_jepa_history import ActionNet as JEPAActionNet
from models.action_net import ACPredictor, ActionNet
from models.networks import ConvNet, UNet
from models.vjepa import (
    ActionEmbedding,
    Predictor,
    TransformerEncoder,
    TubeletEmbedding,
    VJEPAEncoder,
)
from trainers.action_classifier import ActionTraining
from trainers.gaze_predict import GazeTraining
from trainers.jepa import VJEPA, ActionConditionVJEPA
from trainers.jepa_action import ActionHistoryTraining
from trainers.jepa_rollout import RolloutActionJEPA, load_encoder_weights
from utils import InterleavedDataset, skip_run

# The configuration file
config_path = "configs/config.yaml"
config = yaml.load(open(str(config_path)), Loader=yaml.SafeLoader)

CHECKPOINT_DIR = Path(config["action_classifier_checkpoint_dir"])


with skip_run("skip", "data_cleaning") as check, check():
    for game in config["games"]:
        create_webdataset(game, config)


with skip_run("skip", "torch_dataset") as check, check():
    game = config["games"][0]
    preprocessor = ComposePreprocessor([Resize(config), Stack(config)])
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
    plt.ioff()
    plt.show()


with skip_run("skip", "gaze_prediction") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/gaze_prediction/")
    # gaze prediction network
    net = ConvNet(config=config)

    # Dataloader
    preprocessor = ComposePreprocessor([Resize(config)])
    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)
    model = GazeTraining(config, net)

    # Trainer
    trainer = pl.Trainer(
        max_epochs=1,
        logger=logger,
        enable_progress_bar=True,
    )
    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["test"],
    )


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


with skip_run("run", "jepa_training_multi_game") as check, check():
    logger = TensorBoardLogger(
        "tb_logs",
        name="multi_game/vjepa_world_model/",
    )

    preprocessor = ComposePreprocessor(
        [
            Resize(config),
            Stack(config),
        ]
    )

    # Load all game datasets
    datasets = []
    example_loader = None

    for game in config["games"]:
        ds = get_torch_dataloaders(
            game,
            config,
            preprocessor=preprocessor,
        )

        datasets.append(ds["train"].dataset)
        train_loader = ds["train"]

        if example_loader is None:
            example_loader = ds["train"]

        # num_samples = sum(1 for _ in train_loader.dataset)
        # print(f"{game}: ~{num_samples} samples (estimated)")

        print(f"Loaded dataset for {game}")

    combined_dataset = InterleavedDataset(datasets)

    # Build train dataloader
    train_loader = DataLoader(
        combined_dataset,
        batch_size=example_loader.batch_size,
        shuffle=False,
        num_workers=example_loader.num_workers,
        pin_memory=getattr(example_loader, "pin_memory", False),
    )

    # Verify batch shape
    for x, y in train_loader:
        print("Train batch shape:", x.shape)
        print("Label shape:", y.shape)
        break

    # Model configuration
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

    student = TransformerEncoder(
        embed_dim,
        depth=12,
        heads=heads,
        mlp_dim=mlp_dim,
    )

    net = VJEPAEncoder(
        tubelet_embed=tubelet_embed,
        student=student,
    )

    pred = Predictor(
        embed_dim,
        depth=4,
        heads=heads // 2,
        mlp_dim=mlp_dim,
    )

    model = VJEPA(
        model=net,
        pred=pred,
        config=config,
        mask_ratio=0.6,
        lr=1e-4,
        ema_decay=0.996,
    )

    # Trainer
    trainer = pl.Trainer(
        logger=logger,
        max_epochs=config["epochs"],
        accelerator="auto",
        devices="auto",
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(model, train_loader)


with skip_run("skip", "jepa_trainers") as check, check():
    game = config["games"][0]
    logger = TensorBoardLogger("tb_logs", name=f"{game}/vjepa_action_world_model/")

    preprocessor = ComposePreprocessor([Resize(config), StackWithLabels(config)])
    dataloaders = get_torch_dataloaders(game, config, preprocessor=preprocessor)

    patch_dim = 1 if config.get("grey_scale_v", True) else 3
    embed_dim = 768  # 1024
    heads = 12
    mlp_dim = 3072  # 2048

    tubelet_embed = TubeletEmbedding(
        config=config,
        patch_dim=patch_dim,
        embed_dim=embed_dim,
        img_size=config.get("size_x", 84),
    )
    student = TransformerEncoder(embed_dim, depth=12, heads=heads, mlp_dim=mlp_dim)
    net = VJEPAEncoder(tubelet_embed=tubelet_embed, student=student)
    action_embed = ActionEmbedding()
    model = ActionConditionVJEPA(
        model=net,
        action_embed=action_embed,
        config=config,
        lr=1e-4,
        ema_decay=0.996,
    )

    trainer = pl.Trainer(
        logger=logger,
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",  # replaces 'gpus'
        devices=1,  # replaces 'gpus=2'
        max_epochs=config["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(model, dataloaders["train"])


# TODO: run this block and check the validation curves in TensorBoard:
#   1. val/ar_loss_h1 must be lower than val/copy_loss_h1. The copy loss is
#      "the next latent equals the current one"; if the model does not beat it,
#      it has not learned any dynamics.
#   2. val/shuffled_action_loss must be clearly higher than val/ar_loss. If it
#      is about the same, the model ignores the actions.
#   3. train/tf_loss and train/ar_loss should both go down.
#   4. Check the action alignment in RolloutActionJEPA._encode. It assumes the
#      action recorded with a frame is the one that leads to the next frame.
#      Confirm this against the dataset; if it is the other way round, shift
#      the slice by one.
#   If something looks wrong, work out why before changing the model.
with skip_run("skip", "jepa_rollout_with_validation") as check, check():
    game = config["games"][0]

    logger = TensorBoardLogger("tb_logs", name=f"{game}/vjepa_rollout_world_model/")

    # Each sample holds the context window plus ac_num_steps future frames.
    # stack_length stays 4 in the shared config because pretraining needs it.
    ac_config = {
        **config,
        "stack_length": int(config["context_frames"]) + int(config["ac_num_steps"]),
    }

    preprocessor = ComposePreprocessor([Resize(ac_config), StackWithLabels(ac_config)])

    dataloaders = get_torch_dataloaders(game, ac_config, preprocessor=preprocessor)

    # MODEL: frozen pretrained encoder + action-conditioned predictor
    patch_dim = 1 if config.get("grey_scale_v", True) else 3
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

    # Pretrained VJEPA tubelet embedding + student (raises if the file is missing).
    load_encoder_weights(net, config["ckpt_path"])

    predictor = ACPredictor(
        embed_dim=embed_dim,
        dim=int(config["ac_dim"]),
        depth=int(config["ac_depth"]),
        heads=int(config["ac_heads"]),
        num_actions=int(config["num_actions"]),
        num_patches=tubelet_embed.num_patches,
        max_steps=int(config["ac_num_steps"]),
    )

    model = RolloutActionJEPA(net, predictor, ac_config)

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[ModelCheckpoint(monitor="val/ar_loss", mode="min", save_top_k=1)],
        strategy=DDPStrategy(find_unused_parameters=True),
        accelerator="gpu",
        devices=1,
        max_epochs=config["epochs"],
        precision="bf16-mixed",
        log_every_n_steps=10,
    )

    trainer.fit(
        model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders["test"],
    )


with skip_run("skip", "train_action_classifier") as check, check():
    game = config["games"]
    games_name = "_".join(game)
    training_preprocessor = ComposePreprocessor(
        [
            Resize(config),
            StackWithLabels(config),
        ]
    )

    dataset_loaders = get_torch_dataloaders(
        game,
        config,
        preprocessor=training_preprocessor,
    )

    if "train" not in dataset_loaders:
        raise KeyError("get_torch_dataloaders() did not return a 'train' loader.")

    data_loaders = {"train": dataset_loaders["train"]}

    if "val" in dataset_loaders:
        data_loaders["val"] = dataset_loaders["val"]

    if "test" in dataset_loaders:
        data_loaders["test"] = dataset_loaders["test"]

    num_actions = int(config["num_actions"])

    if num_actions != len(ALE_ACTION_NAMES):
        raise ValueError(
            f"The classifier uses the full ALE action space "
            f"({len(ALE_ACTION_NAMES)} classes), but config has {num_actions}."
        )

    # Inverse-frequency weights over the shared 18-way action space.
    train_files = []
    for g in game:
        train_files.extend(get_train_test_files(g, config)[0])

    class_weights = compute_action_class_weights(train_files, num_actions)

    print("\nCLASS WEIGHTS")
    for action_id, weight in enumerate(class_weights.tolist()):
        print(action_id, ALE_ACTION_NAMES[action_id], f"{weight:.3f}")

    action_network = ActionNet(
        num_actions=num_actions,
    )

    training_model = ActionTraining(
        hparams=config,
        net=action_network,
        data_loader=data_loaders,
        class_weights=class_weights,
    )

    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=f"{games_name}-action-classifier-best-curent-test",
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )

    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=f"{games_name}/action_classifier",
    )

    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=1,
        max_epochs=int(config["epochs"]),
        log_every_n_steps=10,
    )

    trainer.fit(training_model)

    if data_loaders.get("test") is not None:
        trainer.test(
            training_model,
            dataloaders=data_loaders["test"],
        )

    print("\nTraining complete")
    print(
        "Last checkpoint:",
        checkpoint_callback.last_model_path,
    )


with skip_run("skip", "action_classifier_in_gym_recording") as check, check():
    runtime_config = dict(config)

    runtime_config["action_classifier_checkpoint"] = (
        "checkpoints/action_classifier/ms_pacman-action-classifier-best-curent-test-v1.ckpt"
    )

    manager = GymManager(
        config=runtime_config,
        preprocessor_pipeline=ComposePreprocessor(
            [
                Resize(config),
                StackWithLabels(config),
            ]
        ),
        action_net=ActionNet(num_actions=int(config["num_actions"])),
        record_video=bool(runtime_config["gym_record_video"]),
        video_folder="./video/action_classifier",
        episode_trigger=lambda episode_id: episode_id == 0,
    )

    evaluate_policy(
        manager,
        num_episodes=int(runtime_config.get("gym_eval_episodes", 10)),
        max_steps=int(runtime_config.get("gym_max_steps", 10_000)),
    )


with skip_run("skip", "train_action_history_classifier") as check, check():
    game = config["games"]

    games_name = "_".join(game)

    # Existing preprocessing

    training_preprocessor = ComposePreprocessor(
        [
            Resize(config),
            StackWithLabels(config),
        ]
    )

    dataset_loaders = get_torch_dataloaders(
        game,
        config,
        preprocessor=training_preprocessor,
    )

    if "train" not in dataset_loaders:
        raise KeyError("get_torch_dataloaders() did not return a 'train' loader.")

    data_loaders = {"train": dataset_loaders["train"]}

    # TODO: get_torch_dataloaders only returns "train" and "test", so the test
    # set is used as validation (it picks the best checkpoint) and is then
    # reported as the test result. Add a real validation split, e.g. hold out
    # some of the train subjects, so the test number is not optimistic.
    if "val" in dataset_loaders:
        data_loaders["val"] = dataset_loaders["val"]

    if "test" in dataset_loaders:
        data_loaders["test"] = dataset_loaders["test"]

    # Full ALE action space
    num_actions = int(config["num_actions"])

    if num_actions != len(ALE_ACTION_NAMES):
        raise ValueError(
            "The classifier uses the full ALE "
            f"action space ({len(ALE_ACTION_NAMES)} classes), "
            f"but config has {num_actions}."
        )

    # Class weights
    train_files = []

    for g in game:
        train_files.extend(
            get_train_test_files(
                g,
                config,
            )[0]
        )

    class_weights = compute_action_class_weights(
        train_files,
        num_actions,
    )

    print("\nCLASS WEIGHTS")

    for (
        action_id,
        weight,
    ) in enumerate(class_weights.tolist()):
        print(
            action_id,
            ALE_ACTION_NAMES[action_id],
            f"{weight:.3f}",
        )

    # Action-conditioned JEPA classifier
    action_network = JEPAActionNet(
        num_actions=num_actions,
        config=config,
        # Must be the pretrained encoder: with None the tubelet embedding
        # and student stay at random init and are then frozen.
        jepa_checkpoint=config["ckpt_path"],
        # First experiment:
        #
        # pretrained JEPA stays fixed
        # action-history token learns
        # classifier learns
        freeze_tubelet=True,
        freeze_student=True,
    )

    # Lightning module
    training_model = ActionHistoryTraining(
        hparams=config,
        net=action_network,
        data_loader=data_loaders,
        class_weights=class_weights,
    )

    # Checkpoint directory
    CHECKPOINT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Best checkpoint
    checkpoint_callback = ModelCheckpoint(
        dirpath=CHECKPOINT_DIR,
        filename=(f"{games_name}-action-history-jepa-best"),
        monitor="val_loss",
        mode="min",
        save_top_k=1,
        every_n_epochs=1,
    )

    # TensorBoard
    logger = TensorBoardLogger(
        save_dir="tb_logs",
        name=(f"{games_name}/action_history_jepa_classifier"),
    )

    # Trainer
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"

    trainer = pl.Trainer(
        logger=logger,
        callbacks=[checkpoint_callback],
        accelerator=accelerator,
        devices=1,
        max_epochs=int(config["epochs"]),
        log_every_n_steps=10,
    )

    # Train
    trainer.fit(training_model)

    # Test
    if data_loaders.get("test") is not None:
        # "best" = the val_loss checkpoint, the same one the gym eval loads.
        trainer.test(
            training_model,
            dataloaders=data_loaders["test"],
            ckpt_path="best",
        )

    print("\nTraining complete")

    print(
        "Best checkpoint:",
        checkpoint_callback.best_model_path,
    )


with skip_run("run", "history_classifier_gym") as check, check():
    runtime_config = dict(config)

    manager = GymManager(
        config=runtime_config,
        preprocessor_pipeline=ComposePreprocessor(
            [
                Resize(runtime_config),
                StackWithLabels(runtime_config),
            ]
        ),
        action_net=JEPAActionNet(
            num_actions=int(runtime_config["num_actions"]),
            config=runtime_config,
            # Encoder weights come from the classifier checkpoint that
            # RuntimeActionNet loads (strict) from action_classifier_checkpoint.
            jepa_checkpoint=None,
            freeze_tubelet=True,
            freeze_student=True,
        ),
        record_video=bool(runtime_config["gym_record_video"]),
        video_folder="./video/action_history_classifier",
        episode_trigger=lambda episode_id: episode_id == 0,
        action_history_length=3,
    )

    evaluate_policy(
        manager,
        num_episodes=int(runtime_config.get("gym_eval_episodes", 10)),
        max_steps=int(runtime_config.get("gym_max_steps", 10_000)),
    )
