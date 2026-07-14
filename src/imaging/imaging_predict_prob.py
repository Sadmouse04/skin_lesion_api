import importlib
from types import SimpleNamespace
import torch
import pandas as pd
from pytorch_lightning import Trainer
import torch
from src.utils.utils import assemble_experiment_path
from src.imaging.imaging_dataset import ImageDataset
from torch.utils.data import DataLoader

# Create global variables to hold the model in RAM
_CACHED_MODEL = None
_CACHED_TRAINER = None

def imaging_predict_prob(config: SimpleNamespace, df: pd.DataFrame) -> pd.DataFrame:
    global _CACHED_MODEL, _CACHED_TRAINER
    experiment_dir, checkpoint_dir = assemble_experiment_path(config)

    _df = df.copy().set_index(config.index_column)
    dataset = ImageDataset(
        df=_df,
        target_column=config.target_column,
        class_to_idx=vars(config.class_to_idx),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
    )

    if _CACHED_MODEL is None:
        print("🚀 First request: Loading 200MB model into RAM...")
        checkpoints = list(checkpoint_dir.rglob("*.ckpt"))
        if len(checkpoints) != 1:
            raise ValueError(f"Expected 1 checkpoint, found {len(checkpoints)}")
        best_checkpoint = checkpoints[0]

        model_module = importlib.import_module("src.models")
        model_class = getattr(model_module, config.model)
        
        # Load the heavy weights ONCE
        _CACHED_MODEL = model_class.load_from_checkpoint(
            best_checkpoint,
            num_labels=dataset.num_classes(),
            lr=config.lr,
        )
        _CACHED_MODEL.eval() # Lock the model for fast inference

        # Initialize the Trainer ONCE
        _CACHED_TRAINER = Trainer(
            default_root_dir=experiment_dir,
            logger=False, # Disable logging to speed up API
            deterministic=True,
        )

    # 3. Use the globally cached model to preprocess the image
    dataset.transform = _CACHED_MODEL.preprocess

    # 4. Predict using the cached model
    print("⚡ Running fast inference...")
    prob_pred = torch.nn.functional.softmax(
        torch.cat(
            _CACHED_TRAINER.predict(
                model=_CACHED_MODEL,
                dataloaders=dataloader,
                # CRITICAL: ckpt_path is REMOVED so it uses the memory, not the hard drive!
            )
        ),
        dim=-1,
    ).numpy()

    idx_to_class = dict((v, k) for k, v in dataset.class_to_idx.items())
    prob_columns = [idx_to_class[i] for i in range(prob_pred.shape[1])]

    return pd.DataFrame(
        prob_pred,
        columns=prob_columns,
        index=_df.index,
    )
