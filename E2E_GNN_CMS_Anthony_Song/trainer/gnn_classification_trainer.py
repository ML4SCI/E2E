import os
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint
from models.graph_classification import GraphLevelGNN



def train_graph_classifier(model_name, device, train_loader,val_loader, test_loader, **model_kwargs):
    pl.seed_everything(42)
    
    CHECKPOINT_PATH = './saved'

    root_dir = os.path.join(CHECKPOINT_PATH, "GraphLevel" + model_name)
    os.makedirs(root_dir, exist_ok=True)
    trainer = pl.Trainer(default_root_dir=root_dir,
                         callbacks=[ModelCheckpoint(save_weights_only=True, mode="max", monitor="val_acc")],
                         max_epochs=20,
                         enable_progress_bar=True)

    model = GraphLevelGNN(**model_kwargs)
    print(model)
    trainer.fit(model, train_loader, val_loader)
    model = GraphLevelGNN.load_from_checkpoint(trainer.checkpoint_callback.best_model_path)
    
    val_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    test_result = trainer.test(model, dataloaders=test_loader, verbose=False)
    print(val_result, test_result)
    result = {"test": test_result[0]['test_acc'], "valid": val_result[0]['test_acc']} 
    return model, result
