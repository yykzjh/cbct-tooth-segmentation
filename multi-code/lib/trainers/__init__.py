from .trainer import Trainer
from .surface_trainer import SurfaceTrainer
from .centroid_trainer import CentroidTrainer


def get_trainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric):
    if opt["dataset_name"] == "MULTIPLE-TOOTH":
        trainer = Trainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    elif opt["dataset_name"] == "MULTIPLE-TOOTH-SURFACE":
        trainer = SurfaceTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    elif opt["dataset_name"] == "MULTIPLE-TOOTH-CENTROID":
        trainer = CentroidTrainer(opt, train_loader, valid_loader, model, optimizer, lr_scheduler, loss_function, metric)
    else:
        raise RuntimeError(f"No {opt['dataset_name']} dataset available when initialize trainer")

    return trainer