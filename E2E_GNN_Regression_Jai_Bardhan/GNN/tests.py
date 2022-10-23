import torch
from tqdm.auto import tqdm

def get_residuals(args, model, test_loader, device):
    pbar = tqdm(test_loader, total=len(test_loader))

    for data in pbar:
        with torch.no_grad():
            data = data.to(device, non_blocking=True)
            out = model(data)