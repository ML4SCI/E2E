import torch
import torch.nn.functional as F


def nll_loss(output, target):
    return F.nll_loss(output, target)

class LossLoader():
    def __init__(self,criterion_type, criterion_beta=None, predict_bins=False) -> None:
        self.criterion_dict = {}
        if criterion_type == 'mse' or criterion_type == 'l2':
            self.criterion_dict['regress'] = torch.nn.MSELoss()
        elif criterion_type == 'l1':
            self.criterion_dict['regress'] = torch.nn.L1Loss()
        elif criterion_type == 'smoothl1':
            self.criterion_dict['regress'] = torch.nn.SmoothL1Loss(beta=criterion_beta)
        
        if predict_bins:
            # self.criterion_dict['class'] = torch.nn.CrossEntropyLoss()
            # self.criterion_dict['class'] = torch.nn.BCELoss()
            self.criterion_dict['class'] = torch.nn.BCEWithLogitsLoss()
        
    def get_loss(self):
        return self.criterion_dict