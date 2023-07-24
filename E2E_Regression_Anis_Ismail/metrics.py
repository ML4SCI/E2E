"""Defining the Loss Function"""


def mae_loss_wgtd(pred, true, is_cuda, wgt=1.):
    loss = wgt * (pred - true).abs()
    if is_cuda:
        loss = loss.cuda()
    return loss.mean()
