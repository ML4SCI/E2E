"""Defining Train and Evaluation Utilities"""
import gc
import torch
import time
import numpy as np
import matplotlib.pyplot as plt
import json
import os
from utils import transform_y, inv_transform, logger
from metrics import mae_loss_wgtd

params = json.load(open("./E2E/E2E_Regression_Anis_Ismail/experiment.json",'r'))


def load_model(model_name, resnet, optimizer, lr_scheduler):
    print(model_name)
    checkpoint = torch.load(os.path.join(params["save_path"], "MODELS") + "/%s/%s " % (params["expt_name"], model_name))
    resnet.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    lr_scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return resnet, optimizer, lr_scheduler


def do_eval(resnet, val_loader, epoch):
    mass_bins = np.arange(85, 500, 25)  # for histogram in eval()
    loss_ = 0.
    m_pred_, m_true_, mae_, mre_ = [], [], [], []
    now = time.time()
    ma_low = transform_y(85.0)  # convert from GeV to network units
    for i, data in enumerate(val_loader):
        X, m0, iphi, ieta = data['X_jet'].cuda(), data['genM'].cuda(), data['iphi'].cuda(), data['ieta'].cuda()
        X = X[m0[:, 0] > ma_low]
        iphi = iphi[m0[:, 0] > ma_low]
        ieta = ieta[m0[:, 0] > ma_low]
        m0 = m0[m0[:, 0] > ma_low]
        logits = resnet([X, iphi, ieta])
        loss_ += mae_loss_wgtd(logits, m0, is_cuda=params["is_cuda"]).item()

        # Undo preprocessing on mass
        logits, m0 = inv_transform(logits), inv_transform(m0)
        mae = (logits - m0).abs()
        mre = ((logits - m0).abs() / m0)
        # Store batch metrics:

        m_pred_.append(logits.tolist())
        m_true_.append(m0.tolist())
        mae_.append(mae.tolist())
        mre_.append(mre.tolist())

        gc.collect()

    now = time.time() - now
    m_true_ = np.concatenate(m_true_)
    m_pred_ = np.concatenate(m_pred_)
    mae_ = np.concatenate(mae_)
    mre_ = np.concatenate(mre_)

    logger('%d: Val m_pred: %s... ' % (epoch, str(np.squeeze(m_pred_[:5]))), f, params["run_logger"])
    logger('%d: Val m_true: %s... ' % (epoch, str(np.squeeze(m_true_[:5]))), f, params["run_logger"])
    logger('%d: Val time:%.2fs in %d steps for N=%d ' % (epoch, now, len(val_loader), len(m_true_)), f,
           params["run_logger"])
    logger('%d: Val loss:%f, mae:%f, mre:%f ' % (epoch, loss_ / len(val_loader), np.mean(mae_), np.mean(mre_)), f,
           params["run_logger"])

    score_str = 'epoch%d_mae%.4f ' % (epoch, np.mean(mae_))

    # Check 1D m_pred
    hst = np.histogram(np.squeeze(m_pred_), bins=mass_bins)[0]
    logger('%d: Val m_pred, [85,500,25] MeV: %s ' % (epoch, str(np.uint(hst))), f, params["run_logger"])
    mlow = hst[0]
    mrms = np.std(hst)
    logger('%d: Val m_pred, [85,500,25] MeV: low:%d, rms: %f ' % (epoch, mlow, mrms), f, params["run_logger"])
    plt.hist(m_true_, range=(80, 510), bins=20, histtype='step', label=r'$\mathrm{m_{true}}$', linestyle='--',
             color='grey', alpha=0.6)
    plt.hist(m_pred_, range=(80, 510), bins=20, histtype='step', label=r'$\mathrm{m_{pred}}$', linestyle='--',
             color='C0', alpha=0.6)
    plt.xlim(80, 510)
    plt.xlabel(r'$\mathrm{m}$', size=16)
    plt.legend(loc='upper right')
    plt.savefig(
        os.path.join(params["save_path"], "PLOTS") + '/%s/test_mpred_%s.png' % (params["expt_name"], score_str),
        bbox_inches='tight')
    plt.close()
    return mre_, m_pred_, m_true_, np.mean(mae_)


def train(resnet, optimizer, lr_scheduler, epochs, train_loader, val_loader):
    if params["load_epoch"] != 0:
        model_name = 'Tops_ResNet_blocks_3_model_epoch_%d ' % (params["load_epoch"])
        resnet, optimizer, lr_scheduler = load_model(model_name, resnet, optimizer, lr_scheduler)
    print_step = 100
    resnet.train()
    for e in range(epochs):
        resnet.train()
        global f
        f = open(os.path.join(params["save_path"], "LOGS") + '/%s.log ' % (params["expt_name"]), 'a')
        epoch = e + 1 + params["load_epoch"]
        n_trained = 0
        loss_ = 0.
        logger('>> Epoch %d <<<<<<<<' % (epoch), f)

        # Run training
        resnet.train()
        now = time.time()
        for i, data in enumerate(train_loader):
            X, m0, iphi, ieta = data['X_jet'].cuda(), data['genM'].cuda(), data['iphi'].cuda(), data['ieta'].cuda()
            optimizer.zero_grad()
            logits = resnet([X, iphi, ieta])
            loss = mae_loss_wgtd(logits, m0, is_cuda=params["is_cuda"])
            loss.backward()
            loss_ += loss.item()
            optimizer.step()

            n_trained += 1

            if i % print_step == 0:
                logits, m0 = inv_transform(logits), inv_transform(m0)
                mae = (logits - m0).abs().mean()
                mre = ((logits - m0).abs() / m0).mean()
                logger('%d: (%d/%d) m_pred: %s' % (epoch, i, len(train_loader), str(np.squeeze(logits.tolist()[:5]))),
                       f, params["run_logger"])
                logger('%d: (%d/%d) m_true: %s' % (epoch, i, len(train_loader), str(np.squeeze(m0.tolist()[:5]))), f,
                       params["run_logger"])
                logger('%d: (%d/%d) Train loss: %f, mae: %f,mre: %f' % (
                    epoch, i, len(train_loader), loss.item(), mae.item(), mre.item()), f, params["run_logger"])
            gc.collect()

        now = time.time() - now
        logits, m0 = inv_transform(logits), inv_transform(m0)
        mae = (logits - m0).abs().mean()
        mre = ((logits - m0).abs() / m0).mean()
        logger('%d: Train time: %.2fs in %d steps for N:%d' % (epoch, now, len(train_loader), n_trained), f,
               params["run_logger"])
        logger('%d: Average Train loss: %f, Final mae: %f,Final mre: %f' % (
            epoch, loss_ / len(train_loader), mae.item(), mre.item()), f, params["run_logger"])

        gc.collect()

        # Run Validation
        resnet.eval()
        _, _, _, val_loss = do_eval(resnet, val_loader, epoch)
        lr_scheduler.step(val_loss)
        gc.collect()
        torch.save({
            'epoch': e,
            'model_state_dict': resnet.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': lr_scheduler.state_dict()
        }, os.path.join(params["save_path"], "MODELS") + '/%s/Tops_ResNet_blocks_3_model_epoch_%d' % (
            params["expt_name"], epoch))

        if params["run_logger"]:
            f.close()
