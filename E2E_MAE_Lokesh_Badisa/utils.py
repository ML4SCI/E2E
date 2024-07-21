import numpy as np

def get_gauss(s,sig):
    x = np.linspace(0, s, s)
    y = np.linspace(0, s, s)
    x, y = np.meshgrid(x, y)
    z = np.exp(-((x - s/2)**2 + (y - s/2)**2) / (2 * sig**2))
    return z

def lr_lambda(epoch, epochs, warmup_epochs):
    #Linear increase followed by cosine decrease
    if epoch < warmup_epochs:
        return (epoch+1) / warmup_epochs
    return 0.5 * (np.cos((epoch - warmup_epochs) / (epochs - warmup_epochs) * np.pi) + 1)