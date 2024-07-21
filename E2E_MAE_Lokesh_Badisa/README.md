To pretrain ViT-MAE:

```bash
python3 pretrain.py --runname <runname> --epochs <> --batch_size <> --lr <> --mask <masking> --encoder_depth <> --encoder_dim <> --enc_heads <> decoder_depth <> --decoder_dim <> --dec_heads <>
```

To finetune ViT-MAE:

```bash
python3 finetune.py -r <runname> -w <weights_file>
```

To linear probe ViT-MAE:

```bash
python3 linearprobe.py -r <runname> -w <weights_file>
```
