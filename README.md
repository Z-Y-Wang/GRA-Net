# GRA-Net Official PyTorch Implementation



## ImageNet-1K Training 
The following commands run the training on a single machine:

### GRA-Net-T
```
torchrun --nproc_per_node=2 main.py \
--model granet_tiny \
--batch_size 256 --update_freq 2 \
--blr 8e-4 \
--epochs 300 \
--warmup_epochs 20 \
--weight_decay 0.05 \
--drop_path 0.2 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--output_dir /path/to/save_results
```

### GRA-Net-B
```
torchrun --nproc_per_node=2 main.py \
--model granet_base \
--batch_size 128 --update_freq 4 \
--blr 6.25e-4 \
--epochs 300 \
--warmup_epochs 20 \
--weight_decay 0.1 \
--drop_path 0.2 \
--reprob 0.25 \
--mixup 0.8 \
--cutmix 1.0 \
--smoothing 0.1 \
--model_ema True --model_ema_eval True \
--use_amp True \
--output_dir /path/to/save_results
```
**For other GRA-Net models, just modify the corresponding settings.**
## ImageNet-1K Testing
For different GRA-Net models, set '--eval' to true to implement ImageNet-1K testing.

```
--eval
```
## ImageNet-1K trained weights 
The weights of the GRA-Net model can be obtained here.
https://pan.baidu.com/s/19ogy4Zx45VqBMkFML5M3Nw 提取码：r3j4
