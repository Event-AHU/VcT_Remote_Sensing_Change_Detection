# VcT: Visual change Transformer for Remote Sensing Image Change Detection 
Here, we provide the pytorch implementation of the paper. For more information, please see our published paper at IEEE TGRS or arXiv. 
[**IEEE TGRS-2023**] **VcT: Visual change Transformer for Remote Sensing Image Change Detection**, Bo Jiang, Zitian Wang, Xixi Wang, Ziyan Zhang, Lan Chen, Xiao Wang, Bin Luo 
[[arXiv](https://arxiv.org/abs/2310.11417)] 
[[IEEE](https://ieeexplore.ieee.org/document/10294300)]



<p align="center">
  <img src="https://github.com/Event-AHU/VcT_Remote_Sensing_Change_Detection/blob/main/VcT_code/framework/framework.jpg" alt="VcT_samples" width="800"/>
</p>



## Requirements

```
Python 3.7
pytorch 1.11.0
einops  0.6.0
torch-scatter 2.0.9
scipy 1.7.3
matplotlib  3.5.3
```

## Train

You can find the training script `run_cd.sh`. You can run the script file by `sh run_cd.sh` in the command environment.

The dataset path is modified in `data_config.py`.

The detailed script file `run_cd.sh` is as follows:

```cmd
gpus=0
checkpoint_root=checkpoints 
data_name=LEVIR  # dataset name 

img_size=256
batch_size=8
lr=0.01
max_epochs=200  #training epochs
net_G=Reliable_transformer # model name
lr_policy=linear

split=train  # training txt
split_val=val  #validation txt
project_name=CD_${net_G}_${data_name}_b${batch_size}_lr${lr}_${split}_${split_val}_${max_epochs}_${lr_policy}

python main_cd.py --img_size ${img_size} --checkpoint_root ${checkpoint_root} --lr_policy ${lr_policy} --split ${split} --split_val ${split_val} --net_G ${net_G} --gpu_ids ${gpus} --max_epochs ${max_epochs} --project_name ${project_name} --batch_size ${batch_size} --data_name ${data_name}  --lr ${lr}
```

## Evaluate

Checkpoints of our model can be downloaded from: [[Onedrive](https://www.dropbox.com/scl/fi/bc8thurpi51385m7q3c6u/VcT_checkpoints.zip?rlkey=pzjx52tfomno0y1fv4gmnxu0p&dl=0)]

You can find the evaluation script `eval.sh`. You can run the script file by `sh eval.sh` in the command environment.

The detailed script file `eval.sh` is as follows:

```cmd
gpus=0
data_name=LEVIR # dataset name
net_G=Reliable_transformer # model name 
split=test # test.txt
project_name=VcT_LEVIR # the name of the subfolder in the checkpoints folder 
checkpoint_name=best_ckpt.pt # the name of evaluated model file 

python eval_cd.py --split ${split} --net_G ${net_G} --checkpoint_name ${checkpoint_name} --gpu_ids ${gpus} --project_name ${project_name} --data_name ${data_name}
```

## Dataset Preparation

### Data structure

```
"""
Change detection data set with pixel-level binary labels；
├─A
├─B
├─label
└─list
"""
```

`A`: images of t1 phase;

`B`:images of t2 phase;

`label`: label maps;

`list`: contains `train.txt, val.txt and test.txt`, each file records the image names (XXX.png) in the change detection dataset.



### Our Processed Dataset Download 

**LEVIR-CD (2.3GB):** [[DropBox](https://www.dropbox.com/scl/fi/6ug4nxeaj9u4y8f0olixq/LEVIR.zip?rlkey=ikm2a7eju5oi9ffhfxkq5mqr4&dl=0)]

**WHU-CD (1.82GB):** [[DropBox](https://www.dropbox.com/scl/fi/9i4ra8k6g2zn8mpdwibay/WHU.zip?rlkey=5w0c0vsvvrhypsqdspnavfyfa&dl=0)]

**DSIFN-CD (3.38GB):** [[DropBox](https://www.dropbox.com/scl/fi/xc3dor9cc85ev8c7morbp/DSIFN.zip?rlkey=yl69h63qlm96sdcgg2qmlnnx6&dl=0)]



## License
Code is released for non-commercial and research purposes **only**. For commercial purposes, please contact the authors.



## Citation
If you use this code for your research, please cite our paper:

```
@article{jiang2023vct,
  title={VcT: Visual change Transformer for Remote Sensing Image Change Detection},
  author={Jiang, Bo and Wang, Zitian and Wang, Xixi and Zhang, Ziyan and Chen, Lan and Wang, Xiao and Luo, Bin},
  journal={arXiv preprint arXiv:2310.11417},
  year={2023}
}
```

