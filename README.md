# SceneFlowNet
Implementation of **3D SceneFlowNet: Self-Supervised 3D Scene Flow Estimation**

Install the dependencies as:
```
conda dependencies/environment.yml
```

## Training SceneFlowNet
```bash
./run_flow_training.sh
```
## Testing SceneFlowNet
```bash
./run_flow_testing.sh
```

## Unzip model.best.zip under checkpoints/ to use pretrained model

## Citation
If you find this useful, please cite as:
```
@inproceedings{lu20213d,
  title={3d sceneflownet: Self-supervised 3d scene flow estimation based on graph cnn},
  author={Lu, Yawen and Zhu, Yuhao and Lu, Guoyu},
  booktitle={2021 IEEE International Conference on Image Processing (ICIP)},
  pages={3647--3651},
  year={2021},
  organization={IEEE}
}
```
