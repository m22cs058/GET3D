## GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images (NeurIPS 2022)<br><sub>Official PyTorch implementation </sub>

![Teaser image](./docs/assets/get3d_model.png)

**GET3D: A Generative Model of High Quality 3D Textured Shapes Learned from Images**<br>
[Jun Gao](http://www.cs.toronto.edu/~jungao/)
, [Tianchang Shen](http://www.cs.toronto.edu/~shenti11/)
, [Zian Wang](http://www.cs.toronto.edu/~zianwang/),
[Wenzheng Chen](http://www.cs.toronto.edu/~wenzheng/), [Kangxue Yin](https://kangxue.org/)
, [Daiqing Li](https://scholar.google.ca/citations?user=8q2ISMIAAAAJ&hl=en),
[Or Litany](https://orlitany.github.io/), [Zan Gojcic](https://zgojcic.github.io/),
[Sanja Fidler](https://www.cs.toronto.edu/~fidler/) <br>
**[Paper](https://nv-tlabs.github.io/GET3D/assets/paper.pdf)
, [Project Page](https://nv-tlabs.github.io/GET3D/)**

## Requirements

* Use [script](./install_get3d.sh) to install packages.

## Preparing datasets

GET3D is trained on synthetic dataset. We provide rendering scripts for Shapenet. Please
refer to [readme](./render_shapenet_data/README.md) to download shapenet dataset and
render it.

## Train the model

#### Clone the gitlab code and necessary files:

```bash
cd YOUR_CODE_PATH
git clone git@github.com:nv-tlabs/GET3D.git
cd GET3D; mkdir cache; cd cache
wget https://api.ngc.nvidia.com/v2/models/nvidia/research/stylegan3/versions/1/files/metrics/inception-2015-12-05.pkl
```

#### Train the model

```bash
cd YOUR_CODE_PATH 
export PYTHONPATH=$PWD:$PYTHONPATH
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
```

- Train on the unified generator on cars.

```bash
python train_3d.py --outdir=PATH_TO_LOG --data=PATH_TO_RENDER_IMG --camera_path PATH_TO_RENDER_CAMERA --gpus=8 --batch=32 --gamma=40 --data_camera_mode shapenet_car  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0
```

If want to debug the model first, reduce the number of gpus to 1 and batch size to 4 via:

```bash
--gpus=1 --batch=4
```

## Inference

### Inference on a pretrained model for visualization

- Download pretrained model from [here](https://drive.google.com/drive/folders/1oJ-FmyVYjIwBZKDAQ4N1EEcE9dJjumdW?usp=sharing).
- Inference could operate on a single GPU with 16 GB memory.

```bash
python train_3d.py --outdir=save_inference_results/shapenet_car  --gpus=1 --batch=4 --gamma=40 --data_camera_mode shapenet_car  --dmtet_scale 1.0  --use_shapenet_split 1  --one_3d_generator 1  --fp32 0 --inference_vis 1 --resume_pretrain MODEL_PATH
```

### Evaluation metrics

##### Compute FID

- To evaluate the model with FID metric, add one option to the inference
  command: `--inference_compute_fid 1`

##### Compute COV & MMD scores for LFD & CD

- First generate 3D objects for evaluation, add one option to the inference
  command: `--inference_generate_geo 1`
- Following [README](./evaluation_scripts/README.md) to compute metrics.

