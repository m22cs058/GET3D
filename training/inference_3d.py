import copy
import os
import numpy as np
import torch
import dnnlib
from torch_utils.ops import conv2d_gradfix, grid_sample_gradfix
from metrics import metric_main


def clean_training_set_kwargs_for_metrics(training_set_kwargs):
    if 'add_camera_cond' in training_set_kwargs:
        training_set_kwargs['add_camera_cond'] = True
    return training_set_kwargs

def initialize_torch_settings(random_seed, num_gpus, rank):
    np.random.seed(random_seed * num_gpus + rank)
    torch.manual_seed(random_seed * num_gpus + rank)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    conv2d_gradfix.enabled = True
    grid_sample_gradfix.enabled = True

def generate_and_save_visualizations(G_ema, device, run_dir, grid_z, grid_c, grid_tex_z=None):
    grid_size = (5, 5)
    n_shape = grid_size[0] * grid_size[1]
    print('==> generate ')
    save_visualization(
        G_ema, grid_z, grid_c, run_dir, 0, grid_size, 0,
        save_all=False,
        grid_tex_z=grid_tex_z
    )

def generate_textured_mesh_for_inference(G_ema, device, run_dir, grid_z, grid_c, grid_tex_z=None):
    print('==> generate inference 3d shapes with texture')
    save_textured_mesh_for_inference(
        G_ema, grid_z, grid_c, run_dir, save_mesh_dir='texture_mesh_for_inference',
        c_to_compute_w_avg=None, grid_tex_z=grid_tex_z
    )

def generate_interpolation_results(G_ema, run_dir):
    print('==> generate interpolation results')
    save_visualization_for_interpolation(G_ema, save_dir=os.path.join(run_dir, 'interpolation'))

def compute_fid_scores(metrics, G_ema, training_set_kwargs, num_gpus, rank, device, run_dir, resume_pretrain):
    print('==> compute FID scores for generation')
    for metric in metrics:
        training_set_kwargs = clean_training_set_kwargs_for_metrics(training_set_kwargs)
        training_set_kwargs['split'] = 'test'
        result_dict = metric_main.calc_metric(
            metric=metric, G=G_ema,
            dataset_kwargs=training_set_kwargs, num_gpus=num_gpus, rank=rank,
            device=device
        )
        metric_main.report_metric(result_dict, run_dir=run_dir, snapshot_pkl=resume_pretrain)

def generate_shapes_for_evaluation(G_ema, run_dir):
    print('==> generate 7500 shapes for evaluation')
    save_geo_for_inference(G_ema, run_dir)

def inference(
        run_dir='.', training_set_kwargs={}, G_kwargs={}, D_kwargs={}, metrics=[],
        random_seed=0, num_gpus=1, rank=0, inference_vis=False,
        inference_to_generate_textured_mesh=False, resume_pretrain=None,
        inference_save_interpolation=False, inference_compute_fid=False,
        inference_generate_geo=False, **dummy_kawargs
):
    from torch_utils.ops import upfirdn2d, bias_act, filtered_lrelu

    upfirdn2d._init()
    bias_act._init()
    filtered_lrelu._init()

    device = torch.device('cuda', rank)
    initialize_torch_settings(random_seed, num_gpus, rank)

    common_kwargs = dict(
        c_dim=0, img_resolution=training_set_kwargs.get('resolution', 1024), img_channels=3
    )
    G_kwargs['device'] = device

    G = dnnlib.util.construct_class_by_name(**G_kwargs, **common_kwargs).train().requires_grad_(False).to(device)
    G_ema = copy.deepcopy(G).eval()

    if resume_pretrain is not None and rank == 0:
        print('==> resume from pretrained path %s' % resume_pretrain)
        model_state_dict = torch.load(resume_pretrain, map_location=device)
        G.load_state_dict(model_state_dict['G'], strict=True)
        G_ema.load_state_dict(model_state_dict['G_ema'], strict=True)

    grid_z = torch.randn([25, G.z_dim], device=device).split(1)
    grid_tex_z = torch.randn([25, G.z_dim], device=device).split(1)
    grid_c = torch.ones(25, device=device).split(1)

    generate_and_save_visualizations(G_ema, device, run_dir, grid_z, grid_c, grid_tex_z)

    if inference_to_generate_textured_mesh:
        generate_textured_mesh_for_inference(G_ema, device, run_dir, grid_z, grid_c, grid_tex_z)

    if inference_save_interpolation:
        generate_interpolation_results(G_ema, run_dir)

    if inference_compute_fid:
        compute_fid_scores(metrics, G_ema, training_set_kwargs, num_gpus, rank, device, run_dir, resume_pretrain)

    if inference_generate_geo:
        generate_shapes_for_evaluation(G_ema, run_dir)
