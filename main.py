import os
import json
import re
import dnnlib

from training import training_loop
from torch_utils import training_stats
import click

def init_dataset_kwargs(data):
    try:
        dataset_kwargs = dnnlib.EasyDict(class_name='training.dataset.ImageFolderDataset', path=data, use_labels=True, max_size=None, xflip=False)
        dataset_obj = dnnlib.util.construct_class_by_name(**dataset_kwargs) # Subclass of training.dataset.Dataset.
        dataset_kwargs.resolution = dataset_obj.resolution # Be explicit about resolution.
        dataset_kwargs.use_labels = dataset_obj.has_labels # Be explicit about labels.
        dataset_kwargs.max_size = len(dataset_obj) # Be explicit about dataset size.
        return dataset_kwargs, dataset_obj.name
    except IOError as err:
        raise click.ClickException(f'--data: {err}')

def launch_training(c, desc, outdir, rank=0):
    # Pick output directory.
    prev_run_dirs = []
    if os.path.isdir(outdir):
        prev_run_dirs = [x for x in os.listdir(outdir) if os.path.isdir(os.path.join(outdir, x))]

    matching_dirs = [re.fullmatch(r'\d{5}' + f'-{desc}', x) for x in prev_run_dirs if re.fullmatch(r'\d{5}' + f'-{desc}', x) is not None]
    if c.restart_every > 0 and len(matching_dirs) > 0:  # expect unique desc, continue in this directory
        assert len(matching_dirs) == 1, f'Multiple directories found for resuming: {matching_dirs}'
        c.run_dir = os.path.join(outdir, matching_dirs[0].group())
    else:                     # fallback to standard
        prev_run_ids = [re.match(r'^\d+', x) for x in prev_run_dirs]
        prev_run_ids = [int(x.group()) for x in prev_run_ids if x is not None]
        cur_run_id = max(prev_run_ids, default=-1) + 1
        c.run_dir = os.path.join(outdir, f'{cur_run_id:05d}-{desc}')
        assert not os.path.exists(c.run_dir)


    # Print options.
    print()
    print('Training options:')
    print(json.dumps(c, indent=2))
    print()
    print(f'Output directory:    {c.run_dir}')
    print(f'Number of GPUs:      {c.num_gpus}')
    print(f'Batch size:          {c.batch_size} images')
    print(f'Training duration:   {c.total_kimg} kimg')
    print(f'Dataset path:        {c.training_set_kwargs.path}')
    print(f'Dataset size:        {c.training_set_kwargs.max_size} images')
    print(f'Dataset resolution:  {c.training_set_kwargs.resolution}')
    print(f'Dataset labels:      {c.training_set_kwargs.use_labels}')
    print(f'Dataset x-flips:     {c.training_set_kwargs.xflip}')
    print()

    # Create output directory.
    print('Creating output directory...')
    os.makedirs(c.run_dir, exist_ok=c.restart_every > 0)
    with open(os.path.join(c.run_dir, 'training_options.json'), 'wt+') as f:
        json.dump(c, f, indent=2)

    # Start training
    dnnlib.util.Logger(file_name=os.path.join(c.run_dir, 'log.txt'), file_mode='a', should_flush=False)
    sync_device = torch.device('cuda', rank) if c.num_gpus > 1 else None
    training_stats.init_multiprocessing(rank=rank, sync_device=sync_device)
    training_loop.training_loop(rank=rank, **c)

def train(**kwargs):
    # Initialize config.
    opts = dnnlib.EasyDict(kwargs) # Command line arguments.
    c = dnnlib.EasyDict() # Main config dict.
    c.G_kwargs = dnnlib.EasyDict(class_name=None, z_dim=64, w_dim=128, mapping_kwargs=dnnlib.EasyDict())
    c.G_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.D_opt_kwargs = dnnlib.EasyDict(class_name='torch.optim.Adam', betas=[0,0.99], eps=1e-8)
    c.data_loader_kwargs = dnnlib.EasyDict(pin_memory=True, prefetch_factor=2)

    # Training set.
    c.training_set_kwargs, dataset_name = init_dataset_kwargs(data=opts.data)
    if opts.cond and not c.training_set_kwargs.use_labels:
        raise ValueError('--cond=True requires labels specified in dataset.json')
    c.training_set_kwargs.use_labels = opts.cond
    c.training_set_kwargs.xflip = opts.mirror

    # Hyperparameters & settings.
    c.num_gpus = opts.gpus
    c.batch_size = opts.batch
    c.batch_gpu = opts.batch_gpu or opts.batch // opts.gpus
    c.G_kwargs.channel_base = opts.cbase
    c.G_kwargs.channel_max = opts.cmax
    c.G_kwargs.mapping_kwargs.num_layers = 2
    c.G_opt_kwargs.lr = (0.002 if opts.cfg == 'stylegan2' else 0.0025) if opts.glr is None else opts.glr
    c.D_opt_kwargs.lr = opts.dlr
    c.total_kimg = opts.kimg
    c.kimg_per_tick = opts.tick
    c.image_snapshot_ticks = c.network_snapshot_ticks = opts.snap
    c.random_seed = c.training_set_kwargs.random_seed = opts.seed
    c.data_loader_kwargs.num_workers = opts.workers

    # Sanity checks.
    if c.batch_size % c.num_gpus != 0:
        raise ValueError('--batch must be a multiple of --gpus')
    if c.batch_size % (c.num_gpus * c.batch_gpu) != 0:
        raise ValueError('--batch must be a multiple of --gpus times --batch-gpu')
  
    # Base configuration.
    c.ema_kimg = c.batch_size * 10 / 32
    if opts.cfg == 'stylegan2':
        c.G_kwargs.class_name = 'pg_modules.networks_stylegan2.Generator'
        c.G_kwargs.fused_modconv_default = 'inference_only' # Speed up training by using regular convolutions instead of grouped convolutions.
        use_separable_discs = True

    elif opts.cfg == 'fastgan':
        c.G_kwargs = dnnlib.EasyDict(class_name='pg_modules.networks_fastgan.Generator', cond=opts.cond)
        c.G_opt_kwargs.lr = c.D_opt_kwargs.lr = 0.0002
        use_separable_discs = False

    # Restart.
    c.restart_every = opts.restart_every

    # Description string.
    desc = f'{opts.cfg:s}-{dataset_name:s}-gpus{c.num_gpus:d}-batch{c.batch_size:d}'
    if opts.desc is not None:
        desc += f'-{opts.desc}'

    # Projected and Multi-Scale Discriminators
    c.loss_kwargs = dnnlib.EasyDict(class_name='training.loss.ProjectedGANLoss')
    c.D_kwargs = dnnlib.EasyDict(
        class_name='pg_modules.discriminator.ProjectedDiscriminator',
        diffaug=True,
        interp224=(c.training_set_kwargs.resolution < 224),
        backbone_kwargs=dnnlib.EasyDict(),
    )

    c.D_kwargs.backbone_kwargs.cout = 64
    c.D_kwargs.backbone_kwargs.expand = True
    c.D_kwargs.backbone_kwargs.proj_type = 2
    c.D_kwargs.backbone_kwargs.num_discs = 4
    c.D_kwargs.backbone_kwargs.separable = use_separable_discs
    c.D_kwargs.backbone_kwargs.cond = opts.cond

    # Launch.
    launch_training(c=c, desc=desc, outdir=opts.outdir)



# start training!

train(
    outdir='training-runs', 
    cfg='fastgan',
    data='C:/Users/sr/Pictures/projectedgan/projected_gan/art-painting', 
    gpus=1, 
    batch=1, 
    cond=False, 
    mirror=1, 
    batch_gpu=1, 
    cbase=32768, 
    cmax=512, 
    glr=None, 
    dlr=0.002, 
    desc='', 
    metrics=[],
    kimg=10000, 
    tick=4, 
    snap=1, 
    seed=0, 
    workers=0,
    restart_every=999999,
    
)