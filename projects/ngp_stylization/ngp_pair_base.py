sampler = dict(
    type='DensityGridSampler_two',
    update_den_freq=16,
)
encoder = dict(
    pos_encoder = dict(
        type='HashEncoder_two',
    ),
    dir_encoder = dict(
        type='SHEncoder',
    ),
)
model = dict(
    type='NGPNetworks_two',
    use_fully=True,
)
loss = dict(
    type='HuberLoss',
    delta=0.1,
)
optim = dict(
    type='Adam',
    lr=1e-1,
    eps=1e-15,
    betas=(0.9,0.99),
)
ema = dict(
    type='EMA',
    decay=0.95,
)
expdecay=dict(
    type='ExpDecay',
    decay_start=40_000,
    decay_interval=20_000,
    decay_base=0.33,
    decay_end=None
)
con_dataset_type = 'NerfDataset'
sty_dataset_type = 'LLFFDataset'
dataset_dir_content = '/data2/lsx/nerfstyle/nerf_synthetic/'
dataset_dir_style = '/data2/lsx/nerfstyle/nerf_llff_data/'

exp_name = 'lego'
sty_name = 'fern'


dataset_content = dict(
    train=dict(
        type=con_dataset_type,
        root_dir=dataset_dir_content,
        batch_size=4096,
        mode='train',
    ),
    test=dict(
        type=con_dataset_type,
        root_dir=dataset_dir_content,
        batch_size=4096,
        mode='test',
        preload_shuffle=False,
    ),
)
dataset_style = dict(
    train=dict(
        type=sty_dataset_type,
        root_dir=dataset_dir_style,
        batch_size=4096,
        mode='train',
    ),
    test=dict(
        type=sty_dataset_type,
        root_dir=dataset_dir_style,
        batch_size=4096,
        mode='test',
        preload_shuffle=False,
    ),
)



log_dir = "./logs"
tot_train_steps = 40000
# Background color, value range from 0 to 1
background_color = [1, 1, 1]
# Hash encoding function used in Instant-NGP
hash_func = "p0 ^ p1 * 19349663 ^ p2 * 83492791"
cone_angle_constant = 0.00390625
near_distance = 0.2
n_rays_per_batch = 4096
n_training_steps = 16
# Expected number of sampling points per batch
target_batch_size = 1<<18
# Set const_dt=True for higher performance
# Set const_dt=False for faster convergence
const_dt=False
# Use fp16 for faster training
fp16 = True