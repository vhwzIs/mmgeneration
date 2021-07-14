_base_ = [
    '../_base_/models/dcgan_64x64.py',
    '../_base_/datasets/unconditional_imgs_cifa10.py',
    '../_base_/default_runtime.py'
]

# output single channel
model = dict(generator=dict(out_channels=3), discriminator=dict(in_channels=3))

# you must set `samples_per_gpu` and `imgs_root`
data = dict(
    samples_per_gpu=256,
    workers_per_gpu=2,
)

# adjust running config
lr_config = None
checkpoint_config = dict(interval=500, by_epoch=False)
custom_hooks = [
    dict(
        type='VisualizeUnconditionalSamples',
        output_dir='training_samples',
        interval=100)
]

log_config = dict(
    interval=100, hooks=[
        dict(type='TextLoggerHook'),
    ])

total_iters = 10000
load_from = 'work_dirs/dcgan/ckpt/iter_5000.pth' 
# metrics = dict(
#     ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
#     swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 64, 64)))

metrics = dict(
    fid50k=dict(
        type='FID',
        num_images=10000,
        inception_pkl='work_dirs/inception_pkl/cifar10test.pkl',
        bgr2rgb=True)
    # ms_ssim10k=dict(type='MS_SSIM', num_images=10000),
    # swd16k=dict(type='SWD', num_images=16384, image_shape=(3, 64, 64))
    )

optimizer = dict(
    generator=dict(type='Adam', lr=0.0004, betas=(0.5, 0.999)),
    discriminator=dict(type='Adam', lr=0.0001, betas=(0.5, 0.999)))