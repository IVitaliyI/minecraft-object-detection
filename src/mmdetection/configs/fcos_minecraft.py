_base_ = [
    "mmdet::fcos/fcos_r50-caffe_fpn_gn-head_1x_coco.py",
]

from src.constants import MINECRAFT_CLASS_COLORS_RGB

model = dict(bbox_head=dict(num_classes=len(MINECRAFT_CLASS_COLORS_RGB.keys())))


dataset_type = "CocoDataset"
data_root = "datasets/minecraft/"
metainfo = dict(
    classes=(tuple(MINECRAFT_CLASS_COLORS_RGB.keys())),
    palette=[color for color in MINECRAFT_CLASS_COLORS_RGB.values()],
)
backend_args = None

train_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, poly2mask=False),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(type="RandomFlip", prob=0.5),
    dict(type="PackDetInputs"),
]

test_pipeline = [
    dict(type="LoadImageFromFile", backend_args=backend_args),
    dict(type="LoadAnnotations", with_bbox=True, poly2mask=False),
    dict(type="Resize", scale=(512, 512), keep_ratio=True),
    dict(type="PackDetInputs"),
]

train_dataloader = dict(
    batch_size=8,
    num_workers=2,
    persistent_workers=True,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="train/annotations.json",
        data_prefix=dict(img="train/"),
        pipeline=train_pipeline,
        backend_args=backend_args,
    ),
)

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=True),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="valid/annotations.json",
        data_prefix=dict(img="valid/"),
        pipeline=test_pipeline,
        backend_args=backend_args,
    ),
)

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type="DefaultSampler", shuffle=False),
    dataset=dict(
        type=dataset_type,
        data_root=data_root,
        metainfo=metainfo,
        ann_file="test/annotations.json",
        data_prefix=dict(img="test/"),
        pipeline=test_pipeline,
        test_mode=True,
        backend_args=backend_args,
    ),
)

val_evaluator = dict(  # Validation evaluator config
    type="CocoMetric",  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + "valid/" + "annotations.json",  # Annotation file path
    metric=[
        "bbox",
    ],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    backend_args=backend_args,
    outfile_prefix="src/artifacts/metrics/valid",
)

test_evaluator = dict(
    type="CocoMetric",  # The coco metric used to evaluate AR, AP, and mAP for detection and instance segmentation
    ann_file=data_root + "test/" + "annotations.json",  # Annotation file path
    metric=[
        "bbox",
    ],  # Metrics to be evaluated, `bbox` for detection and `segm` for instance segmentation
    format_only=False,
    backend_args=backend_args,
    outfile_prefix="src/artifacts/metrics/test",
)

train_cfg = dict(
    type="EpochBasedTrainLoop",  # The training loop type. Refer to https://github.com/open-mmlab/mmengine/blob/main/mmengine/runner/loops.py
    max_epochs=12,  # Maximum training epochs
    val_interval=1,
)
val_cfg = dict(type="ValLoop")  # The validation loop type
test_cfg = dict(type="TestLoop")  # The testing loop type

optim_wrapper = dict(  # Optimizer wrapper config
    type="OptimWrapper",  # Optimizer wrapper type, switch to AmpOptimWrapper to enable mixed precision training.
    optimizer=dict(  # Optimizer config. Support all kinds of optimizers in PyTorch. Refer to https://pytorch.org/docs/stable/optim.html#algorithms
        type="SGD",  # Stochastic gradient descent optimizer
        lr=0.02,  # The base learning rate
        momentum=0.9,  # Stochastic gradient descent with momentum
        weight_decay=0.0001,
    ),  # Weight decay of SGD
    clip_grad=None,  # Gradient clip option. Set None to disable gradient clip. Find usage in https://mmengine.readthedocs.io/en/latest/tutorials/optimizer.html
)

param_scheduler = [
    # Linear learning rate warm-up scheduler
    dict(
        type="LinearLR",  # Use linear policy to warmup learning rate
        start_factor=0.001,  # The ratio of the starting learning rate used for warmup
        by_epoch=False,  # The warmup learning rate is updated by iteration
        begin=0,  # Start from the first iteration
        end=500,
    ),  # End the warmup at the 500th iteration
    # The main LRScheduler
    dict(
        type="MultiStepLR",  # Use multi-step learning rate policy during training
        by_epoch=True,  # The learning rate is updated by epoch
        begin=0,  # Start from the first epoch
        end=12,  # End at the 12th epoch
        milestones=[8, 11],  # Epochs to decay the learning rate
        gamma=0.1,
    ),  # The learning rate decay ratio
]

default_hooks = dict(
    timer=dict(type="IterTimerHook"),
    param_scheduler=dict(type="ParamSchedulerHook"),
    checkpoint=dict(type="CheckpointHook", interval=1),
    sampler_seed=dict(type="DistSamplerSeedHook"),
    visualization=dict(type="DetVisualizationHook"),
)

default_scope = "mmdet"  # The default registry scope to find modules. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/registry.html

env_cfg = dict(
    cudnn_benchmark=False,  # Whether to enable cudnn benchmark
    mp_cfg=dict(  # Multi-processing config
        mp_start_method="fork",  # Use fork to start multi-processing threads. 'fork' usually faster than 'spawn' but maybe unsafe. See discussion in https://github.com/pytorch/pytorch/issues/1355
        opencv_num_threads=0,
    ),  # Disable opencv multi-threads to avoid system being overloaded
    dist_cfg=dict(backend="nccl"),  # Distribution configs
)

vis_backends = [
    dict(type="LocalVisBackend")
]  # Visualization backends. Refer to https://mmengine.readthedocs.io/en/latest/advanced_tutorials/visualization.html
visualizer = dict(
    type="DetLocalVisualizer", vis_backends=vis_backends, name="visualizer"
)
log_processor = dict(
    type="LogProcessor",  # Log processor to process runtime logs
    window_size=50,  # Smooth interval of log values
    by_epoch=True,
)  # Whether to format logs with epoch type. Should be consistent with the train loop's type.

log_level = "INFO"  # The level of logging.
load_from = None  # Load model checkpoint as a pre-trained model from a given path. This will not resume training.
resume = False  # Whether to resume from the checkpoint defined in `load_from`. If `load_from` is None, it will resume the latest checkpoint in the `work_dir`.
