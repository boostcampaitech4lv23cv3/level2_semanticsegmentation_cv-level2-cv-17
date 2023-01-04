# yapf:disable
log_config = dict(
    interval=50,
    hooks=[
        dict(type="TextLoggerHook", by_epoch=False),
        # dict(type='MlflowLoggerHook'),
        # dict(type='TensorboardLoggerHook'),
        dict(
            type="MMSegWandbHook",  # "WandbLoggerHook",
            init_kwargs={
                "project": "Semantic Segmentation",
                "entity": "boostcamp-ai-tech-4-cv-17",
                # "name": "mmseg",
            },
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100,
        ),
    ],
)
# yapf:enable

dist_params = dict(backend="nccl")
log_level = "INFO"
load_from = None
resume_from = None

workflow = [("train", 1)]
# workflow = [("train", 1), ("val", 1)]

cudnn_benchmark = True
