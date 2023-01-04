# yapf:disable
_base_ = [
    'models/',
    'datasets/',
    'schedules/'
]

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook', by_epoch=False),
        dict(type='MMSegWandbHook',
            by_epoch=False,
            init_kwargs=dict(
                entity='boostcamp-ai-tech-4-cv-17',
                project='Semantic Segmentation',
                name='MMSeg'
            ),
            interval=10,
            log_checkpoint=True,
            log_checkpoint_metadata=True,
            num_eval_images=100
        )
    ]
)
# yapf:enable
dist_params = dict(backend='nccl')
log_level = 'INFO'
load_from = None
resume_from = None
workflow = [('train', 1)]
cudnn_benchmark = True
