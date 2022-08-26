import paddle
import paddlers
from paddlers.tasks.change_detector import BaseChangeDetector

from attach_tools import Attach

attach = Attach.to(paddlers.tasks.change_detector)


def make_trainer(net_type, *args, **kwargs):
    def _init_func(self,
                   num_classes=2,
                   use_mixed_loss=False,
                   losses=None,
                   **params):
        super().__init__(
            model_name=net_type.__name__,
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)

    if not issubclass(net_type, paddle.nn.Layer):
        raise TypeError("net must be a subclass of paddle.nn.Layer")

    trainer_name = net_type.__name__

    trainer_type = type(trainer_name, (BaseChangeDetector, ),
                        {'__init__': _init_func})

    return trainer_type(*args, **kwargs)


@attach
class CustomTrainer(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 in_channels=3,
                 att_types='ct',
                 use_dropout=False,
                 **params):
        params.update({
            'in_channels': in_channels,
            'att_types': att_types,
            'use_dropout': use_dropout
        })
        super().__init__(
            model_name='CustomModel',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)
