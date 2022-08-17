import paddlers
from paddlers.tasks.change_detector import BaseChangeDetector

from attach_tools import Attach

attach = Attach.to(paddlers.tasks.change_detector)


@attach
class IterativeBIT(BaseChangeDetector):
    def __init__(self,
                 num_classes=2,
                 use_mixed_loss=False,
                 losses=None,
                 num_iters=1,
                 gamma=0.1,
                 bit_kwargs=None,
                 **params):
        params.update({
            'num_iters': num_iters,
            'gamma': gamma,
            'bit_kwargs': bit_kwargs
        })
        super().__init__(
            model_name='IterativeBIT',
            num_classes=num_classes,
            use_mixed_loss=use_mixed_loss,
            losses=losses,
            **params)
