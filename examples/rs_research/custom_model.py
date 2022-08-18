import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlers
from paddlers.rs_models.cd import BIT
from attach_tools import Attach

attach = Attach.to(paddlers.rs_models.cd)


@attach
class IterativeBIT(nn.Layer):
    def __init__(self, num_iters=1, gamma=0.1, num_classes=2, bit_kwargs=None):
        super().__init__()

        if num_iters <= 0:
            raise ValueError(
                f"`num_iters` should have positive value, but got {num_iters}.")

        self.num_iters = num_iters
        self.gamma = gamma

        if bit_kwargs is None:
            bit_kwargs = dict()

        if 'num_classes' in bit_kwargs:
            raise KeyError("'num_classes' should not be set in `bit_kwargs`.")
        bit_kwargs['num_classes'] = num_classes

        self.bit = BIT(**bit_kwargs)

    def forward(self, t1, t2):
        rate_map = self._init_rate_map(t1.shape)

        for it in range(self.num_iters):
            # Construct inputs
            x1 = self._constr_iter_input(t1, rate_map)
            x2 = self._constr_iter_input(t2, rate_map)
            # Get logits
            logits_list = self.bit(x1, x2)
            # Construct rate map
            prob_map = F.softmax(logits_list[0], axis=1)
            rate_map = self._constr_rate_map(prob_map)

        return logits_list

    def _constr_iter_input(self, im, rate_map):
        return paddle.concat([im, rate_map], axis=1)

    def _init_rate_map(self, im_shape):
        b, _, h, w = im_shape
        return paddle.zeros((b, 1, h, w))

    def _constr_rate_map(self, prob_map):
        if prob_map.shape[1] != 2:
            raise ValueError(
                f"`prob_map.shape[1]` must be 2, but got {prob_map.shape[1]}.")
        return (prob_map[:, 1:2] * self.gamma)
