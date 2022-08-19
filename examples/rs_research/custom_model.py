import paddle
import paddle.nn as nn
import paddle.nn.functional as F
import paddlers
from paddlers.rs_models.cd import BIT
from attach_tools import Attach

attach = Attach.to(paddlers.rs_models.cd)


@attach
class IterativeBIT(BIT):
    def __init__(self,
                 num_iters=1,
                 feat_channels=32,
                 num_classes=2,
                 bit_kwargs=None):
        if num_iters <= 0:
            raise ValueError(
                f"`num_iters` should have positive value, but got {num_iters}.")

        self.num_iters = num_iters

        if bit_kwargs is None:
            bit_kwargs = dict()

        if 'num_classes' in bit_kwargs:
            raise KeyError("'num_classes' should not be set in `bit_kwargs`.")
        bit_kwargs['num_classes'] = num_classes

        super().__init__(**bit_kwargs)

        self.conv_fuse = nn.Sequential(
            nn.Conv2D(feat_channels + 1, feat_channels, 1), nn.Sigmoid())

    def forward(self, t1, t2):
        # Extract features via shared backbone.
        x1 = self.backbone(t1)
        x2 = self.backbone(t2)

        # Tokenization
        if self.use_tokenizer:
            token1 = self._get_semantic_tokens(x1)
            token2 = self._get_semantic_tokens(x2)
        else:
            token1 = self._get_reshaped_tokens(x1)
            token2 = self._get_reshaped_tokens(x2)

        # Transformer encoder forward
        token = paddle.concat([token1, token2], axis=1)
        token = self.encode(token)
        token1, token2 = paddle.chunk(token, 2, axis=1)

        # Get initial rate map
        rate_map = self._init_rate_map(x1.shape)

        for it in range(self.num_iters):
            # Construct inputs
            x1_iter = self._constr_iter_input(x1, rate_map)
            x2_iter = self._constr_iter_input(x2, rate_map)

            # Transformer decoder forward
            y1 = self.decode(x1_iter, token1)
            y2 = self.decode(x2_iter, token2)

            # Feature differencing
            y = paddle.abs(y1 - y2)

            # Construct rate map
            rate_map = self._constr_rate_map(y)

        y = self.upsample(y)
        pred = self.conv_out(y)

        return [pred]

    def _init_rate_map(self, im_shape):
        b, _, h, w = im_shape
        return paddle.full((b, 1, h, w), 0.5)

    def _constr_iter_input(self, x, rate_map):
        return self.conv_fuse(paddle.concat([x, rate_map], axis=1))

    def _constr_rate_map(self, x):
        rate_map = x.mean(1, keepdim=True).detach()  # Cut off gradient workflow
        # min-max normalization
        rate_map -= rate_map.min()
        rate_map /= rate_map.max()
        return rate_map
