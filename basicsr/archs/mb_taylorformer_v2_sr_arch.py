import os, sys
import torch
import torch.nn as nn
from basicsr.utils.registry import ARCH_REGISTRY

# Thêm đường dẫn tới repo MB-TaylorFormerV2
MBTF2_PATH = os.path.abspath('/content/mb-taylorformerv2')
if MBTF2_PATH not in sys.path:
    sys.path.insert(0, MBTF2_PATH)

# TODO: import đúng lớp backbone từ repo MBTFv2 của bạn.
# Tên module cụ thể có thể khác nhau giữa các commit; kiểm tra __init__.py / models.
# Ví dụ (giả định):
# from models.mbtaylorformer_v2 import MBTaylorFormerV2Backbone
# Nếu repo của bạn khác layout, cập nhật import cho đúng.

# Trong lúc bạn kiểm tra chính xác class name, giữ một backbone đơn giản để file hợp lệ:
class FallbackBackbone(nn.Module):
    def __init__(self, dim=180):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(dim, dim, 3, 1, 1), nn.GELU(),
            nn.Conv2d(dim, dim, 3, 1, 1)
        )
    def forward(self, x): return self.net(x)

def pixelshuffle_upsampler(in_ch, out_ch, scale):
    layers = []
    r = scale
    layers += [nn.Conv2d(in_ch, in_ch * (r ** 2), 3, 1, 1)]
    layers += [nn.PixelShuffle(r)]
    layers += [nn.Conv2d(in_ch, out_ch, 3, 1, 1)]
    return nn.Sequential(*layers)

@ARCH_REGISTRY.register()
class MBTaylorFormerV2SR(nn.Module):
    """
    MB-TaylorFormer V2 cho Super-Resolution x2 (Paired SR).
    Head -> (MBTFv2 backbone) -> Tail(PixelShuffle x2).
    """
    def __init__(self, scale=2, in_ch=3, out_ch=3, dim=180, **kwargs):
        super().__init__()
        assert scale == 2, "Config này set cho x2; muốn x3/x4 cần chỉnh tail."
        self.head = nn.Conv2d(in_ch, dim, 3, 1, 1)

        # TODO: thay FallbackBackbone bằng backbone thật từ repo MBTFv2
        # self.body = MBTaylorFormerV2Backbone(embed_dim=dim, **kwargs)
        self.body = FallbackBackbone(dim=dim)

        self.tail = pixelshuffle_upsampler(dim, out_ch, scale)

    def forward(self, x):
        x = self.head(x)
        x = self.body(x)
        x = self.tail(x)
        return x
