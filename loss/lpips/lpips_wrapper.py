from torch import Tensor
from torchvision import transforms
from loss.lpips import LPIPS


class LPIPSWrapper(LPIPS):
    def __init__(self, net_type: str = 'alex', version: str = '0.1'):
        super(LPIPSWrapper, self).__init__(net_type, version)
        net_size = {
            'alex': 224,
            'vgg16': 224,
            'squeeze': 224
        }
        size = net_size[net_type]
        self.resize = transforms.Resize([size, size])

    def forward(self, x: Tensor, y: Tensor):
        return super(LPIPSWrapper, self).forward(self.resize(x), self.resize(y))
