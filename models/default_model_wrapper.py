from torch import nn, Tensor


class ImageEncoderWrapper(nn.Module):
    def __init__(self, inner: nn.Module):
        super(ImageEncoderWrapper, self).__init__()
        self.inner = inner
        del self.inner.classifier[-1] # Assuming this will be VGG. Removing classification layer
        del self.inner.classifier[-1] # Assuming this will be VGG. Removing dropout layer
    
    def forward (self, x: Tensor) -> Tensor:
        return self.inner(x)
    
    def encode_image(self, x: Tensor) -> Tensor:
        return self.inner(x)
