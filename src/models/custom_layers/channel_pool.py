import torch
import torch.nn.functional as F


class ChannelPool(torch.nn.MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w*h).permute(0, 2, 1)
        pooled = F.max_pool1d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)
