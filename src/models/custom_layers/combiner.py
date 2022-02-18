from torch import nn


class Combined(nn.Module):
    def __init__(self, first, second):
        super().__init__()

        self.first = first
        self.second = second

    def forward(self, x, *argv):
        return self.second(self.first(x), *argv)


# class AutoEncoder_adaptive(nn.Module):
#     def __init__(self, encoder, decoder):
#         super(AutoEncoder_adaptive, self).__init__()

#         self.encoder = encoder
#         self.decoder = decoder

#     def forward(self, x, alpha=1):
#         return self.decoder(self.encoder(x), alpha)
