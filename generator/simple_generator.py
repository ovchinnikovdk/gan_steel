import torch


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        torch.nn.Module.__init__(self)
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class SimpleGenerator(torch.nn.Module):
    def __init__(self, input_size=256, output_size=128, kernel_size=15, _print=False):
        super(SimpleGenerator, self).__init__()
        self.input_size = input_size
        self.output_shape = output_size
        self.kernel_size = kernel_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 10*8*8),
            torch.nn.ELU(),
            Reshape([-1, 10, 8, 8]),
            torch.nn.ConvTranspose2d(10, 64, kernel_size=(5,5)),
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(64, 64, kernel_size=(5,5)),
            torch.nn.ELU(),
            torch.nn.Upsample(scale_factor=2),
            torch.nn.ConvTranspose2d(64, 32, kernel_size=(5,5)),
            torch.nn.ELU(),
            torch.nn.ConvTranspose2d(32, 32, kernel_size=(5,5)),
            torch.nn.Conv2d(32, 3, kernel_size=(5,5))
        )
        if _print:
            print(self.model)

    def forward(self, input):
        return self.model(input)