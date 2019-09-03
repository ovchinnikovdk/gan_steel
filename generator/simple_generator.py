import torch


class Reshape(torch.nn.Module):
    def __init__(self, shape):
        torch.nn.Module.__init__(self)
        self.shape = shape

    def forward(self, input):
        return input.view(self.shape)


class SimpleGenerator(torch.nn.Module):
    def __init__(self, input_size=256, output_size=128, kernel_size=15):
        super(SimpleGenerator, self).__init__()
        self.input_size = input_size
        self.output_shape = output_size
        self.kernel_size = kernel_size
        self.model = torch.nn.Sequential(
            torch.nn.Linear(self.input_size, 6 * 8 * 8),
            Reshape([-1, 6, 8, 8]))
        for i in range((output_size - 8) // ((kernel_size // 2) * 2)):
            self.model.add_module('upsample_block_' + str(i),
                                  torch.nn.Sequential(torch.nn.ConvTranspose2d(6, 6, kernel_size=kernel_size),
                                                      torch.nn.ReLU()))
        print(self.model)

    def forward(self, input):
        return self.model(input)
