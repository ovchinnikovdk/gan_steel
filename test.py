from torchvision.models import resnet18
import torch
import matplotlib.pyplot as plt
from utils.sample_utils import RealDataset
from utils.loss import GenLoss, DiscLoss
from generator.simple_generator import SimpleGenerator
from utils.trainer import  GANTrainer


def sample_images(nrow, ncol, sharp=False, generator=None, real=False, dataset=None):
    if real and dataset is not None:
        images = []
        for i in range(nrow * ncol):
            images.append(dataset[i][None])
        images = torch.cat(images, axis=0)
    else:
        images = generator(torch.randn(nrow * ncol, noise_size).cuda())
    images = images.data.cpu().numpy().transpose([0, 2, 3, 1])
    #         if np.var(images) != 0:
    #             images = images.clip(np.min(images),np.max(images))
    for i in range(nrow * ncol):
        plt.subplot(nrow, ncol, i + 1)
        if sharp:
            plt.imshow(images[i], cmap="gray", interpolation="none")
        else:
            plt.imshow(images[i], cmap="gray")
    plt.show()


path = '../input/severstal-steel-defect-detection/train_images/'
noise_size = 256
output_size = 32
batch_size = 32
lr = 1e-3
real_dataset = RealDataset(data_path=path, size=(output_size, output_size))
sample_images(5, 5, real=True, dataset=real_dataset)
generator = SimpleGenerator(input_size=noise_size, output_size=output_size, kernel_size=13, _print=False)
discriminator = resnet18(pretrained=False, num_classes=2)
# Training
trainer = GANTrainer(generator, 
                     discriminator,
                     real_dataset, 
                     batch_size, 
                     lr, 
                     GenLoss(), 
                     DiscLoss(), 
                     discr_train_steps=10)