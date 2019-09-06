import torch
from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import tqdm
import matplotlib.pyplot as plt
import numpy as np


class GANTrainer(object):
    def __init__(self, gen, discr, d_data, batch_size, lr, g_loss, d_loss, discr_train_steps=3):
        self.gen = gen
        self.discr = discr
        self.batch_size = batch_size
        self.g_opt = Adam(gen.parameters(), lr=lr)
        self.d_opt = SGD(discr.parameters(), lr=lr)
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.cur_epoch = 0
        self.d_dataloader = DataLoader(d_data, num_workers=4, batch_size=self.batch_size)
        self.d_train_steps = discr_train_steps
        self.gpu = torch.cuda.is_available()
        if self.gpu:
            self.discr.cuda()
            self.gen.cuda()

    def iterate(self):
        self.d_opt.zero_grad()
        self.g_opt.zero_grad()
        losses = []
        for real in tqdm.tqdm(self.d_dataloader,
                              desc=f"Training epoch {self.cur_epoch}"):
            fake = torch.randn(real.shape[0], 256)
            if self.gpu:
                fake = fake.cuda()
                real = real.cuda()

            for i in range(self.d_train_steps):
                gen_fake = self.gen(fake)
                disc_real = self.discr(real)
                disc_fake = self.discr(gen_fake)
                d_loss = self.d_loss(disc_fake, disc_real)
                d_loss.backward()
                self.d_opt.step()
                self.d_opt.zero_grad()
            gen_fake = self.gen(fake)
            disc_fake = self.discr(gen_fake)
            g_loss = self.g_loss(disc_fake)
            losses.append(g_loss.item())
            g_loss.backward()
            self.g_opt.step()
            self.g_opt.zero_grad()
        print(np.mean(losses))

    def run(self, n_epochs):
        for i in range(n_epochs):
            self.cur_epoch = i
            self.iterate()
            if (i + 1) % 10 == 0:
                self.sample_images(3, 3, False)

    def sample_images(self, nrow, ncol, sharp=False):
        images = self.gen(torch.randn(nrow * ncol, self.gen.input_size).cuda())
        images = images.data.cpu().numpy().transpose([0, 2, 3, 1])
        #         if np.var(images) != 0:
        #             images = images.clip(np.min(data),np.max(data))
        for i in range(nrow * ncol):
            plt.subplot(nrow, ncol, i + 1)
            if sharp:
                plt.imshow(images[i], cmap="gray", interpolation="none")
            else:
                plt.imshow(images[i], cmap="gray")
        plt.show()