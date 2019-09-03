from torch.optim import Adam, SGD
from torch.utils.data import DataLoader
import tqdm


class GANTrainer(object):
    def __init__(self, gen, discr, g_data, d_data, batch_size, lr, g_loss, d_loss, discr_train_steps=5):
        self.gen = gen
        self.discr = discr
        self.batch_size = batch_size
        self.g_opt = Adam(gen.parameters(), lr=lr)
        self.d_opt = SGD(discr.parameters(), lr=lr)
        self.g_loss = g_loss
        self.d_loss = d_loss
        self.cur_epoch = 0
        self.g_dataloader = DataLoader(g_data, num_workers=4, batch_size=self.batch_size)
        self.d_dataloader = DataLoader(d_data, num_workers=4, batch_size=self.batch_size)
        self.d_train_steps = discr_train_steps

    def iterate(self):
        for fake, real in tqdm.tqdm(zip(self.g_dataloader, self.d_dataloader), desc=f"Training epoch {self.cur_epoch}"):
            for i in range(self.d_train_steps):
                fake = self.gen(fake)
                loss = self.d_loss(fake, real)
                self.d_opt.zero_grad()
                loss.backward()
                self.d_opt.step()

            loss = self.g_loss(fake)
            self.g_opt.zero_grad()
            loss.backward()
            self.g_opt.step()

    def run(self, n_epochs):
        for i in range(n_epochs):
            self.cur_epoch = i
            self.iterate()
