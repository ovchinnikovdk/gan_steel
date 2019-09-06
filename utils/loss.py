import torch


class GenLoss(torch.nn.Module):
    def forward(self, fake_disc_opinion):
        logp_disc_opinion = torch.nn.LogSigmoid()(fake_disc_opinion)
        loss = - torch.mean(logp_disc_opinion)
        return loss


class DiscLoss(torch.nn.Module):
    def forward(self, fake_disc, real_disc):
        logp_real_is_real = torch.nn.LogSigmoid()(real_disc)
        logp_gen_is_fake = torch.nn.LogSigmoid()(-fake_disc)
        loss = -torch.mean(logp_real_is_real + logp_gen_is_fake)
        return loss
