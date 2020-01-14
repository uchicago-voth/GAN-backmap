import torch
import torch.nn as nn


# Gradient penalty loss
def calc_gradient_penalty(netD, aa_real, aa_fake, cg_real, BATCH_SIZE, AA_NUM_ATOMS, CG_NUM_ATOMS, NUM_DIMS, GP_LAMBDA, device):
    alpha = torch.rand(BATCH_SIZE, 1)
    alpha = alpha.expand(BATCH_SIZE, int(aa_real.nelement()/BATCH_SIZE)).contiguous()
    alpha = alpha.view(BATCH_SIZE, AA_NUM_ATOMS, NUM_DIMS)
    alpha = alpha.to(device)


    aa_fake = aa_fake.view(BATCH_SIZE, AA_NUM_ATOMS, NUM_DIMS)

    interpolates = alpha * aa_real.detach() + (1-alpha) * aa_fake.detach()
    interpolates = interpolates.to(device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs = interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]


    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * GP_LAMBDA
    return gradient_penalty


#L1 norm

#Cycle losses
def forward_cycle_loss(norm_loss, G_f, netG, aa, noise, CYCLE_LAMBDA):
    cg = torch.matmul(torch.transpose(aa, 1, 2), G_f)
    cg = torch.transpose(cg, 1, 2).contiguous()
    cycled = netG(noise, cg)
    return norm_loss(cycled, aa) * CYCLE_LAMBDA
                
                
def backward_cycle_loss(norm_loss, G_f, netG, cg, noise, CYCLE_LAMBDA):
    aa = netG(noise, cg)
    cycled = torch.matmul(torch.transpose(aa, 1, 2), G_f)
    cycled = torch.transpose(cycled, 1, 2).contiguous()
    return norm_loss(cycled, cg) * CYCLE_LAMBDA



#Wasserstein loss
def wasserstein_loss(output, label):
    return torch.mean(output * label)
