import torch
import torch.nn as nn


# Gradient penalty loss
def calc_gradient_penalty(netD, aa_real, aa_fake, cg_real, param):
    alpha = torch.rand(param.BATCH_SIZE, 1)
    alpha = alpha.expand(param.BATCH_SIZE, int(aa_real.nelement()/param.BATCH_SIZE)).contiguous()
    alpha = alpha.view(param.BATCH_SIZE, param.AA_NUM_ATOMS, param.NUM_DIMS)
    alpha = alpha.to(param.device)


    aa_fake = aa_fake.view(param.BATCH_SIZE, param.AA_NUM_ATOMS, param.NUM_DIMS)

    interpolates = alpha * aa_real.detach() + (1-alpha) * aa_fake.detach()
    interpolates = interpolates.to(param.device)
    interpolates.requires_grad_(True)

    disc_interpolates = netD(interpolates)

    gradients = torch.autograd.grad(outputs=disc_interpolates, inputs = interpolates,
            grad_outputs=torch.ones(disc_interpolates.size()).to(device),
            create_graph=True, retain_graph=True, only_inputs=True)[0]


    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * param.GP_LAMBDA
    return gradient_penalty


#L1 norm
norm_loss = nn.L1Loss()
#Cycle losses
def forward_cycle_loss(G_f, netG, aa, noise, CYCLE_LAMBDA):
    cg = torch.matmul(torch.transpose(aa, 1, 2), G_f)
    cg = torch.transpose(cg, 1, 2).contiguous()
    cycled = netG(noise, cg)
    return norm_loss(cycled, aa) * CYCLE_LAMBDA
                
                
def backward_cycle_loss(norm_loss, G_f, netG, cg, noise, CYCLE_LAMBDA):
    aa = netG(noise, cg)
    cycled = torch.matmul(torch.transpose(aa, 1, 2), G_f)
    cycled = torch.transpose(cycled, 1, 2).contiguous()
    return norm_loss(cycled, cg) * CYCLE_LAMBDA


def general_cycle_loss(net1, net2, data, noise, CYCLE_LAMBDA):
    half_cycled = net1(noise, data) 
    cycled = net2(noise, half_cycled)
    return norm_loss(cycled, aa) * CYCLE_LAMBDA

#Wasserstein loss
def wasserstein_loss(output, label):
    return torch.mean(output * label)
