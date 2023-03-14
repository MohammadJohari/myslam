import torch


class EWC(object):
    """
    EWC for continual learning.

    """

    def __init__(self, all_planes, decoders, device):
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes

        self.saved_xy, self.saved_xz, self.saved_yz, self.c_saved_xy, self.c_saved_xz, self.c_saved_yz = [], [], [], [], [], []
        self.imp_xy, self.imp_xz, self.imp_yz, self.c_imp_xy, self.c_imp_xz, self.c_imp_yz = [], [], [], [], [], []

        for i in range(len(planes_xy)):
            self.saved_xy.append(torch.zeros_like(planes_xy[i], device=device))
            self.saved_xz.append(torch.zeros_like(planes_xz[i], device=device))
            self.saved_yz.append(torch.zeros_like(planes_yz[i], device=device))

            self.imp_xy.append(torch.zeros_like(planes_xy[i], device=device))
            self.imp_xz.append(torch.zeros_like(planes_xz[i], device=device))
            self.imp_yz.append(torch.zeros_like(planes_yz[i], device=device))

        for i in range(len(c_planes_xy)):
            self.c_saved_xy.append(torch.zeros_like(c_planes_xy[i], device=device))
            self.c_saved_xz.append(torch.zeros_like(c_planes_xz[i], device=device))
            self.c_saved_yz.append(torch.zeros_like(c_planes_yz[i], device=device))

            self.c_imp_xy.append(torch.zeros_like(c_planes_xy[i], device=device))
            self.c_imp_xz.append(torch.zeros_like(c_planes_xz[i], device=device))
            self.c_imp_yz.append(torch.zeros_like(c_planes_yz[i], device=device))

        self.saved_params = {}
        self.imp_params = {}
        for name, p in decoders.named_parameters():
            self.saved_params[name] = torch.zeros_like(p, device=device)
            self.imp_params[name] = torch.zeros_like(p, device=device)

    def update_params(self, all_planes, decoders):
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes

        for i in range(len(planes_xy)):
            self.saved_xy[i] = planes_xy[i].clone().detach()
            self.saved_xz[i] = planes_xz[i].clone().detach()
            self.saved_yz[i] = planes_yz[i].clone().detach()

        for i in range(len(c_planes_xy)):
            self.c_saved_xy[i] = c_planes_xy[i].clone().detach()
            self.c_saved_xz[i] = c_planes_xz[i].clone().detach()
            self.c_saved_yz[i] = c_planes_yz[i].clone().detach()

        for name, p in decoders.named_parameters():
            self.saved_params[name] = p.clone().detach()

    def update_imps(self, all_planes, decoders, alpha=0.99, weight=1):
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes

        for i in range(len(planes_xy)):
            mask = planes_xy[i].grad != 0
            self.imp_xy[i][mask] = alpha * self.imp_xy[i][mask] + (1 - alpha) * (planes_xy[i].grad[mask] ** 2) * weight
            # breakpoint()

            mask = planes_xz[i].grad != 0
            self.imp_xz[i][mask] = alpha * self.imp_xz[i][mask] + (1 - alpha) * (planes_xz[i].grad[mask] ** 2) * weight

            mask = planes_yz[i].grad != 0
            self.imp_yz[i][mask] = alpha * self.imp_yz[i][mask] + (1 - alpha) * (planes_yz[i].grad[mask] ** 2) * weight

        for i in range(len(c_planes_xy)):
            mask = c_planes_xy[i].grad != 0
            self.c_imp_xy[i][mask] = alpha * self.c_imp_xy[i][mask] + (1 - alpha) * (c_planes_xy[i].grad[mask] ** 2) * weight

            mask = c_planes_xz[i].grad != 0
            self.c_imp_xz[i][mask] = alpha * self.c_imp_xz[i][mask] + (1 - alpha) * (c_planes_xz[i].grad[mask] ** 2) * weight

            mask = c_planes_yz[i].grad != 0
            self.c_imp_yz[i][mask] = alpha * self.c_imp_yz[i][mask] + (1 - alpha) * (c_planes_yz[i].grad[mask] ** 2) * weight

        for name, p in decoders.named_parameters():
            self.imp_params[name] = alpha * self.imp_params[name] + (1 - alpha) * (p.grad ** 2) * weight

    def penalty(self, all_planes, decoders, weight=6e2):
        planes_xy, planes_xz, planes_yz, c_planes_xy, c_planes_xz, c_planes_yz = all_planes

        loss = 0
        # for i in range(len(planes_xy)):
        #     loss = loss + torch.sum(self.imp_xy[i] * torch.square(planes_xy[i] - self.saved_xy[i]))
        #     loss = loss + torch.sum(self.imp_xz[i] * torch.square(planes_xz[i] - self.saved_xz[i]))
        #     loss = loss + torch.sum(self.imp_yz[i] * torch.square(planes_yz[i] - self.saved_yz[i]))

        # for i in range(len(c_planes_xy)):
        #     loss = loss + torch.sum(self.c_imp_xy[i] * torch.square(c_planes_xy[i] - self.c_saved_xy[i]))
        #     loss = loss + torch.sum(self.c_imp_xz[i] * torch.square(c_planes_xz[i] - self.c_saved_xz[i]))
        #     loss = loss + torch.sum(self.c_imp_yz[i] * torch.square(c_planes_yz[i] - self.c_saved_yz[i]))

        for name, p in decoders.named_parameters():
            loss = loss + torch.mean(self.imp_params[name] * torch.square(p - self.saved_params[name]))

        loss = weight * loss

        return loss