import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class airnet_uncertainty_loss(nn.Module):
    def __init__(self):
        super(airnet_uncertainty_loss, self).__init__()
        #self.device = torch.device()

    def forward(self, restored, clean_patch_1, de_id, un):
        pred_device = restored.device
        old_loss_dict = [torch.tensor(0, device=pred_device) for _ in range(5)]
        uncertainty_loss_dict = [torch.tensor(0, device=pred_device) for _ in range(5)]
        uncertainty_bn_dict = [torch.tensor(0, device=pred_device) for _ in range(5)]

        # 初始化每个类别的列表
        category_lists = {
            0: ([], [], []),  # denoise_15
            1: ([], [], []),  # denoise_25
            2: ([], [], []),  # denoise_50
            3: ([], [], []),  # derain
            4: ([], [], [])   # dehaze
        }

        # 将数据分类到相应的类别列表中
        for p, t, y, idx in zip(restored, clean_patch_1, un, de_id):
            category_lists[idx.item()][0].append(t)
            category_lists[idx.item()][1].append(p)
            category_lists[idx.item()][2].append(y)

        total_loss = 0
        category_losses = [0] * 5
        un_hq_list = []
        un_lq_list = []
        num = 0
        eps = 1e-6
        # 计算每个类别的损失
        for idx in range(5):
            lq_list, list_, un_list = category_lists[idx]
            if lq_list:
                
                un_map = torch.mean(torch.stack(un_list), dim=0) + eps
                un_num = torch.mean(un_map)
                s = 1.0 / un_map
                num += 1
                for u, v in zip(lq_list, list_):
                    un_hq_list.append(torch.mul(v, s))
                    un_lq_list.append(torch.mul(u, s))

                old_loss_dict[idx] = F.l1_loss(torch.stack(lq_list), torch.stack(list_), reduction='mean')
                uncertainty_bn_dict[idx] = 2 * torch.log(un_num)
                total_loss += F.l1_loss(torch.stack(un_lq_list), torch.stack(un_hq_list), reduction='mean') + uncertainty_bn_dict[idx]
                category_losses[idx] = F.l1_loss(torch.stack(un_lq_list), torch.stack(un_hq_list), reduction='mean').detach() +uncertainty_bn_dict[idx].detach()
                uncertainty_loss_dict[idx] = F.l1_loss(torch.stack(un_lq_list), torch.stack(un_hq_list), reduction='mean').detach()
        
        #print(total_loss)
        return total_loss/num,category_losses,old_loss_dict,uncertainty_bn_dict,uncertainty_loss_dict
    

@LOSS_REGISTRY.register()
class airnet_uncertainty_loss_5_types(nn.Module):
    def __init__(self):
        super(airnet_uncertainty_loss_5_types, self).__init__()
        #self.device = torch.device()

    def forward(self, restored, clean_patch_1, de_id, un):
        pred_device = restored.device
        old_loss_dict = [torch.tensor(0, device=pred_device) for _ in range(7)]
        uncertainty_loss_dict = [torch.tensor(0, device=pred_device) for _ in range(7)]
        uncertainty_bn_dict = [torch.tensor(0, device=pred_device) for _ in range(7)]

        # 初始化每个类别的列表
        category_lists = {
            0: ([], [], []),  # denoise_15
            1: ([], [], []),  # denoise_25
            2: ([], [], []),  # denoise_50
            3: ([], [], []),  # derain
            4: ([], [], []),  # dehaze
            6: ([], [], []),  # gopro
            7: ([], [], [])   # lol  
        }

        # 将数据分类到相应的类别列表中
        for p, t, y, idx in zip(restored, clean_patch_1, un, de_id):
            category_lists[idx.item()][0].append(t)
            category_lists[idx.item()][1].append(p)
            category_lists[idx.item()][2].append(y)

        total_loss = 0
        category_losses = [0] * 7
        un_hq_list = []
        un_lq_list = []
        num = 0
        eps = 1e-6
        # 计算每个类别的损失
        for idx in range(7):
            if idx <5:
                lq_list, list_, un_list = category_lists[idx]
            else:
                lq_list, list_, un_list = category_lists[idx+1]
            if lq_list:
                
                un_map = torch.mean(torch.stack(un_list), dim=0) + eps
                un_num = torch.mean(un_map)
                s = 1.0 / un_map
                num += 1
                for u, v in zip(lq_list, list_):
                    un_hq_list.append(torch.mul(v, s))
                    un_lq_list.append(torch.mul(u, s))

                old_loss_dict[idx] = F.l1_loss(torch.stack(lq_list), torch.stack(list_), reduction='mean')
                uncertainty_bn_dict[idx] = 2 * torch.log(un_num)
                total_loss += F.l1_loss(torch.stack(un_lq_list), torch.stack(un_hq_list), reduction='mean') + uncertainty_bn_dict[idx]
                category_losses[idx] = F.l1_loss(torch.stack(un_lq_list), torch.stack(un_hq_list), reduction='mean').detach() +uncertainty_bn_dict[idx].detach()
                uncertainty_loss_dict[idx] = F.l1_loss(torch.stack(un_lq_list), torch.stack(un_hq_list), reduction='mean').detach()
        
        #print(total_loss)
        return total_loss/num,category_losses,old_loss_dict,uncertainty_bn_dict,uncertainty_loss_dict