import torch
from torch import nn as nn
from torch.nn import functional as F

from basicsr.utils.registry import LOSS_REGISTRY
from .loss_util import weighted_loss

_reduction_modes = ['none', 'mean', 'sum']


@LOSS_REGISTRY.register()
class l1_loss_type(nn.Module):

    def __init__(self):
        super(l1_loss_type, self).__init__()

    def forward(self,  pred, target,types):
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list=[]

        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list=[]

        for p,t,img_type in zip(pred,target,types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)

        blur_loss=0
        if blur_lq_list != []:
            blur_loss=F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')

        noise_loss = 0
        if noise_lq_list != []:
            noise_loss = F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')

        jpeg_loss = 0
        if jpeg_lq_list != []:
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')

        rain_loss = 0
        if rain_lq_list != []:
            rain_loss = F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')

        haze_loss = 0
        if haze_lq_list != []:
            haze_loss = F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')

        dark_loss = 0
        if dark_lq_list != []:
            dark_loss = F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')
        
        sr_loss = 0
        if sr_lq_list != []:
            sr_loss = F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')

        return [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss]

@LOSS_REGISTRY.register()
class l1_loss_type_DWA(nn.Module):
    #@profile
    
    def __init__(self):
        super(l1_loss_type_DWA, self).__init__()
        self.device = torch.device('cuda')
        
        #用于记录上一iter的loss值，还没有引入，先全赋值为1测试
        # self.loss_blur_last=torch.Tensor([1])[0].to(self.device)
        # self.loss_noise_last=torch.Tensor([1])[0].to(self.device)
        # self.loss_jpeg_last=torch.Tensor([1])[0].to(self.device)
        # self.loss_rain_last=torch.Tensor([1])[0].to(self.device)
        # self.loss_haze_last=torch.Tensor([1])[0].to(self.device)
        # self.loss_dark_last=torch.Tensor([1])[0].to(self.device)
        # self.loss_sr_last=torch.Tensor([1])[0].to(self.device)
    #@profile
    def forward(self, pred, target, types ,loss_list_last):# 还需要额外引入一个list，传入上1 iter的loss数据
        dwa_loss = 0
        #loss_dict = [0,0,0,0,0,0,0]#[torch.tensor([0], device=pred.device) for i in range(7)] #记录7个任务的loss
        loss_grad_dict = [torch.tensor(-10000, device=pred.device) for i in range(7)]#[-1e4,-1e4,-1e4,-1e4,-1e4,-1e4,-1e4]#[torch.tensor([1], device=pred.device) for i in range(7)]  #记录当前iter loss与上iter loss的比值
        #loss_dict = [torch.tensor([0], device=pred.device) for i in range(7)]
        #loss_dict = torch.zeros(1,7).to(self.device)
        #loss_dict.requires_grad_(True)
        weight_list_dict = [torch.tensor([0], device=pred.device) for i in range(7)]#[0,0,0,0,0,0,0]
        
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list = []
        T = 10
        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list = []

        for p, t, img_type in zip(pred, target, types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)

        
        blur_loss = 0
        if blur_lq_list != []:
            blur_loss = F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')
            #loss_dict[0,0] = blur_loss
            loss_grad_dict[0]=(blur_loss/loss_list_last[0])/T
            #loss_grad_dict[0]=blur_loss
        noise_loss = 0
        if noise_lq_list != []:
            noise_loss = F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')
            #loss_dict[0,1] = noise_loss
            #loss_grad_dict[1] = noise_loss/self.loss_noise_last
            loss_grad_dict[1] = (noise_loss/loss_list_last[1])/T
        jpeg_loss = 0
        if jpeg_lq_list != []:
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')
            #loss_dict[0,2] = jpeg_loss
            #loss_grad_dict[2] = jpeg_loss/self.loss_jpeg_last
            loss_grad_dict[2] = (jpeg_loss/loss_list_last[2])/T
        rain_loss = 0
        if rain_lq_list != []:
            rain_loss = F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')
            #loss_dict[0,3] = rain_loss
            #loss_grad_dict[3] = rain_loss/self.loss_rain_last
            loss_grad_dict[3] = (rain_loss/loss_list_last[3])/T
        haze_loss = 0
        if haze_lq_list != []:
            haze_loss = F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')
            #loss_dict[0,4] = haze_loss
            #loss_grad_dict[4] = haze_loss/self.loss_haze_last
            loss_grad_dict[4] = (haze_loss/loss_list_last[4])/T
        dark_loss = 0
        if dark_lq_list != []:
            dark_loss = F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')
            #loss_dict[0,5] = dark_loss
            #loss_grad_dict[5] = dark_loss/self.loss_dark_last 
            loss_grad_dict[5] = (dark_loss/loss_list_last[5])/T
        sr_loss = 0
        if sr_lq_list != []:
            sr_loss = F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')
            #loss_dict[0,6] = sr_loss
            #loss_grad_dict[6] = sr_loss/self.loss_sr_last
            loss_grad_dict[6] = (sr_loss/loss_list_last[6])/T
        
        weight_list = torch.softmax(torch.tensor(loss_grad_dict),dim = 0).to(self.device)
        
        weight_list_dict[0] = weight_list[0].detach()
        weight_list_dict[1] = weight_list[1].detach()
        weight_list_dict[2] = weight_list[2].detach()
        weight_list_dict[3] = weight_list[3].detach()
        weight_list_dict[4] = weight_list[4].detach()
        weight_list_dict[5] = weight_list[5].detach()
        weight_list_dict[6] = weight_list[6].detach()
        
        #print(loss_grad_dict)
        
        dwa_loss += blur_loss*weight_list[0] + noise_loss*weight_list[1] + jpeg_loss*weight_list[2] + rain_loss*weight_list[3] + haze_loss*weight_list[4] + dark_loss*weight_list[5] + sr_loss*weight_list[6]
        """
        tensor(0.0272, device='cuda:3', grad_fn=<AddBackward0>)
        """
        my_zero = torch.tensor(0, device=pred.device)
        if blur_lq_list ==[]:
            loss_grad_dict[0] = my_zero
        if noise_lq_list ==[]:
            loss_grad_dict[1] = my_zero
        if jpeg_lq_list ==[]:
            loss_grad_dict[2] = my_zero
        if rain_lq_list ==[]:
            loss_grad_dict[3] = my_zero    
        if haze_lq_list ==[]:
            loss_grad_dict[4] = my_zero
        if dark_lq_list ==[]:
            loss_grad_dict[5] = my_zero
        if sr_lq_list ==[]:
            loss_grad_dict[6] = my_zero
       
        
        """
        [tensor(0.1620, device='cuda:1', grad_fn=<L1LossBackward0>), tensor(0.1820, device='cuda:1', grad_fn=<L1LossBackward0>), 0, tensor(0.2076, device='cuda:1', grad_fn=<L1LossBackward0>), tensor(0.1742, device='cuda:1', grad_fn=<L1LossBackward0>), 0, 0]
        """
        #del weight_list,loss_grad_dict
        #新增日志列表，观察情况变化
        return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss],loss_grad_dict,weight_list_dict
        
        # return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss]
    

           
       
@LOSS_REGISTRY.register()
class l1_loss_type_DWA_T_10(nn.Module):
    
    def __init__(self):
        super(l1_loss_type_DWA_T_10, self).__init__()
        self.device = torch.device('cuda')

    def forward(self, pred, target, types ,loss_list_last):# 还需要额外引入一个list，传入上1 iter的loss数据
        dwa_loss = 0
        loss_grad_dict = [torch.tensor(-10000, device=pred.device) for i in range(7)]#[-1e4,-1e4,-1e4,-1e4,-1e4,-1e4,-1e4]#[torch.tensor([1], device=pred.device) for i in range(7)]  #记录当前iter loss与上iter loss的比值
        weight_list_dict = [torch.tensor([0], device=pred.device) for i in range(7)]#[0,0,0,0,0,0,0]
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list = []
        T = 10
        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list = []

        for p, t, img_type in zip(pred, target, types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)

        
        blur_loss = 0
        if blur_lq_list != []:
            blur_loss = F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')
            loss_grad_dict[0]=(blur_loss/loss_list_last[0])/T
            
        noise_loss = 0
        if noise_lq_list != []:
            noise_loss = F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')
            loss_grad_dict[1] = (noise_loss/loss_list_last[1])/T
        jpeg_loss = 0
        if jpeg_lq_list != []:
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')
            loss_grad_dict[2] = (jpeg_loss/loss_list_last[2])/T
            
        rain_loss = 0
        if rain_lq_list != []:
            rain_loss = F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')
            loss_grad_dict[3] = (rain_loss/loss_list_last[3])/T
            
        haze_loss = 0
        if haze_lq_list != []:
            haze_loss = F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')
            loss_grad_dict[4] = (haze_loss/loss_list_last[4])/T
            
        dark_loss = 0
        if dark_lq_list != []:
            dark_loss = F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')
            loss_grad_dict[5] = (dark_loss/loss_list_last[5])/T
            
        sr_loss = 0
        if sr_lq_list != []:
            sr_loss = F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')
            loss_grad_dict[6] = (sr_loss/loss_list_last[6])/T
        
        weight_list = torch.softmax(torch.tensor(loss_grad_dict),dim = 0).to(self.device)
        
        weight_list_dict[0] = weight_list[0].detach()
        weight_list_dict[1] = weight_list[1].detach()
        weight_list_dict[2] = weight_list[2].detach()
        weight_list_dict[3] = weight_list[3].detach()
        weight_list_dict[4] = weight_list[4].detach()
        weight_list_dict[5] = weight_list[5].detach()
        weight_list_dict[6] = weight_list[6].detach()
        
        
        dwa_loss += blur_loss*weight_list[0] + noise_loss*weight_list[1] + jpeg_loss*weight_list[2] + rain_loss*weight_list[3] + haze_loss*weight_list[4] + dark_loss*weight_list[5] + sr_loss*weight_list[6]
        
        my_zero = torch.tensor(0, device=pred.device)
        if blur_lq_list ==[]:
            loss_grad_dict[0] = my_zero
        if noise_lq_list ==[]:
            loss_grad_dict[1] = my_zero
        if jpeg_lq_list ==[]:
            loss_grad_dict[2] = my_zero
        if rain_lq_list ==[]:
            loss_grad_dict[3] = my_zero    
        if haze_lq_list ==[]:
            loss_grad_dict[4] = my_zero
        if dark_lq_list ==[]:
            loss_grad_dict[5] = my_zero
        if sr_lq_list ==[]:
            loss_grad_dict[6] = my_zero
       
        return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss],loss_grad_dict,weight_list_dict
        
        # return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss]        
      
@LOSS_REGISTRY.register()
class l1_loss_type_DWA_T_0_1(nn.Module):
    
    def __init__(self):
        super(l1_loss_type_DWA_T_0_1, self).__init__()
        self.device = torch.device('cuda')

    def forward(self, pred, target, types ,loss_list_last):# 还需要额外引入一个list，传入上1 iter的loss数据
        dwa_loss = 0
        loss_grad_dict = [torch.tensor(-10000, device=pred.device) for i in range(7)]#[-1e4,-1e4,-1e4,-1e4,-1e4,-1e4,-1e4]#[torch.tensor([1], device=pred.device) for i in range(7)]  #记录当前iter loss与上iter loss的比值
        weight_list_dict = [torch.tensor([0], device=pred.device) for i in range(7)]#[0,0,0,0,0,0,0]
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list = []
        T = 0.1
        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list = []

        for p, t, img_type in zip(pred, target, types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)

        
        blur_loss = 0
        if blur_lq_list != []:
            blur_loss = F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')
            loss_grad_dict[0]=(blur_loss/loss_list_last[0])/T
            
        noise_loss = 0
        if noise_lq_list != []:
            noise_loss = F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')
            loss_grad_dict[1] = (noise_loss/loss_list_last[1])/T
        jpeg_loss = 0
        if jpeg_lq_list != []:
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')
            loss_grad_dict[2] = (jpeg_loss/loss_list_last[2])/T
            
        rain_loss = 0
        if rain_lq_list != []:
            rain_loss = F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')
            loss_grad_dict[3] = (rain_loss/loss_list_last[3])/T
            
        haze_loss = 0
        if haze_lq_list != []:
            haze_loss = F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')
            loss_grad_dict[4] = (haze_loss/loss_list_last[4])/T
            
        dark_loss = 0
        if dark_lq_list != []:
            dark_loss = F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')
            loss_grad_dict[5] = (dark_loss/loss_list_last[5])/T
            
        sr_loss = 0
        if sr_lq_list != []:
            sr_loss = F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')
            loss_grad_dict[6] = (sr_loss/loss_list_last[6])/T
        
        weight_list = torch.softmax(torch.tensor(loss_grad_dict),dim = 0).to(self.device)
        
        weight_list_dict[0] = weight_list[0].detach()
        weight_list_dict[1] = weight_list[1].detach()
        weight_list_dict[2] = weight_list[2].detach()
        weight_list_dict[3] = weight_list[3].detach()
        weight_list_dict[4] = weight_list[4].detach()
        weight_list_dict[5] = weight_list[5].detach()
        weight_list_dict[6] = weight_list[6].detach()
        
        
        dwa_loss += blur_loss*weight_list[0] + noise_loss*weight_list[1] + jpeg_loss*weight_list[2] + rain_loss*weight_list[3] + haze_loss*weight_list[4] + dark_loss*weight_list[5] + sr_loss*weight_list[6]
        
        my_zero = torch.tensor(0, device=pred.device)
        if blur_lq_list ==[]:
            loss_grad_dict[0] = my_zero
        if noise_lq_list ==[]:
            loss_grad_dict[1] = my_zero
        if jpeg_lq_list ==[]:
            loss_grad_dict[2] = my_zero
        if rain_lq_list ==[]:
            loss_grad_dict[3] = my_zero    
        if haze_lq_list ==[]:
            loss_grad_dict[4] = my_zero
        if dark_lq_list ==[]:
            loss_grad_dict[5] = my_zero
        if sr_lq_list ==[]:
            loss_grad_dict[6] = my_zero
       
        return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss],loss_grad_dict,weight_list_dict
        
        # return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss]          
     
@LOSS_REGISTRY.register()
class l1_loss_type_DWA_T_1(nn.Module):
    
    def __init__(self):
        super(l1_loss_type_DWA_T_1, self).__init__()
        self.device = torch.device('cuda')

    def forward(self, pred, target, types ,loss_list_last):# 还需要额外引入一个list，传入上1 iter的loss数据
        dwa_loss = 0
        loss_grad_dict = [torch.tensor(-10000, device=pred.device) for i in range(7)]#[-1e4,-1e4,-1e4,-1e4,-1e4,-1e4,-1e4]#[torch.tensor([1], device=pred.device) for i in range(7)]  #记录当前iter loss与上iter loss的比值
        weight_list_dict = [torch.tensor([0], device=pred.device) for i in range(7)]#[0,0,0,0,0,0,0]
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list = []
        T = 1
        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list = []

        for p, t, img_type in zip(pred, target, types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)

        
        blur_loss = 0
        if blur_lq_list != []:
            blur_loss = F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')
            loss_grad_dict[0]=(blur_loss/loss_list_last[0])/T
            
        noise_loss = 0
        if noise_lq_list != []:
            noise_loss = F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')
            loss_grad_dict[1] = (noise_loss/loss_list_last[1])/T
        jpeg_loss = 0
        if jpeg_lq_list != []:
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')
            loss_grad_dict[2] = (jpeg_loss/loss_list_last[2])/T
            
        rain_loss = 0
        if rain_lq_list != []:
            rain_loss = F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')
            loss_grad_dict[3] = (rain_loss/loss_list_last[3])/T
            
        haze_loss = 0
        if haze_lq_list != []:
            haze_loss = F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')
            loss_grad_dict[4] = (haze_loss/loss_list_last[4])/T
            
        dark_loss = 0
        if dark_lq_list != []:
            dark_loss = F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')
            loss_grad_dict[5] = (dark_loss/loss_list_last[5])/T
            
        sr_loss = 0
        if sr_lq_list != []:
            sr_loss = F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')
            loss_grad_dict[6] = (sr_loss/loss_list_last[6])/T
        
        weight_list = torch.softmax(torch.tensor(loss_grad_dict),dim = 0).to(self.device)
        
        weight_list_dict[0] = weight_list[0].detach()
        weight_list_dict[1] = weight_list[1].detach()
        weight_list_dict[2] = weight_list[2].detach()
        weight_list_dict[3] = weight_list[3].detach()
        weight_list_dict[4] = weight_list[4].detach()
        weight_list_dict[5] = weight_list[5].detach()
        weight_list_dict[6] = weight_list[6].detach()
        
        
        dwa_loss += blur_loss*weight_list[0] + noise_loss*weight_list[1] + jpeg_loss*weight_list[2] + rain_loss*weight_list[3] + haze_loss*weight_list[4] + dark_loss*weight_list[5] + sr_loss*weight_list[6]
        
        my_zero = torch.tensor(0, device=pred.device)
        if blur_lq_list ==[]:
            loss_grad_dict[0] = my_zero
        if noise_lq_list ==[]:
            loss_grad_dict[1] = my_zero
        if jpeg_lq_list ==[]:
            loss_grad_dict[2] = my_zero
        if rain_lq_list ==[]:
            loss_grad_dict[3] = my_zero    
        if haze_lq_list ==[]:
            loss_grad_dict[4] = my_zero
        if dark_lq_list ==[]:
            loss_grad_dict[5] = my_zero
        if sr_lq_list ==[]:
            loss_grad_dict[6] = my_zero
       
        return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss],loss_grad_dict,weight_list_dict
        
        # return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss]   
        
        
@LOSS_REGISTRY.register()
class l1_loss_type_LossWA_T_10(nn.Module):
    
    def __init__(self):
        super(l1_loss_type_LossWA_T_10, self).__init__()
        self.device = torch.device('cuda')

    def forward(self, pred, target, types ,loss_list_last):# 还需要额外引入一个list，传入上1 iter的loss数据
        dwa_loss = 0
        loss_grad_dict = [torch.tensor(-10000, device=pred.device) for i in range(7)]#[-1e4,-1e4,-1e4,-1e4,-1e4,-1e4,-1e4]#[torch.tensor([1], device=pred.device) for i in range(7)]  #记录当前iter loss与上iter loss的比值
        weight_list_dict = [torch.tensor([0], device=pred.device) for i in range(7)]#[0,0,0,0,0,0,0]
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list = []
        T = 10
        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list = []

        for p, t, img_type in zip(pred, target, types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)

        
        blur_loss = 0
        if blur_lq_list != []:
            blur_loss = F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')
            loss_grad_dict[0]= loss_list_last[0]/T
            
        noise_loss = 0
        if noise_lq_list != []:
            noise_loss = F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')
            loss_grad_dict[1] = loss_list_last[1]/T
        jpeg_loss = 0
        if jpeg_lq_list != []:
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')
            loss_grad_dict[2] = loss_list_last[2]/T
            
        rain_loss = 0
        if rain_lq_list != []:
            rain_loss = F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')
            loss_grad_dict[3] = loss_list_last[3]/T
            
        haze_loss = 0
        if haze_lq_list != []:
            haze_loss = F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')
            loss_grad_dict[4] = loss_list_last[4]/T
            
        dark_loss = 0
        if dark_lq_list != []:
            dark_loss = F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')
            loss_grad_dict[5] = loss_list_last[5]/T
            
        sr_loss = 0
        if sr_lq_list != []:
            sr_loss = F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')
            loss_grad_dict[6] = loss_list_last[6]/T
        
        weight_list = torch.softmax(torch.tensor(loss_grad_dict),dim = 0).to(self.device)
        
        weight_list_dict[0] = weight_list[0].detach()
        weight_list_dict[1] = weight_list[1].detach()
        weight_list_dict[2] = weight_list[2].detach()
        weight_list_dict[3] = weight_list[3].detach()
        weight_list_dict[4] = weight_list[4].detach()
        weight_list_dict[5] = weight_list[5].detach()
        weight_list_dict[6] = weight_list[6].detach()
        
        
        dwa_loss += blur_loss*weight_list[0] + noise_loss*weight_list[1] + jpeg_loss*weight_list[2] + rain_loss*weight_list[3] + haze_loss*weight_list[4] + dark_loss*weight_list[5] + sr_loss*weight_list[6]
        
        my_zero = torch.tensor(0, device=pred.device)
        if blur_lq_list ==[]:
            loss_grad_dict[0] = my_zero
        if noise_lq_list ==[]:
            loss_grad_dict[1] = my_zero
        if jpeg_lq_list ==[]:
            loss_grad_dict[2] = my_zero
        if rain_lq_list ==[]:
            loss_grad_dict[3] = my_zero    
        if haze_lq_list ==[]:
            loss_grad_dict[4] = my_zero
        if dark_lq_list ==[]:
            loss_grad_dict[5] = my_zero
        if sr_lq_list ==[]:
            loss_grad_dict[6] = my_zero
       
        return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss],loss_grad_dict,weight_list_dict
        
        # return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss]     

@LOSS_REGISTRY.register()
class l1_loss_type_LossWA_T_10_with_ema(nn.Module):
    
    def __init__(self):
        super(l1_loss_type_LossWA_T_10_with_ema, self).__init__()
        self.device = torch.device('cuda')

    def forward(self, pred, target, types ,loss_list_last):# 还需要额外引入一个list，传入上1 iter的loss数据
        dwa_loss = 0
        loss_grad_dict = [torch.tensor(-10000, device=pred.device) for i in range(7)]#[-1e4,-1e4,-1e4,-1e4,-1e4,-1e4,-1e4]#[torch.tensor([1], device=pred.device) for i in range(7)]  #记录当前iter loss与上iter loss的比值
        weight_list_dict = [torch.tensor([0], device=pred.device) for i in range(7)]#[0,0,0,0,0,0,0]
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list = []
        T = 10
        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list = []

        for p, t, img_type in zip(pred, target, types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)

        
        blur_loss = 0
        if blur_lq_list != []:
            blur_loss = F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')
            loss_grad_dict[0]= loss_list_last[0]/T
            
        noise_loss = 0
        if noise_lq_list != []:
            noise_loss = F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')
            loss_grad_dict[1] = loss_list_last[1]/T
        jpeg_loss = 0
        if jpeg_lq_list != []:
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')
            loss_grad_dict[2] = loss_list_last[2]/T
            
        rain_loss = 0
        if rain_lq_list != []:
            rain_loss = F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')
            loss_grad_dict[3] = loss_list_last[3]/T
            
        haze_loss = 0
        if haze_lq_list != []:
            haze_loss = F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')
            loss_grad_dict[4] = loss_list_last[4]/T
            
        dark_loss = 0
        if dark_lq_list != []:
            dark_loss = F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')
            loss_grad_dict[5] = loss_list_last[5]/T
            
        sr_loss = 0
        if sr_lq_list != []:
            sr_loss = F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')
            loss_grad_dict[6] = loss_list_last[6]/T
        
        weight_list = torch.softmax(torch.tensor(loss_grad_dict),dim = 0).to(self.device)
        
        weight_list_dict[0] = weight_list[0].detach()
        weight_list_dict[1] = weight_list[1].detach()
        weight_list_dict[2] = weight_list[2].detach()
        weight_list_dict[3] = weight_list[3].detach()
        weight_list_dict[4] = weight_list[4].detach()
        weight_list_dict[5] = weight_list[5].detach()
        weight_list_dict[6] = weight_list[6].detach()
        
        
        dwa_loss += blur_loss*weight_list[0] + noise_loss*weight_list[1] + jpeg_loss*weight_list[2] + rain_loss*weight_list[3] + haze_loss*weight_list[4] + dark_loss*weight_list[5] + sr_loss*weight_list[6]
        
        my_zero = torch.tensor(0, device=pred.device)
        if blur_lq_list ==[]:
            loss_grad_dict[0] = my_zero
        if noise_lq_list ==[]:
            loss_grad_dict[1] = my_zero
        if jpeg_lq_list ==[]:
            loss_grad_dict[2] = my_zero
        if rain_lq_list ==[]:
            loss_grad_dict[3] = my_zero    
        if haze_lq_list ==[]:
            loss_grad_dict[4] = my_zero
        if dark_lq_list ==[]:
            loss_grad_dict[5] = my_zero
        if sr_lq_list ==[]:
            loss_grad_dict[6] = my_zero
       
        return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss],loss_grad_dict,weight_list_dict
        
        # return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss]     

@LOSS_REGISTRY.register()
class l1_loss_type_LossWA_T_0_1_with_ema(nn.Module):
    
    def __init__(self):
        super(l1_loss_type_LossWA_T_0_1_with_ema, self).__init__()
        self.device = torch.device('cuda')

    def forward(self, pred, target, types ,loss_list_last):# 还需要额外引入一个list，传入上1 iter的loss数据
        dwa_loss = 0
        loss_grad_dict = [torch.tensor(-10000, device=pred.device) for i in range(7)]#[-1e4,-1e4,-1e4,-1e4,-1e4,-1e4,-1e4]#[torch.tensor([1], device=pred.device) for i in range(7)]  #记录当前iter loss与上iter loss的比值
        weight_list_dict = [torch.tensor([0], device=pred.device) for i in range(7)]#[0,0,0,0,0,0,0]
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list = []
        T = 0.1
        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list = []

        for p, t, img_type in zip(pred, target, types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)

        
        blur_loss = 0
        if blur_lq_list != []:
            blur_loss = F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')
            loss_grad_dict[0]= loss_list_last[0]/T
            
        noise_loss = 0
        if noise_lq_list != []:
            noise_loss = F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')
            loss_grad_dict[1] = loss_list_last[1]/T
        jpeg_loss = 0
        if jpeg_lq_list != []:
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')
            loss_grad_dict[2] = loss_list_last[2]/T
            
        rain_loss = 0
        if rain_lq_list != []:
            rain_loss = F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')
            loss_grad_dict[3] = loss_list_last[3]/T
            
        haze_loss = 0
        if haze_lq_list != []:
            haze_loss = F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')
            loss_grad_dict[4] = loss_list_last[4]/T
            
        dark_loss = 0
        if dark_lq_list != []:
            dark_loss = F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')
            loss_grad_dict[5] = loss_list_last[5]/T
            
        sr_loss = 0
        if sr_lq_list != []:
            sr_loss = F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')
            loss_grad_dict[6] = loss_list_last[6]/T
        
        weight_list = torch.softmax(torch.tensor(loss_grad_dict),dim = 0).to(self.device)
        
        weight_list_dict[0] = weight_list[0].detach()
        weight_list_dict[1] = weight_list[1].detach()
        weight_list_dict[2] = weight_list[2].detach()
        weight_list_dict[3] = weight_list[3].detach()
        weight_list_dict[4] = weight_list[4].detach()
        weight_list_dict[5] = weight_list[5].detach()
        weight_list_dict[6] = weight_list[6].detach()
        
        
        dwa_loss += blur_loss*weight_list[0] + noise_loss*weight_list[1] + jpeg_loss*weight_list[2] + rain_loss*weight_list[3] + haze_loss*weight_list[4] + dark_loss*weight_list[5] + sr_loss*weight_list[6]
        
        my_zero = torch.tensor(0, device=pred.device)
        if blur_lq_list ==[]:
            loss_grad_dict[0] = my_zero
        if noise_lq_list ==[]:
            loss_grad_dict[1] = my_zero
        if jpeg_lq_list ==[]:
            loss_grad_dict[2] = my_zero
        if rain_lq_list ==[]:
            loss_grad_dict[3] = my_zero    
        if haze_lq_list ==[]:
            loss_grad_dict[4] = my_zero
        if dark_lq_list ==[]:
            loss_grad_dict[5] = my_zero
        if sr_lq_list ==[]:
            loss_grad_dict[6] = my_zero
       
        return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss],loss_grad_dict,weight_list_dict
        
        # return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss]     

@LOSS_REGISTRY.register()
class l1_loss_type_LossWA_T_1(nn.Module):
    
    def __init__(self):
        super(l1_loss_type_LossWA_T_1, self).__init__()
        self.device = torch.device('cuda')

    def forward(self, pred, target, types ,loss_list_last):# 还需要额外引入一个list，传入上1 iter的loss数据
        dwa_loss = 0
        loss_grad_dict = [torch.tensor(-10000, device=pred.device) for i in range(7)]#[-1e4,-1e4,-1e4,-1e4,-1e4,-1e4,-1e4]#[torch.tensor([1], device=pred.device) for i in range(7)]  #记录当前iter loss与上iter loss的比值
        weight_list_dict = [torch.tensor([0], device=pred.device) for i in range(7)]#[0,0,0,0,0,0,0]
        blur_list = []
        noise_list = []
        jpeg_list = []
        rain_list = []
        haze_list = []
        dark_list = []
        sr_list = []
        T = 1
        blur_lq_list = []
        noise_lq_list = []
        jpeg_lq_list = []
        rain_lq_list = []
        haze_lq_list = []
        dark_lq_list = []
        sr_lq_list = []

        for p, t, img_type in zip(pred, target, types):
            if 'snow' in img_type:
                print(ddd)
            elif 'rain' in img_type:
                rain_list.append(t)
                rain_lq_list.append(p)
            elif 'haze' in img_type:
                haze_list.append(t)
                haze_lq_list.append(p)
            elif 'blur' in img_type:
                blur_list.append(t)
                blur_lq_list.append(p)
            elif 'noise' in img_type:
                noise_list.append(t)
                noise_lq_list.append(p)
            elif 'jpeg' in img_type:
                jpeg_list.append(t)
                jpeg_lq_list.append(p)
            elif 'dark' in img_type:
                dark_list.append(t)
                dark_lq_list.append(p)
            elif 'sr' in img_type:
                sr_list.append(t)
                sr_lq_list.append(p)

        
        blur_loss = 0
        if blur_lq_list != []:
            blur_loss = F.l1_loss(torch.stack(blur_lq_list), torch.stack(blur_list), reduction='mean')
            loss_grad_dict[0]= loss_list_last[0]/T
            
        noise_loss = 0
        if noise_lq_list != []:
            noise_loss = F.l1_loss(torch.stack(noise_lq_list), torch.stack(noise_list), reduction='mean')
            loss_grad_dict[1] = loss_list_last[1]/T
        jpeg_loss = 0
        if jpeg_lq_list != []:
            jpeg_loss = F.l1_loss(torch.stack(jpeg_lq_list), torch.stack(jpeg_list), reduction='mean')
            loss_grad_dict[2] = loss_list_last[2]/T
            
        rain_loss = 0
        if rain_lq_list != []:
            rain_loss = F.l1_loss(torch.stack(rain_lq_list), torch.stack(rain_list), reduction='mean')
            loss_grad_dict[3] = loss_list_last[3]/T
            
        haze_loss = 0
        if haze_lq_list != []:
            haze_loss = F.l1_loss(torch.stack(haze_lq_list), torch.stack(haze_list), reduction='mean')
            loss_grad_dict[4] = loss_list_last[4]/T
            
        dark_loss = 0
        if dark_lq_list != []:
            dark_loss = F.l1_loss(torch.stack(dark_lq_list), torch.stack(dark_list), reduction='mean')
            loss_grad_dict[5] = loss_list_last[5]/T
            
        sr_loss = 0
        if sr_lq_list != []:
            sr_loss = F.l1_loss(torch.stack(sr_lq_list), torch.stack(sr_list), reduction='mean')
            loss_grad_dict[6] = loss_list_last[6]/T
        
        weight_list = torch.softmax(torch.tensor(loss_grad_dict),dim = 0).to(self.device)
        
        weight_list_dict[0] = weight_list[0].detach()
        weight_list_dict[1] = weight_list[1].detach()
        weight_list_dict[2] = weight_list[2].detach()
        weight_list_dict[3] = weight_list[3].detach()
        weight_list_dict[4] = weight_list[4].detach()
        weight_list_dict[5] = weight_list[5].detach()
        weight_list_dict[6] = weight_list[6].detach()
        
        
        dwa_loss += blur_loss*weight_list[0] + noise_loss*weight_list[1] + jpeg_loss*weight_list[2] + rain_loss*weight_list[3] + haze_loss*weight_list[4] + dark_loss*weight_list[5] + sr_loss*weight_list[6]
        
        my_zero = torch.tensor(0, device=pred.device)
        if blur_lq_list ==[]:
            loss_grad_dict[0] = my_zero
        if noise_lq_list ==[]:
            loss_grad_dict[1] = my_zero
        if jpeg_lq_list ==[]:
            loss_grad_dict[2] = my_zero
        if rain_lq_list ==[]:
            loss_grad_dict[3] = my_zero    
        if haze_lq_list ==[]:
            loss_grad_dict[4] = my_zero
        if dark_lq_list ==[]:
            loss_grad_dict[5] = my_zero
        if sr_lq_list ==[]:
            loss_grad_dict[6] = my_zero
       
        return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss],loss_grad_dict,weight_list_dict
        
        # return dwa_loss , [blur_loss,noise_loss,jpeg_loss,rain_loss,haze_loss,dark_loss,sr_loss]     
@weighted_loss
def l1_loss(pred, target):
    return F.l1_loss(pred, target, reduction='none')


@weighted_loss
def mse_loss(pred, target):
    return F.mse_loss(pred, target, reduction='none')


@weighted_loss
def charbonnier_loss(pred, target, eps=1e-12):
    return torch.sqrt((pred - target)**2 + eps)


@LOSS_REGISTRY.register()
class L1Loss(nn.Module):
    """L1 (mean absolute error, MAE) loss.

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(L1Loss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * l1_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class MSELoss(nn.Module):
    """MSE (L2) loss.

    Args:
        loss_weight (float): Loss weight for MSE loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        super(MSELoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * mse_loss(pred, target, weight, reduction=self.reduction)


@LOSS_REGISTRY.register()
class CharbonnierLoss(nn.Module):
    """Charbonnier loss (one variant of Robust L1Loss, a differentiable
    variant of L1Loss).

    Described in "Deep Laplacian Pyramid Networks for Fast and Accurate
        Super-Resolution".

    Args:
        loss_weight (float): Loss weight for L1 loss. Default: 1.0.
        reduction (str): Specifies the reduction to apply to the output.
            Supported choices are 'none' | 'mean' | 'sum'. Default: 'mean'.
        eps (float): A value used to control the curvature near zero. Default: 1e-12.
    """

    def __init__(self, loss_weight=1.0, reduction='mean', eps=1e-12):
        super(CharbonnierLoss, self).__init__()
        if reduction not in ['none', 'mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: {_reduction_modes}')

        self.loss_weight = loss_weight
        self.reduction = reduction
        self.eps = eps

    def forward(self, pred, target, weight=None, **kwargs):
        """
        Args:
            pred (Tensor): of shape (N, C, H, W). Predicted tensor.
            target (Tensor): of shape (N, C, H, W). Ground truth tensor.
            weight (Tensor, optional): of shape (N, C, H, W). Element-wise weights. Default: None.
        """
        return self.loss_weight * charbonnier_loss(pred, target, weight, eps=self.eps, reduction=self.reduction)


@LOSS_REGISTRY.register()
class WeightedTVLoss(L1Loss):
    """Weighted TV loss.

    Args:
        loss_weight (float): Loss weight. Default: 1.0.
    """

    def __init__(self, loss_weight=1.0, reduction='mean'):
        if reduction not in ['mean', 'sum']:
            raise ValueError(f'Unsupported reduction mode: {reduction}. Supported ones are: mean | sum')
        super(WeightedTVLoss, self).__init__(loss_weight=loss_weight, reduction=reduction)

    def forward(self, pred, weight=None):
        if weight is None:
            y_weight = None
            x_weight = None
        else:
            y_weight = weight[:, :, :-1, :]
            x_weight = weight[:, :, :, :-1]

        y_diff = super().forward(pred[:, :, :-1, :], pred[:, :, 1:, :], weight=y_weight)
        x_diff = super().forward(pred[:, :, :, :-1], pred[:, :, :, 1:], weight=x_weight)

        loss = x_diff + y_diff

        return loss


@LOSS_REGISTRY.register()
class PerceptualLoss(nn.Module):
    """Perceptual loss with commonly used style loss.

    Args:
        layer_weights (dict): The weight for each layer of vgg feature.
            Here is an example: {'conv5_4': 1.}, which means the conv5_4
            feature layer (before relu5_4) will be extracted with weight
            1.0 in calculating losses.
        vgg_type (str): The type of vgg network used as feature extractor.
            Default: 'vgg19'.
        use_input_norm (bool):  If True, normalize the input image in vgg.
            Default: True.
        range_norm (bool): If True, norm images with range [-1, 1] to [0, 1].
            Default: False.
        perceptual_weight (float): If `perceptual_weight > 0`, the perceptual
            loss will be calculated and the loss will multiplied by the
            weight. Default: 1.0.
        style_weight (float): If `style_weight > 0`, the style loss will be
            calculated and the loss will multiplied by the weight.
            Default: 0.
        criterion (str): Criterion used for perceptual loss. Default: 'l1'.
    """

    def __init__(self,
                 layer_weights,
                 vgg_type='vgg19',
                 use_input_norm=True,
                 range_norm=False,
                 perceptual_weight=1.0,
                 style_weight=0.,
                 criterion='l1'):
        super(PerceptualLoss, self).__init__()
        self.perceptual_weight = perceptual_weight
        self.style_weight = style_weight
        self.layer_weights = layer_weights
        self.vgg = VGGFeatureExtractor(
            layer_name_list=list(layer_weights.keys()),
            vgg_type=vgg_type,
            use_input_norm=use_input_norm,
            range_norm=range_norm)

        self.criterion_type = criterion
        if self.criterion_type == 'l1':
            self.criterion = torch.nn.L1Loss()
        elif self.criterion_type == 'l2':
            self.criterion = torch.nn.MSELoss()
        elif self.criterion_type == 'fro':
            self.criterion = None
        else:
            raise NotImplementedError(f'{criterion} criterion has not been supported.')

    def forward(self, x, gt):
        """Forward function.

        Args:
            x (Tensor): Input tensor with shape (n, c, h, w).
            gt (Tensor): Ground-truth tensor with shape (n, c, h, w).

        Returns:
            Tensor: Forward results.
        """
        # extract vgg features
        x_features = self.vgg(x)
        gt_features = self.vgg(gt.detach())

        # calculate perceptual loss
        if self.perceptual_weight > 0:
            percep_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    percep_loss += torch.norm(x_features[k] - gt_features[k], p='fro') * self.layer_weights[k]
                else:
                    percep_loss += self.criterion(x_features[k], gt_features[k]) * self.layer_weights[k]
            percep_loss *= self.perceptual_weight
        else:
            percep_loss = None

        # calculate style loss
        if self.style_weight > 0:
            style_loss = 0
            for k in x_features.keys():
                if self.criterion_type == 'fro':
                    style_loss += torch.norm(
                        self._gram_mat(x_features[k]) - self._gram_mat(gt_features[k]), p='fro') * self.layer_weights[k]
                else:
                    style_loss += self.criterion(self._gram_mat(x_features[k]), self._gram_mat(
                        gt_features[k])) * self.layer_weights[k]
            style_loss *= self.style_weight
        else:
            style_loss = None

        return percep_loss, style_loss

    def _gram_mat(self, x):
        """Calculate Gram matrix.

        Args:
            x (torch.Tensor): Tensor with shape of (n, c, h, w).

        Returns:
            torch.Tensor: Gram matrix.
        """
        n, c, h, w = x.size()
        features = x.view(n, c, w * h)
        features_t = features.transpose(1, 2)
        gram = features.bmm(features_t) / (c * h * w)
        return gram
