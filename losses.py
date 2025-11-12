import torch
import torch.nn as nn
import torch.nn.functional as F
import pdb

try:
    from LovaszSoftmax.pytorch.lovasz_losses import lovasz_hinge
except ImportError:
    pass

__all__ = ['BCEDiceLoss', 'LovaszHingeLoss']
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class BCEDiceLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        pos_weight = torch.Tensor([300]).to(device)
        bce = torch.nn.BCEWithLogitsLoss(
                        pos_weight = pos_weight)(input, target)
        # bce = F.binary_cross_entropy_with_logits(input, target)
        #pdb.set_trace()
        smooth = 1e-5
        input = torch.sigmoid(input)
        num = target.reshape(1,-1).size(1)
        input = input.reshape(num, -1)
        target = target.reshape(num, -1)
        
        intersection = (input * target)
        dice = (2. * intersection.sum(1) + smooth) / (input.sum(1) + target.sum(1) + smooth)
        dice = 1 - dice.sum() / num
        return 0.5 * bce + dice


class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        input = input.squeeze(1)
        target = target.squeeze(1)
        loss = lovasz_hinge(input, target, per_image=True)

        return loss
    
class KnowledgeDistillationLoss(nn.Module):
    def __init__(self, temperature=1.0):
        super(KnowledgeDistillationLoss, self).__init__()
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits):
        # 计算温度调整后的软标签
        soft_teacher_logits = F.softmax(teacher_logits / self.temperature, dim=1)
        # 计算交叉熵损失
        loss = -torch.sum(soft_teacher_logits * F.log_softmax(student_logits / self.temperature, dim=1))
        # 返回损失
        return loss
