"""
This file contains specific functions for computing losses of SiamCAR
file
"""

import torch
from torch import nn
import numpy as np
import torch.nn.functional as F



INF = 100000000




def IoU(pred, target):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect

        iou = (area_intersect + 1) / (area_union + 1)
        return iou




def get_cls_loss(pred, label, select):
    if len(select.size()) == 0 or \
            select.size() == torch.Size([0]):
        return 0
    pred = torch.index_select(pred, 0, select)
    label = torch.index_select(label, 0, select)
    return F.nll_loss(pred, label)


def select_cross_entropy_loss(pred, label):
    pred = pred.view(-1, 2)
    label = label.view(-1)
    pos = label.data.eq(1).nonzero().squeeze().cuda()
    neg = label.data.eq(0).nonzero().squeeze().cuda()
    loss_pos = get_cls_loss(pred, label, pos)
    loss_neg = get_cls_loss(pred, label, neg)
    return loss_pos * 0.5 + loss_neg * 0.5


def weight_l1_loss(pred_loc, label_loc, loss_weight):
    b, _, sh, sw = pred_loc.size()
    pred_loc = pred_loc.view(b, 4, -1, sh, sw)
    diff = (pred_loc - label_loc).abs()
    diff = diff.sum(dim=1).view(b, -1, sh, sw)
    loss = diff * loss_weight
    return loss.sum().div(b)


class IOULoss(nn.Module):
    def forward(self, pred, target, weight):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        iou = (area_intersect + 1.0) / (area_union + 1.0)
        losses = -torch.log(iou)

        if weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class DIOULoss(nn.Module):
    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_aera = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_aera = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + \
                      torch.min(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + \
                      torch.min(pred_top, target_top)

        area_intersect = w_intersect * h_intersect
        area_union = target_aera + pred_aera - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)

        # cal outer boxes
        outer_w = torch.max(pred_left, target_left) + \
                      torch.max(pred_right, target_right)
        outer_h = torch.max(pred_bottom, target_bottom) + \
                      torch.max(pred_top, target_top)
        outer_diagonal_line = outer_w.pow(2) + outer_h.pow(2)
        # outer_w = np.maximum(outer_w, 0.0)
        # outer_h = np.maximum(outer_h, 0.0)
        # outer_diagonal_line = np.square(outer_w) + np.square(outer_h)

        # cal center distance
        boxes1_cx = (target_left + target_right) * 0.5
        boxes1_cy = (target_top + target_bottom) * 0.5
        boxes2_cx = (pred_left + pred_right) * 0.5
        boxes2_cy = (pred_top + pred_bottom) * 0.5
        center_dis = (boxes1_cx- boxes2_cx).pow(2) + (boxes1_cy - boxes2_cy).pow(2)

        # cal diou
        dious = ious - center_dis / outer_diagonal_line

        losses = 1 - dious
        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class GIOULoss(nn.Module):
    def __init__(self, loc_loss_type='giou'):
        super(GIOULoss, self).__init__()
        self.loc_loss_type = loc_loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
        target_area = (target_left + target_right) * (target_top + target_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion

        if self.loc_loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loc_loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loc_loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum() / weight.sum()
        else:
            assert losses.numel() != 0
            return losses.mean()


class SiamCARLossComputation(object):
    """
    This class computes the SiamCAR losses.
    """

    def __init__(self,cfg):
        # self.box_reg_loss_func = DIOULoss()
        self.box_reg_loss_func = IOULoss()
        self.centerness_loss_func = nn.BCEWithLogitsLoss()
        self.cfg = cfg
    def rank_cen_Loss(self,input, label):
        loss_all = []
        batch_size = input.shape[0]
        # view相当于reshape
        pred = input.view(batch_size, -1)
        label = label.view(batch_size, -1)
        for batch_id in range(batch_size):
            # label为1为正样本
            pos_index = np.where(label[batch_id].cpu() == 1)[0].tolist()
            # label为0为负样本
            neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
            # label 为 -1 的直接省略
            if len(pos_index) > 0:
                # 分别代表正负样本为前景的概率
                pos_prob = torch.exp(pred[batch_id][pos_index])
                neg_prob = torch.exp(pred[batch_id][neg_index])

                num_pos = len(pos_index)
                # 第0维降序排序
                neg_value, _ = neg_prob.sort(0, descending=True)
                pos_value, _ = pos_prob.sort(0, descending=True)
                # 筛选出难负样本
                neg_idx2 = neg_prob > 0.5  # 0.5
                if neg_idx2.sum() == 0:
                    continue

                # 保证正负样本的数量相同
                neg_value = neg_value[0:num_pos]
                pos_value = pos_value[0:num_pos]

                # 公式6,7
                neg_q = F.softmax(neg_value, dim=0)
                neg_dist = torch.sum(neg_value * neg_q)
                pos_dist = torch.sum(pos_value) / len(pos_value)

                # 对应paper中的rank_cls_los的公式8
                loss = torch.log(1. + torch.exp(4 * (neg_dist - pos_dist + 0.5))) / 4
            # 没有正样本,公式中的P^+为1

            else:
                neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
                neg_prob = torch.exp(pred[batch_id][neg_index])
                neg_value, _ = neg_prob.sort(0, descending=True)
                neg_idx2 = neg_prob > 0.5  # 0.5的视为难负样本
                if neg_idx2.sum() == 0:
                    continue
                num_neg = len(neg_prob[neg_idx2])
                num_neg = max(num_neg, 8)  # 8
                neg_value = neg_value[0:num_neg]
                neg_q = F.softmax(neg_value, dim=0)
                neg_dist = torch.sum(neg_value * neg_q)
                # 对应paper中的rank_cls_los的公式8
                loss = torch.log(1. + torch.exp(4 * (neg_dist - 1. + 0.5))) / 4

            loss_all.append(loss)
        if len(loss_all):
            final_loss = torch.stack(loss_all).mean()
        else:
            final_loss = torch.zeros(1).cuda()

        return final_loss
    #TODO : 增加计算IoU 1. 计算IoU ， 2. 传入cls, label_cls iou来计算rank Loss 3. 计算rank loss
    def rank_cls_Loss(self, input, label):
        loss_all = []
        batch_size = input.shape[0]
        # view相当于reshape
        pred = input.view(batch_size, -1, 2)
        label = label.view(batch_size, -1)
        for batch_id in range(batch_size):
            # label为1为正样本
            pos_index = np.where(label[batch_id].cpu() == 1)[0].tolist()
            # label为0为负样本
            neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
            # label 为 -1 的直接省略
            if len(pos_index) > 0:
                # 分别代表正负样本为前景的概率
                pos_prob = torch.exp(pred[batch_id][pos_index][:, 1])
                neg_prob = torch.exp(pred[batch_id][neg_index][:, 1])

                num_pos = len(pos_index)
                # 第0维降序排序
                neg_value, _ = neg_prob.sort(0, descending=True)
                pos_value, _ = pos_prob.sort(0, descending=True)
                # 筛选出难负样本
                neg_idx2 = neg_prob > 0.5 # 0.5
                if neg_idx2.sum() == 0:
                    continue

                # 保证正负样本的数量相同
                neg_value = neg_value[0:num_pos]
                pos_value = pos_value[0:num_pos]

                # 公式6,7
                neg_q = F.softmax(neg_value, dim=0)
                neg_dist = torch.sum(neg_value * neg_q)
                pos_dist = torch.sum(pos_value) / len(pos_value)

                # 对应paper中的rank_cls_los的公式8
                loss = torch.log(1. + torch.exp(4 * (neg_dist - pos_dist + 0.5))) / 4
            # 没有正样本,公式中的P^+为1

            else:
                neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
                neg_prob = torch.exp(pred[batch_id][neg_index][:, 1])
                neg_value, _ = neg_prob.sort(0, descending=True)
                neg_idx2 = neg_prob > 0.5 # 0.5的视为难负样本
                if neg_idx2.sum() == 0:
                    continue
                num_neg = len(neg_prob[neg_idx2])
                num_neg = max(num_neg, 8)  # 8
                neg_value = neg_value[0:num_neg]
                neg_q = F.softmax(neg_value, dim=0)
                neg_dist = torch.sum(neg_value * neg_q)
                # 对应paper中的rank_cls_los的公式8
                loss = torch.log(1. + torch.exp(4 * (neg_dist - 1. + 0.5))) / 4

            loss_all.append(loss)
        if len(loss_all):
            final_loss = torch.stack(loss_all).mean()
        else:
            final_loss = torch.zeros(1).cuda()

        return final_loss


    def prepare_targets(self, points, labels, gt_bbox, neg):

        labels, reg_targets, pos_area = self.compute_targets_for_locations(
            points, labels, gt_bbox, neg
        )

        return labels, reg_targets, pos_area

    def compute_targets_for_locations(self, locations, labels, gt_bbox, neg):
        # reg_targets = []
        # locations [625,2]
        xs, ys = locations[:, 0], locations[:, 1]

        bboxes = gt_bbox
        labels = labels.view(self.cfg.TRAIN.OUTPUT_SIZE**2, -1) #25
        pos_area = torch.zeros_like(labels)

        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets_per_im = torch.stack([l, t, r, b], dim=2)

##########################################points in the gt_bbox area################################################
        all_s1 = reg_targets_per_im[:, :, 0] > 0
        all_s2 = reg_targets_per_im[:, :, 2] > 0
        all_s3 = reg_targets_per_im[:, :, 1] > 0
        all_s4 = reg_targets_per_im[:, :, 3] > 0
        all_in_boxes = all_s1 * all_s2 * all_s3 * all_s4
        all_pos = np.where(all_in_boxes.cpu() == 1)
        pos_area[all_pos] = 1

##########################################ignore labels################################################
        ignore_s1 = reg_targets_per_im[:, :, 0] > 0.2 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        ignore_s2 = reg_targets_per_im[:, :, 2] > 0.2 * ((bboxes[:, 2] - bboxes[:, 0]) / 2).float()
        ignore_s3 = reg_targets_per_im[:, :, 1] > 0.2 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        ignore_s4 = reg_targets_per_im[:, :, 3] > 0.2 * ((bboxes[:, 3] - bboxes[:, 1]) / 2).float()
        ignore_in_boxes = ignore_s1 * ignore_s2 * ignore_s3 * ignore_s4
        ignore_pos = np.where(ignore_in_boxes.cpu() == 1)
        labels[ignore_pos] = -1

        s1 = reg_targets_per_im[:, :, 0] > 0.5*((bboxes[:,2]-bboxes[:,0])/2).float()
        s2 = reg_targets_per_im[:, :, 2] > 0.5*((bboxes[:,2]-bboxes[:,0])/2).float()
        s3 = reg_targets_per_im[:, :, 1] > 0.5*((bboxes[:,3]-bboxes[:,1])/2).float()
        s4 = reg_targets_per_im[:, :, 3] > 0.5*((bboxes[:,3]-bboxes[:,1])/2).float()
        is_in_boxes = s1*s2*s3*s4
        pos = np.where(is_in_boxes.cpu() == 1)
        labels[pos] = 1
        labels = labels * (1 - neg.long())

        return labels.permute(1, 0).contiguous(), reg_targets_per_im.permute(1, 0, 2).contiguous(), pos_area.permute(1, 0).contiguous()

    def compute_centerness_targets(self, reg_targets):
        left_right = reg_targets[:, [0, 2]]
        top_bottom = reg_targets[:, [1, 3]]
        # 中心度的计算方式, 可以改为IoU,对于小目标有效果
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                      (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def __call__(self, locations, box_cls, box_regression, centerness, labels, reg_targets, neg):
        """
        Arguments:
            locations (list[BoxList]) [625,2]
            box_cls (list[Tensor]) [26,1,25,25,2]
            box_regression (list[Tensor]) [26,4,25,25]
            centerness (list[Tensor])[26,1,25,25]
            labels = lables_cls
            targets (list[BoxList])[26,4]
            neg:[26] == batch_size
        Returns:
            cls_loss (Tensor)
            reg_loss (Tensor)
            centerness_loss (Tensor)
        """
        # Cls_Rank_loss_1 = self.rank_cls_Loss(
        #     box_cls, labels
        # )
        label_cls, reg_targets, pos_area = self.prepare_targets(locations, labels, reg_targets, neg)

        box_regression_flatten = (box_regression.permute(0, 2, 3, 1).contiguous().view(-1, 4))#重新排列
        labels_flatten = (label_cls.view(-1))
        reg_targets_flatten = (reg_targets.view(-1, 4))
        centerness_flatten = (centerness.view(-1))

        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)

        ###########################change cen and reg area###################################
        pos_area_flatten = (pos_area.view(-1))
        all_pos_idx = torch.nonzero(pos_area_flatten > 0).squeeze(1)

        box_regression_flatten = box_regression_flatten[all_pos_idx]
        reg_targets_flatten = reg_targets_flatten[all_pos_idx]
        centerness_flatten = centerness_flatten[all_pos_idx]

        #####################################################################################

        # box_regression_flatten = box_regression_flatten[pos_inds]
        # reg_targets_flatten = reg_targets_flatten[pos_inds]
        # centerness_flatten = centerness_flatten[pos_inds]

        # TODO:修改损失对齐cen和p
        cls_loss = select_cross_entropy_loss(box_cls, labels_flatten)
        Cls_Rank_loss = self.rank_cls_Loss(
            box_cls, label_cls
        )

        if pos_inds.numel() > 0:
            # 这个位置的 centerness_targets也传入到了IoU Loss
            centerness_targets = self.compute_centerness_targets(reg_targets_flatten)
            # Cen_Rank_Loss = self.rank_cen_Loss(centerness,label_cls)
            reg_loss = self.box_reg_loss_func(
                box_regression_flatten,
                reg_targets_flatten,
                centerness_targets
            )
            centerness_loss = self.centerness_loss_func(
                centerness_flatten,
                centerness_targets
            )


        else:
            reg_loss = box_regression_flatten.sum()
            centerness_loss = centerness_flatten.sum()

        return cls_loss, reg_loss, centerness_loss ,Cls_Rank_loss #,Cen_Rank_Loss




#TODO : MyCode


class Cen_IoU_Loss(nn.Module):

    def forward(self
                ,centerness_flatten,  # [1107]
                centerness_targets,  # [1107]
                box_regression_flatten,  # [1107,4]
                reg_targets_flatten):  # [1107,4]

        """
            首先，给他们排序，记录从大到小的顺序
            其次，按照公式计算
        """
        loss_all_1 = []

        # 计算IoU
        iou = IoU(reg_targets_flatten, box_regression_flatten)

        iou_value, iou_index = iou.sort(0,descending=True)

        #los
        for i in range(iou.shape[0] - 1):
            input1 = torch.LongTensor(iou.shape[0] - i - 1)
            input2 = torch.LongTensor(iou.shape[0] - i - 1)
            index = 0
            for j in range((i + 1), iou.shape[0]):
                input1[index] = iou_index[i]
                input2[index] = iou_index[j]
                index = index + 1
            input1, input2 = input1.cuda(), input2.cuda()

            loss1 = torch.exp(-3 * centerness_flatten[input1] - centerness_flatten[input2]).mean()
            if torch.isnan(loss1):
                continue
            else:
                loss_all_1.append(loss1)
        if len(loss_all_1):
            final_loss1 = torch.stack(loss_all_1).mean()
        else:
            final_loss1 = torch.FloatTensor([0]).cuda()[0]
        return final_loss1




def make_siamcar_loss_evaluator(cfg):
    loss_evaluator = SiamCARLossComputation(cfg)
    return loss_evaluator

class Rank_CLS_Loss(nn.Module):
    def __init__(self, L=4, margin=0.5):
        super(Rank_CLS_Loss, self).__init__()
        self.margin = margin
        self.L = L

    # 传入的lable是cls_label,input是预测的标签
    def forward(self, input, label):
        loss_all = []
        batch_size = input.shape[0]
        # view相当于reshape
        pred = input.view(batch_size, -1, 2)
        label = label.view(batch_size, -1)
        for batch_id in range(batch_size):
            # label为1为正样本
            pos_index = np.where(label[batch_id].cpu() == 1)[0].tolist()
            # label为0为负样本
            neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
            # label 为 -1 的直接省略
            if len(pos_index) > 0:
                # 分别代表正负样本为前景的概率
                pos_prob = torch.exp(pred[batch_id][pos_index][:, 1])
                neg_prob = torch.exp(pred[batch_id][neg_index][:, 1])

                num_pos = len(pos_index)
                # 第0维降序排序
                neg_value, _ = neg_prob.sort(0, descending=True)
                pos_value, _ = pos_prob.sort(0, descending=True)
                # 筛选出难负样本
                neg_idx2 = neg_prob > 0.5 # 0.5
                if neg_idx2.sum() == 0:
                    continue

                # 保证正负样本的数量相同
                neg_value = neg_value[0:num_pos]
                pos_value = pos_value[0:num_pos]

                # 公式6,7
                neg_q = F.softmax(neg_value, dim=0)
                neg_dist = torch.sum(neg_value * neg_q)
                pos_dist = torch.sum(pos_value) / len(pos_value)

                # 对应paper中的rank_cls_los的公式8
                loss = torch.log(1. + torch.exp(self.L * (neg_dist - pos_dist + self.margin))) / self.L
            # 没有正样本,公式中的P^+为1

            else:
                neg_index = np.where(label[batch_id].cpu() == 0)[0].tolist()
                neg_prob = torch.exp(pred[batch_id][neg_index][:, 1])
                neg_value, _ = neg_prob.sort(0, descending=True)
                neg_idx2 = neg_prob > 0.5 # 0.5的视为难负样本
                if neg_idx2.sum() == 0:
                    continue
                num_neg = len(neg_prob[neg_idx2])
                num_neg = max(num_neg, 8)  # 8
                neg_value = neg_value[0:num_neg]
                neg_q = F.softmax(neg_value, dim=0)
                neg_dist = torch.sum(neg_value * neg_q)
                # 对应paper中的rank_cls_los的公式8
                loss = torch.log(1. + torch.exp(self.L * (neg_dist - 1. + self.margin))) / self.L

            loss_all.append(loss)
        if len(loss_all):
            final_loss = torch.stack(loss_all).mean()
        else:
            final_loss = torch.zeros(1).cuda()

        return final_loss

def rank_cls_loss():
    loss = Rank_CLS_Loss()
    return loss


#TEST
if __name__ == '__main__':
    a = np.array([1,2], [3,4])
    torch.tensor(a)
