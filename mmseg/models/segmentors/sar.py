# Copyright (c) OpenMMLab. All rights reserved.
import torch
import torch.nn as nn
import torch.nn.functional as F

from mmseg.core import add_prefix
from mmseg.ops import resize
from .. import builder
from ..builder import SEGMENTORS
from .encoder_decoder import EncoderDecoder
from mmcv.cnn import ConvModule
from mmcv.cnn import build_norm_layer

@SEGMENTORS.register_module()
class SAR(EncoderDecoder):

    def __init__(self,
                 backbone,
                 decode_head,
                 neck=None,
                 auxiliary_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 init_cfg=None,
                 cfg=None):
        super(SAR, self).__init__(
            backbone=backbone,
            decode_head=decode_head,
            neck=neck,
            auxiliary_head=auxiliary_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            pretrained=pretrained,
            init_cfg=init_cfg)
        
        self.loss_type=cfg['loss_type'],
        self.filter_conf=cfg['filter_conf']
        self.update_conf=cfg['update_conf']
        self.trade_off_ce = cfg['trade_off_ce']
        self.trade_off = cfg['trade_off']
        self.loss_fn = torch.nn.MSELoss(reduction='none')
        if type(self.decode_head) == torch.nn.modules.container.ModuleList:
            self.class_numbers = self.decode_head[-1].num_classes
            self.feat_channel = self.decode_head[-1].channels
        else:
            self.class_numbers = self.decode_head.num_classes
            self.feat_channel = self.decode_head.channels
        self.register_buffer('random_center', torch.randn(self.class_numbers, self.feat_channel))
        self.register_buffer('center_lbl', torch.arange(self.num_classes))
        self.register_buffer('local_iter', torch.arange(1))
        norm_cfg={'type': 'SyncBN', 'requires_grad': True}
        self.projecter_linear = nn.Sequential(nn.Linear(self.feat_channel, self.feat_channel), 
                                        build_norm_layer(norm_cfg, self.feat_channel)[1],
                                        nn.ReLU(inplace=True), 
                                        nn.Linear(self.feat_channel, self.feat_channel), 
                                        build_norm_layer(norm_cfg, self.feat_channel)[1], 
                                        nn.ReLU(inplace=True))
        self.projecter = ConvModule(
                    self.feat_channel,
                    self.feat_channel,
                    kernel_size=1,
                    norm_cfg={'type': 'SyncBN', 'requires_grad': True},
                    act_cfg={'type': 'ReLU'})
        
        self.register_buffer('update_flag', torch.tensor(0))
        self.register_buffer('momentum_center_feat', torch.zeros(self.num_classes, self.feat_channel))
        self.masked_label = 255



    def cal_feat_loss(self, mini_gt, mask, pred_feat, center):
        loss_dict = {}
    
        center = self.projecter_linear(center).unsqueeze(2).unsqueeze(2)
        center_feat = self.projecter(center)

        center_logits = self.decode_head.conv_seg(center_feat)
        center_ce_mat = F.cross_entropy(center_logits, self.center_lbl.unsqueeze(1).unsqueeze(1), reduction='none')
        center_ce = center_ce_mat.flatten()

        prob = torch.softmax(center_logits.flatten(-3), dim=1).diag()
        prob[prob > self.filter_conf] = 1
        w = torch.log(prob) / (torch.log(prob).sum() + 1e-9)
        # filter
        loss_ce = (center_ce * w).sum()
        loss_dict['center_ce_loss'] = loss_ce * self.trade_off_ce

        pred_lbl = torch.argmax(center_logits, dim=1).flatten()
        right_pred_anchor = pred_lbl == self.center_lbl 

        if (right_pred_anchor.sum() == self.num_classes) and \
            ( (center_ce_mat < -torch.log(torch.tensor(self.update_conf))).sum() == self.num_classes ):
            if self.update_flag == 0:
                self.momentum_center_feat = center_feat.clone().detach().flatten(-3).requires_grad_(False)
                self.update_flag = torch.tensor(1).to(self.update_flag.device)
            else:
                self.momentum_center_feat = 0.999*self.momentum_center_feat + 0.001*center_feat.clone().detach().flatten(-3).requires_grad_(False)

            one_hot_mini_lbl = self.process_label(mini_gt)[:,:-1,:,:].permute(0,2,3,1) #b,h,w,cls
            sup_feat_map = one_hot_mini_lbl @ (self.momentum_center_feat.clone().detach()) # b,h,w,channel
            sup_feat_map = sup_feat_map.permute(0,3,1,2)

            trade_off = self.trade_off     # item

            feat_loss_mat = self.loss_fn(pred_feat, sup_feat_map) 
            feat_loss = (feat_loss_mat * mask).sum() / feat_loss_mat.shape[1] 

            if mask.sum() > 0 and feat_loss != 0:
                feat_loss = feat_loss.sum() / mask.sum()
                feat_loss = feat_loss * trade_off

            loss_dict['feature_loss'] = feat_loss

        return loss_dict, pred_feat 
    

    def forward_train(self, img, img_metas, gt_semantic_seg):
        """Forward function for training.

        Args:
            img (Tensor): Input images.
            img_metas (list[dict]): List of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmseg/datasets/pipelines/formatting.py:Collect`.
            gt_semantic_seg (Tensor): Semantic segmentation masks
                used if the architecture supports semantic segmentation task.

        Returns:
            dict[str, Tensor]: a dictionary of loss components
        """

        x = self.extract_feat(img)
        losses = dict()
        loss_decode = self._decode_head_forward_train(x, img_metas,
                                                      gt_semantic_seg)
        main_pred_feat = loss_decode.pop('decode.sup_feat')
        seg_logits = loss_decode.pop('decode.seg_logits')
        losses.update(loss_decode)
        mini_gt = F.interpolate(
            input=gt_semantic_seg.float(),
            size=main_pred_feat.shape[2:],
            mode='nearest')

        mask = (mini_gt != 255) 
        if type(self.masked_label) == list:
            for label in self.masked_label:
                mask = (mask & (mini_gt != label))
        elif type(self.masked_label) == int:
            mask = mask & (mini_gt!=self.masked_label)

        loss_dict, perd_feat = self.cal_feat_loss(mask=mask, mini_gt=mini_gt, pred_feat=main_pred_feat, center=self.random_center)
        losses.update(add_prefix(loss_dict, 'main'))

        if self.with_auxiliary_head:
            loss_aux = self._auxiliary_head_forward_train(
                x, img_metas, gt_semantic_seg)
            losses.update(loss_aux)
        self.local_iter = self.local_iter + 1
 
        return losses
    def process_label(self, label):        
        batch, _, w, h = label.size()
        pred1 = torch.zeros(batch, self.class_numbers + 1, w, h).cuda()
        idx = torch.where(
            label < self.class_numbers,
            label,
            torch.Tensor([self.class_numbers]).cuda(),
        )
        pred1 = pred1.scatter_(1, idx.long(), 1)
        return pred1
    
    def simple_test(self, img, img_meta, rescale=True):
        """Simple test with single image."""
        seg_logit, feat = self.inference(img, img_meta, rescale)
        if self.out_channels == 1:
            seg_pred = (seg_logit >
                        self.decode_head.threshold).to(seg_logit).squeeze(1)
        else:
            seg_pred = seg_logit.argmax(dim=1)
        if torch.onnx.is_in_onnx_export():
            # our inference backend only support 4D output
            seg_pred = seg_pred.unsqueeze(0)
            return seg_pred
        seg_pred = seg_pred.cpu().numpy()
        # unravel batch dim
        seg_pred = list(seg_pred)
        return seg_pred
    
    def encode_decode(self, img, img_metas):
        """Encode images with backbone and decode into a semantic segmentation
        map of the same size as input."""
        x = self.extract_feat(img)
        out, feat = self._decode_head_forward_test(x, img_metas)
        out = resize(
            input=out,
            size=img.shape[2:],
            mode='bilinear',
            align_corners=self.align_corners)
        return out, feat
    
        # TODO refactor
    def slide_inference(self, img, img_meta, rescale):
        h_stride, w_stride = self.test_cfg.stride
        h_crop, w_crop = self.test_cfg.crop_size
        batch_size, _, h_img, w_img = img.size()
        out_channels = self.out_channels
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        preds = img.new_zeros((batch_size, out_channels, h_img, w_img))
        count_mat = img.new_zeros((batch_size, 1, h_img, w_img))
        feat = img.new_zeros((batch_size, self.feat_channel, h_img, w_img))
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                crop_seg_logit, crop_feat = self.encode_decode(crop_img, img_meta)
                preds += F.pad(crop_seg_logit,
                               (int(x1), int(preds.shape[3] - x2), int(y1),
                                int(preds.shape[2] - y2)))
                crop_feat = resize(
                            input=crop_feat,
                            size=crop_seg_logit.shape[2:],
                            mode='bilinear',
                            align_corners=self.align_corners)
                feat += F.pad(crop_feat,
                              (int(x1), int(preds.shape[3] - x2), int(y1),
                               int(preds.shape[2] - y2)))
                count_mat[:, :, y1:y2, x1:x2] += 1
        assert (count_mat == 0).sum() == 0
        if torch.onnx.is_in_onnx_export():
            # cast count_mat to constant while exporting to ONNX
            count_mat = torch.from_numpy(
                count_mat.cpu().detach().numpy()).to(device=img.device)
        preds = preds / count_mat
        feat = feat / count_mat
        if rescale:
            # remove padding area
            resize_shape = img_meta[0]['img_shape'][:2]
            preds = preds[:, :, :resize_shape[0], :resize_shape[1]]
            preds = resize(
                preds,
                size=img_meta[0]['ori_shape'][:2],
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

            feat = feat[:, :, :resize_shape[0], :resize_shape[1]]
            import math 
            feat_shape = (math.ceil(resize_shape[0]/8), math.ceil(resize_shape[1]/8)) 
            feat = resize(
                feat,
                size=feat_shape,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)
        
        return preds, feat
        
    def whole_inference(self, img, img_meta, rescale):
        """Inference with full image."""

        seg_logit, feat = self.encode_decode(img, img_meta)
        if rescale:
            # support dynamic shape for onnx
            if torch.onnx.is_in_onnx_export():
                size = img.shape[2:]
            else:   # not appear
                # remove padding area
                resize_shape = img_meta[0]['img_shape'][:2]
                seg_logit = seg_logit[:, :, :resize_shape[0], :resize_shape[1]]
                size = img_meta[0]['ori_shape'][:2]
            seg_logit = resize(
                seg_logit,
                size=size,
                mode='bilinear',
                align_corners=self.align_corners,
                warning=False)

        return seg_logit, feat
    

    def inference(self, img, img_meta, rescale):
        assert self.test_cfg.mode in ['slide', 'whole']
        ori_shape = img_meta[0]['ori_shape']
        assert all(_['ori_shape'] == ori_shape for _ in img_meta)
        if self.test_cfg.mode == 'slide':
            seg_logit, feat = self.slide_inference(img, img_meta, rescale)
        else:
            seg_logit, feat = self.whole_inference(img, img_meta, rescale)
        if self.out_channels == 1:
            output = F.sigmoid(seg_logit)
        else:
            output = F.softmax(seg_logit, dim=1)
        flip = img_meta[0]['flip']
        if flip:
            flip_direction = img_meta[0]['flip_direction']
            assert flip_direction in ['horizontal', 'vertical']
            if flip_direction == 'horizontal':
                output = output.flip(dims=(3, ))
                feat = feat.flip(dims=(3,))
            elif flip_direction == 'vertical':
                output = output.flip(dims=(2, ))
                feat = feat.flip(dims=(2,))

        return output, feat
    