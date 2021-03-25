from __future__ import division

import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import normal_init

from mmdet.core import (PointGenerator, multi_apply, multiclass_nms_kp,
                        point_target_kp)
from mmdet.ops import DeformConv
from ..builder import build_loss
from ..registry import HEADS
from ..utils import ConvModule, bias_init_with_prob


class MultiColumnDeformConvBlock(nn.Module):
    def __init__(self,
                 in_channels=256,
                 feat_channels=256,
                 gradient_mul=0.1):
        super().__init__()
        self.gradient_mul = gradient_mul
        self.deform_offset_dim = 2 * (9 + 25 + 49)

        # initializae dcn base offset
        # DeformConv3x3
        self.dcn_kernel_3 = int(np.sqrt(9))
        self.dcn_pad_3 = int((self.dcn_kernel_3 - 1) / 2)
        dcn_base_3 = np.arange(-self.dcn_pad_3,
                               self.dcn_pad_3 + 1).astype(np.float64)
        dcn_base_y_3 = np.repeat(dcn_base_3, self.dcn_kernel_3)
        dcn_base_x_3 = np.tile(dcn_base_3, self.dcn_kernel_3)
        dcn_base_offset_3 = np.stack(
            [dcn_base_y_3, dcn_base_x_3], axis=1).reshape((-1))
        self.dcn_base_offset_3 = torch.tensor(dcn_base_offset_3).view(
            1, -1, 1, 1)
        # DeformConv5x5
        self.dcn_kernel_5 = int(np.sqrt(25))
        self.dcn_pad_5 = int((self.dcn_kernel_5 - 1) / 2)
        dcn_base_5 = np.arange(-self.dcn_pad_5,
                               self.dcn_pad_5 + 1).astype(np.float64)
        dcn_base_y_5 = np.repeat(dcn_base_5, self.dcn_kernel_5)
        dcn_base_x_5 = np.tile(dcn_base_5, self.dcn_kernel_5)
        dcn_base_offset_5 = np.stack(
            [dcn_base_y_5, dcn_base_x_5], axis=1).reshape((-1))
        self.dcn_base_offset_5 = torch.tensor(dcn_base_offset_5).view(1, -1, 1, 1)
        # DeformConv7x7
        self.dcn_kernel_7 = int(np.sqrt(49))
        self.dcn_pad_7 = int((self.dcn_kernel_7 - 1) / 2)
        dcn_base_7 = np.arange(-self.dcn_pad_7,
                               self.dcn_pad_7 + 1).astype(np.float64)
        dcn_base_y_7 = np.repeat(dcn_base_7, self.dcn_kernel_7)
        dcn_base_x_7 = np.tile(dcn_base_7, self.dcn_kernel_7)
        dcn_base_offset_7 = np.stack(
            [dcn_base_y_7, dcn_base_x_7], axis=1).reshape((-1))
        self.dcn_base_offset_7 = torch.tensor(dcn_base_offset_7).view(
            1, -1, 1, 1)

        # initialize Layers
        self.dfmconv_3 = DeformConv(in_channels, feat_channels,
                                    self.dcn_kernel_3, 1,
                                    self.dcn_pad_3)
        self.dfmconv_5 = DeformConv(in_channels, feat_channels,
                                    self.dcn_kernel_5, 1,
                                    self.dcn_pad_5)
        self.dfmconv_7 = DeformConv(in_channels, feat_channels,
                                    self.dcn_kernel_7, 1,
                                    self.dcn_pad_7)

        # initialize weights
        normal_init(self.dfmconv_3, std=0.01)
        normal_init(self.dfmconv_5, std=0.01)
        normal_init(self.dfmconv_7, std=0.01)

    def forward(self, feat, deform_offset):
        assert deform_offset.size(1) == self.deform_offset_dim
        deform_offset_3 = deform_offset[:, :2*9, :, :]
        deform_offset_5 = deform_offset[:, 2*9:2*(9+25), :, :]
        deform_offset_7 = deform_offset[:, 2*(9+25):2*(9+25+49), :, :]

        dcn_base_offset_3 = self.dcn_base_offset_3.type_as(feat)
        dcn_base_offset_5 = self.dcn_base_offset_5.type_as(feat)
        dcn_base_offset_7 = self.dcn_base_offset_7.type_as(feat)

        dcn_offset_grad_mul_3 = self.gradient_mul * deform_offset_3 \
            + (1 - self.gradient_mul) * deform_offset_3.detach()
        dcn_offset_3 = dcn_offset_grad_mul_3 - dcn_base_offset_3
        dcn_offset_grad_mul_5 = self.gradient_mul * deform_offset_5 \
            + (1 - self.gradient_mul) * deform_offset_5.detach()
        dcn_offset_5 = dcn_offset_grad_mul_5 - dcn_base_offset_5
        dcn_offset_grad_mul_7 = self.gradient_mul * deform_offset_7 \
            + (1 - self.gradient_mul) * deform_offset_7.detach()
        dcn_offset_7 = dcn_offset_grad_mul_7 - dcn_base_offset_7

        dfmconv_feat_3 = self.dfmconv_3(feat, dcn_offset_3)
        dfmconv_feat_5 = self.dfmconv_5(feat, dcn_offset_5)
        dfmconv_feat_7 = self.dfmconv_7(feat, dcn_offset_7)
        dfmconv_feat = torch.cat(
            [dfmconv_feat_3, dfmconv_feat_5, dfmconv_feat_7],
            dim=1)

        return dfmconv_feat


class KpDetModule(nn.Module):
    """ Sequential Block
    """
    def __init__(self,
                 deform_conv,
                 cls_out_channels,
                 in_channels=256,
                 feat_channels=256,
                 num_reppts=9,
                 num_keypts=17,
                 gradient_mul=0.1,
                 transform_method='minmax',
                 moment_mul=0.01):
        super().__init__()
        self.deform_conv = deform_conv
        self.gradient_mul = gradient_mul
        self.transform_method = transform_method
        self.moment_mul = moment_mul
        keypts_out_dim = 2 * num_keypts
        deform_offset_dim = 2 * (9 + 25 + 49)
        self.relu = nn.ReLU(inplace=False)

        # initiate conv layers
        if deform_conv:
            self.deform_offset_out = nn.Conv2d(
                keypts_out_dim, deform_offset_dim, 1, 1, 0)

            self.cls_dfm_block = MultiColumnDeformConvBlock(
                in_channels, feat_channels, gradient_mul)
            self.cls_out = nn.Conv2d(
                feat_channels*3, cls_out_channels, 1, 1, 0)

            self.bbox_param_dfm_block = MultiColumnDeformConvBlock(
                in_channels, feat_channels, gradient_mul)
            self.bbox_param_out = nn.Conv2d(
                feat_channels*3, 4, 1, 1, 0)

            self.kpt_dfm_block = MultiColumnDeformConvBlock(
                in_channels, feat_channels, gradient_mul)
            self.kpt_out = nn.Conv2d(
                feat_channels*3, keypts_out_dim, 1, 1, 0)

        else:
            self.cls_conv = nn.Conv2d(
                in_channels, feat_channels, 3, 1, 1)
            self.cls_out = nn.Conv2d(
                feat_channels, cls_out_channels, 1, 1, 0)

            self.bbox_param_conv = nn.Conv2d(
                in_channels, feat_channels, 3, 1, 1)
            self.bbox_param_out = nn.Conv2d(
                feat_channels, 4, 1, 1, 0)

            self.kpt_conv = nn.Conv2d(
                in_channels, feat_channels, 3, 1, 1)
            self.kpt_out = nn.Conv2d(
                feat_channels, keypts_out_dim, 1, 1, 0)

        # init weights
        bias_cls = bias_init_with_prob(0.01)

        if self.deform_conv:
            normal_init(self.deform_offset_out, std=0.01)
        else:
            normal_init(self.cls_conv, std=0.01)
            normal_init(self.bbox_param_conv, std=0.01)
            normal_init(self.kpt_conv, std=0.01)
        normal_init(self.cls_out, std=0.01, bias=bias_cls)
        normal_init(self.bbox_param_out, std=0.01)
        normal_init(self.kpt_out, std=0.01)

    def forward(self, cls_feat, pts_feat, kpt_offset_prev=None):
        if self.deform_conv:
            deform_offset = self.deform_offset_out(kpt_offset_prev)

            cls_dfm_feat = self.relu(
                self.cls_dfm_block(cls_feat, deform_offset))
            cls_score_map = self.cls_out(cls_dfm_feat)

            kpt_dfm_feat = self.relu(
                self.kpt_dfm_block(pts_feat, deform_offset))
            kpt_offset = self.kpt_out(kpt_dfm_feat)
            kpt_offset = kpt_offset + kpt_offset_prev.detach()

            bbox_param_dfm_feat = self.relu(
                self.bbox_param_dfm_block(pts_feat, deform_offset))
            bbox_param = self.bbox_param_out(bbox_param_dfm_feat)
            bbox_offset = self.points2bbox(
                kpt_offset.detach(), tranfer_param=bbox_param)
        else:
            cls_score_map = self.cls_out(self.relu(self.cls_conv(cls_feat)))

            kpt_offset = self.kpt_out(self.relu(
                self.kpt_conv(pts_feat)))

            bbox_param = self.bbox_param_out(self.relu(
                self.bbox_param_conv(pts_feat)))
            bbox_offset = self.points2bbox(
                kpt_offset.detach(), tranfer_param=bbox_param)

        return cls_score_map, bbox_offset, kpt_offset

    def points2bbox(self, pts, y_first=True, tranfer_param=None):
        """
        Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :param transfer_param:
            size: [B, 4, H, W]
            Meaning of each channel:
                - translate_x
                - translate_y
                - scale_x
                - scale_y
        :return: each points set is converting to a bbox [x1, y1, x2, y2].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom],
                             dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + (
                self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                             dim=1)
        elif self.transform_method == 'minmax_param':
            assert tranfer_param is not None
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_top = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]

            bbox_center_x = (bbox_left + bbox_right) / 2
            bbox_center_y = (bbox_top + bbox_bottom) / 2

            tranfer_param = (tranfer_param * self.moment_mul) + (
                tranfer_param.detach() * (1 - self.moment_mul))
            half_width = (bbox_center_x - bbox_left) * torch.exp(
                tranfer_param[:, 0:1, :, :])
            half_height = (bbox_center_y - bbox_top) * torch.exp(
                tranfer_param[:, 1:2, :, :])
            bbox_center_x = bbox_center_x + tranfer_param[:, 2:3, :, :]
            bbox_center_y = bbox_center_y + tranfer_param[:, 3:4, :, :]

            bbox = torch.cat([
                bbox_center_x - half_width, bbox_center_y - half_height,
                bbox_center_x + half_width, bbox_center_y + half_height
            ],
                             dim=1)
        elif self.transform_method == 'moment_param':
            assert tranfer_param is not None
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)

            tranfer_param = (tranfer_param * self.moment_mul) + (
                tranfer_param.detach() * (1 - self.moment_mul))

            pts_x_mean = pts_x_mean + tranfer_param[:, 0:1, :, :]
            pts_y_mean = pts_y_mean + tranfer_param[:, 1:2, :, :]

            half_width = pts_x_std * torch.exp(tranfer_param[:, 2:3, :, :])
            half_height = pts_y_std * torch.exp(tranfer_param[:, 3:4, :, :])
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ],
                             dim=1)
        else:
            raise NotImplementedError
        return bbox


@HEADS.register_module
class CascadeKpDetHead(nn.Module):
    """RepPoint head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        transform_method (str): The methods to transform RepPoints to bbox.
    """  # noqa: W605

    def __init__(self,
                 num_classes,
                 in_channels,
                 feat_channels=256,
                 point_feat_channels=256,
                 stacked_convs=3,
                 num_reppts=9,
                 num_keypts=17,
                 gradient_mul=0.1,
                 point_strides=[8, 16, 32, 64, 128],
                 point_base_scale=4,
                 conv_cfg=None,
                 norm_cfg=None,
                 loss_cls_1=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.5),
                 loss_cls_2=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=0.5),
                 loss_cls_3=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_1=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_2=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_3=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_kpt_1=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_kpt_2=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_kpt_3=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 use_grid_points=False,
                 center_init=True,
                 transform_method='moment',
                 moment_mul=0.01):
        super().__init__()
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.feat_channels = feat_channels
        self.point_feat_channels = point_feat_channels
        self.stacked_convs = stacked_convs
        self.num_keypts = num_keypts
        self.num_reppts = 9+25+49
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls_3.get('use_sigmoid', False)
        self.sampling = loss_cls_3['type'] not in ['FocalLoss']
        self.loss_cls_1 = build_loss(loss_cls_1)
        self.loss_cls_2 = build_loss(loss_cls_2)
        self.loss_cls_3 = build_loss(loss_cls_3)
        self.loss_bbox_1 = build_loss(loss_bbox_1)
        self.loss_bbox_2 = build_loss(loss_bbox_2)
        self.loss_bbox_3 = build_loss(loss_bbox_3)
        self.loss_kpt_1 = build_loss(loss_kpt_1)
        self.loss_kpt_2 = build_loss(loss_kpt_2)
        self.loss_kpt_3 = build_loss(loss_kpt_3)
        self.use_grid_points = use_grid_points
        self.center_init = center_init
        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
        self.moment_mul = moment_mul
        if self.use_sigmoid_cls:
            self.cls_out_channels = self.num_classes - 1
        else:
            self.cls_out_channels = self.num_classes
        self.point_generators = [PointGenerator() for _ in self.point_strides]

        self._init_layers()

    def _init_layers(self):
        self.relu = nn.ReLU(inplace=False)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        # stage 1
        self.kp_det_1 = KpDetModule(
            False,
            self.cls_out_channels,
            self.feat_channels,
            self.point_feat_channels,
            self.num_reppts,
            self.num_keypts,
            self.gradient_mul,
            self.transform_method,
            self.moment_mul)
        # stage 2
        self.kp_det_2 = KpDetModule(
            True,
            self.cls_out_channels,
            self.feat_channels,
            self.point_feat_channels,
            self.num_reppts,
            self.num_keypts,
            self.gradient_mul,
            self.transform_method,
            self.moment_mul)
        # stage 3
        self.kp_det_3 = KpDetModule(
            True,
            self.cls_out_channels,
            self.feat_channels,
            self.point_feat_channels,
            self.num_reppts,
            self.num_keypts,
            self.gradient_mul,
            self.transform_method,
            self.moment_mul)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)

    def points2kpt(self, pts, y_first=True):
        """
        Converting the points set into keypoints.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
        :return: each points set is converting to keypoint list
            [x1, y1, x2, y2 ... xk, yk].
        """
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1,
                                                                      ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0,
                                                                      ...]
        pts = torch.cat([pts_x, pts_y], dim=2).view(*pts.shape)
        return pts

    def forward_single(self, x):

        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)

        # stage 1
        cls_score_map_1, box_offset_1, kpt_offset_1 = \
            self.kp_det_1(cls_feat, pts_feat)

        # stage 2
        cls_score_map_2, box_offset_2, kpt_offset_2 = \
            self.kp_det_2(cls_feat, pts_feat, kpt_offset_1)

        # stage 3
        cls_score_map_3, box_offset_3, kpt_offset_3 = \
            self.kp_det_3(cls_feat, pts_feat, kpt_offset_2)

        return (cls_score_map_1, cls_score_map_2, cls_score_map_3,
                box_offset_1, box_offset_2, box_offset_3,
                kpt_offset_1, kpt_offset_2, kpt_offset_3)

    def forward(self, feats):
        return multi_apply(self.forward_single, feats)

    def get_points(self, featmap_sizes, img_metas):
        """Get points according to feature map sizes.

        Args:
            featmap_sizes (list[tuple]): Multi-level feature map sizes.
            img_metas (list[dict]): Image meta info.

        Returns:
            tuple: points of each image, valid flags of each image
        """
        num_imgs = len(img_metas)
        num_levels = len(featmap_sizes)

        # since feature map sizes of all images are the same, we only compute
        # points center for one time
        multi_level_points = []
        for i in range(num_levels):
            points = self.point_generators[i].grid_points(
                featmap_sizes[i], self.point_strides[i])
            multi_level_points.append(points)
        points_list = [[point.clone() for point in multi_level_points]
                       for _ in range(num_imgs)]

        # for each image, we compute valid flags of multi level grids
        valid_flag_list = []
        for img_id, img_meta in enumerate(img_metas):
            multi_level_flags = []
            for i in range(num_levels):
                point_stride = self.point_strides[i]
                feat_h, feat_w = featmap_sizes[i]
                h, w, _ = img_meta['pad_shape']
                valid_feat_h = min(int(np.ceil(h / point_stride)), feat_h)
                valid_feat_w = min(int(np.ceil(w / point_stride)), feat_w)
                flags = self.point_generators[i].valid_flags(
                    (feat_h, feat_w), (valid_feat_h, valid_feat_w))
                multi_level_flags.append(flags)
            valid_flag_list.append(multi_level_flags)

        return points_list, valid_flag_list

    def offset_to_pts(self, center_list, pred_list, y_first=True):
        """Change from point offset to point coordinate.
        """
        num_points = pred_list[0].size(1) // 2
        pts_list = []
        for i_lvl in range(len(self.point_strides)):
            pts_lvl = []
            for i_img in range(len(center_list)):
                pts_center = center_list[i_img][i_lvl][:, :2].repeat(
                    1, num_points)
                pts_shift = pred_list[i_lvl][i_img]
                if y_first:
                    yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                        -1, 2 * num_points)
                    y_pts_shift = yx_pts_shift[..., 0::2]
                    x_pts_shift = yx_pts_shift[..., 1::2]
                    xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                    xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                else:
                    xy_pts_shift = pts_shift.permute(1, 2, 0).view(
                        -1, 2 * num_points)
                    xy_pts_shift = xy_pts_shift.view(*xy_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def loss_single(self, cls_score_1, cls_score_2, cls_score_3,
                    kpt_pred_1, kpt_pred_2, kpt_pred_3,
                    bbox_pred_1, bbox_pred_2, bbox_pred_3,
                    labels, label_weights,
                    bbox_gt, bbox_weights,
                    kpt_gt, kpt_weights,
                    stride, num_total_samples):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)

        cls_score_1 = cls_score_1.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        cls_score_2 = cls_score_2.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)
        cls_score_3 = cls_score_3.permute(0, 2, 3, 1).reshape(
            -1, self.cls_out_channels)

        loss_cls_1 = self.loss_cls_1(
            cls_score_1,
            labels,
            label_weights,
            avg_factor=num_total_samples)
        loss_cls_2 = self.loss_cls_2(
            cls_score_2,
            labels,
            label_weights,
            avg_factor=num_total_samples)
        loss_cls_3 = self.loss_cls_3(
            cls_score_3,
            labels,
            label_weights,
            avg_factor=num_total_samples)

        # bbox loss
        bbox_gt = bbox_gt.reshape(-1, 4)
        bbox_weights = bbox_weights.reshape(-1, 4)

        bbox_pred_1 = bbox_pred_1.reshape(-1, 4)
        bbox_pred_2 = bbox_pred_2.reshape(-1, 4)
        bbox_pred_3 = bbox_pred_3.reshape(-1, 4)

        normalize_term = self.point_base_scale * stride
        loss_bbox_1 = self.loss_bbox_1(
            bbox_pred_1 / normalize_term,
            bbox_gt / normalize_term,
            bbox_weights,
            avg_factor=num_total_samples)
        loss_bbox_2 = self.loss_bbox_2(
            bbox_pred_2 / normalize_term,
            bbox_gt / normalize_term,
            bbox_weights,
            avg_factor=num_total_samples)
        loss_bbox_3 = self.loss_bbox_3(
            bbox_pred_3 / normalize_term,
            bbox_gt / normalize_term,
            bbox_weights,
            avg_factor=num_total_samples)

        # keypoint loss
        kpt_gt = kpt_gt.reshape(-1, self.num_keypts * 2)
        kpt_weights = kpt_weights.reshape(-1, self.num_keypts * 2)
        kpt_pos_num = kpt_weights.sum(1)
        kpt_weights[kpt_pos_num > 0] /= kpt_pos_num[
            kpt_pos_num > 0].unsqueeze(1)
        kpt_weights *= 4

        kpt_pred_1 = kpt_pred_1.reshape(-1, self.num_keypts * 2)
        kpt_pred_2 = kpt_pred_2.reshape(-1, self.num_keypts * 2)
        kpt_pred_3 = kpt_pred_3.reshape(-1, self.num_keypts * 2)

        normalize_term = self.point_base_scale * stride
        loss_kpt_1 = self.loss_kpt_1(
            kpt_pred_1 / normalize_term,
            kpt_gt / normalize_term,
            kpt_weights,
            avg_factor=num_total_samples)
        loss_kpt_2 = self.loss_kpt_2(
            kpt_pred_2 / normalize_term,
            kpt_gt / normalize_term,
            kpt_weights,
            avg_factor=num_total_samples)
        loss_kpt_3 = self.loss_kpt_3(
            kpt_pred_3 / normalize_term,
            kpt_gt / normalize_term,
            kpt_weights,
            avg_factor=num_total_samples)
        return (loss_cls_1, loss_cls_2, loss_cls_3,
                loss_bbox_1, loss_bbox_2, loss_bbox_3,
                loss_kpt_1, loss_kpt_2, loss_kpt_3)

    def loss(self,
             cls_scores_1,
             cls_scores_2,
             cls_scores_3,
             box_offset_preds_1,
             box_offset_preds_2,
             box_offset_preds_3,
             kpt_preds_1,
             kpt_preds_2,
             kpt_preds_3,
             gt_bboxes,
             gt_labels,
             gt_keypoints,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores_3]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        # prediction of the 1st stage
        kpt_coord_preds_1 = self.offset_to_pts(center_list, kpt_preds_1)
        box_coord_preds_1 = self.offset_to_pts(
            center_list, box_offset_preds_1, y_first=False)

        # target for the 2nd stage
        kpt_coord_preds_2 = self.offset_to_pts(center_list, kpt_preds_2)
        box_coord_preds_2 = self.offset_to_pts(
            center_list, box_offset_preds_2, y_first=False)

        # target for the 3rd stage
        kpt_coord_preds_3 = self.offset_to_pts(center_list, kpt_preds_3)
        box_coord_preds_3 = self.offset_to_pts(
            center_list, box_offset_preds_3, y_first=False)

        # target for all stages
        if cfg.uniform.assigner['type'] == 'PointAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            raise(NotImplementedError)
        cls_reg_targets = point_target_kp(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            gt_keypoints,
            img_metas,
            cfg.uniform,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        (labels_list, label_weights_list,
         bbox_gt_list, candidate_list, bbox_weights_list,
         keypoint_gt_list, keypoint_weights_list,
         num_total_pos, num_total_neg) = cls_reg_targets
        num_total_samples = (
            num_total_pos +
            num_total_neg if self.sampling else num_total_pos)

        # compute loss
        (losses_cls_1, losses_cls_2, losses_cls_3,
         losses_bbox_1, losses_bbox_2, losses_bbox_3,
         losses_kpt_1, losses_kpt_2, losses_kpt_3) = multi_apply(
            self.loss_single,
            cls_scores_1,
            cls_scores_2,
            cls_scores_3,
            kpt_coord_preds_1,
            kpt_coord_preds_2,
            kpt_coord_preds_3,
            box_coord_preds_1,
            box_coord_preds_2,
            box_coord_preds_3,
            labels_list,
            label_weights_list,
            bbox_gt_list,
            bbox_weights_list,
            keypoint_gt_list,
            keypoint_weights_list,
            self.point_strides,
            num_total_samples=num_total_samples)
        loss_dict_all = {
            'loss_cls_1': losses_cls_1,
            'loss_cls_2': losses_cls_2,
            'loss_cls_3': losses_cls_3,
            'loss_bbox_1': losses_bbox_1,
            'loss_bbox_2': losses_bbox_2,
            'loss_bbox_3': losses_bbox_3,
            'loss_kpt_1': losses_kpt_1,
            'loss_kpt_2': losses_kpt_2,
            'loss_kpt_3': losses_kpt_3
        }
        return loss_dict_all

    def get_bboxes(self,
                   cls_scores_1,
                   cls_scores_2,
                   cls_scores_3,
                   box_offset_preds_1,
                   box_offset_preds_2,
                   box_offset_preds_3,
                   kpt_preds_1,
                   kpt_preds_2,
                   kpt_preds_3,
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True):

        cls_score_final = cls_scores_3
        keypts_preds_final = kpt_preds_3
        bbox_preds_final = box_offset_preds_3

        assert len(cls_score_final) == len(keypts_preds_final) \
            == len(bbox_preds_final)

        bbox_preds = bbox_preds_final
        kpt_preds = [
            self.points2kpt(keypts_pred)
            for keypts_pred in keypts_preds_final
        ]
        num_levels = len(cls_score_final)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_score_final[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_score_final[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds[i][img_id].detach()
                for i in range(num_levels)
            ]
            kpt_pred_list = [
                kpt_preds[i][img_id].detach()
                for i in range(num_levels)
            ]
            img_shape = img_metas[img_id]['img_shape']
            scale_factor = img_metas[img_id]['scale_factor']
            proposals = self.get_bboxes_single(cls_score_list, bbox_pred_list,
                                               kpt_pred_list,
                                               mlvl_points, img_shape,
                                               scale_factor, cfg, rescale, nms)
            result_list.append(proposals)
        return result_list

    def get_bboxes_single(self,
                          cls_scores,
                          bbox_preds,
                          kpt_preds,
                          mlvl_points,
                          img_shape,
                          scale_factor,
                          cfg,
                          rescale=False,
                          nms=True):
        assert len(cls_scores) == len(bbox_preds) == len(mlvl_points) \
                                                  == len(kpt_preds)
        mlvl_bboxes = []
        mlvl_kpts = []
        mlvl_scores = []
        num_kpt = self.num_keypts
        num_kp_channel = kpt_preds[0].size(0) // num_kpt
        assert num_kp_channel == 2 or num_kp_channel == 3
        for i_lvl, (cls_score, bbox_pred, kpt_pred, points) in enumerate(
                zip(cls_scores, bbox_preds, kpt_preds, mlvl_points)):
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:] \
                                         == kpt_pred.size()[-2:]
            cls_score = cls_score.permute(1, 2,
                                          0).reshape(-1, self.cls_out_channels)
            if self.use_sigmoid_cls:
                scores = cls_score.sigmoid()
            else:
                scores = cls_score.softmax(-1)
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
            if num_kp_channel == 3:
                kpt_pred = kpt_pred.permute(1, 2, 0).reshape(
                                -1, num_kpt * num_kp_channel)
            # if kpt visibility is not predicted, set it to 1
            elif num_kp_channel == 2:
                kpt_pred = kpt_pred.permute(1, 2, 0).reshape(
                                -1, num_kpt, num_kp_channel)
                pad_ones = kpt_pred.new_full(kpt_pred[:, :, :1].size(), 1)
                kpt_pred = torch.cat([kpt_pred, pad_ones], dim=2)
                kpt_pred = kpt_pred.reshape(-1, num_kpt * 3)
            nms_pre = cfg.get('nms_pre', -1)
            if nms_pre > 0 and scores.shape[0] > nms_pre:
                if self.use_sigmoid_cls:
                    max_scores, _ = scores.max(dim=1)
                else:
                    max_scores, _ = scores[:, 1:].max(dim=1)
                _, topk_inds = max_scores.topk(nms_pre)
                points = points[topk_inds, :]
                bbox_pred = bbox_pred[topk_inds, :]
                kpt_pred = kpt_pred[topk_inds, :]
                scores = scores[topk_inds, :]
            bbox_pos_center = torch.cat([points[:, :2], points[:, :2]], dim=1)
            kpt_pos_center = points[:, :2].unsqueeze(dim=1)
            bboxes = bbox_pred * self.point_strides[i_lvl] + bbox_pos_center
            kpt_pred = kpt_pred.view(-1, num_kpt, 3)
            kpt_pred[:, :, :2] = kpt_pred[:, :, :2] \
                * self.point_strides[i_lvl] + kpt_pos_center
            kpts = kpt_pred
            x1 = bboxes[:, 0].clamp(min=0, max=img_shape[1])
            y1 = bboxes[:, 1].clamp(min=0, max=img_shape[0])
            x2 = bboxes[:, 2].clamp(min=0, max=img_shape[1])
            y2 = bboxes[:, 3].clamp(min=0, max=img_shape[0])
            bboxes = torch.stack([x1, y1, x2, y2], dim=-1)
            kpts[:, 0::3] = kpts[:, 0::3].clamp(min=0, max=img_shape[1])
            kpts[:, 1::3] = kpts[:, 1::3].clamp(min=0, max=img_shape[0])
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            mlvl_kpts.append(kpts)
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_kpts = torch.cat(mlvl_kpts)
        if rescale:
            mlvl_bboxes /= mlvl_bboxes.new_tensor(scale_factor)
            mlvl_kpts[:, :, 0:2] = mlvl_kpts[:, :, 0:2] \
                / mlvl_kpts.new_tensor(scale_factor)
            mlvl_kpts = mlvl_kpts.reshape(-1, num_kpt*3)
        mlvl_scores = torch.cat(mlvl_scores)
        # kpt mAP increses after multipling bbox score
        if self.use_sigmoid_cls:
            padding = mlvl_scores.new_zeros(mlvl_scores.shape[0], 1)
            mlvl_scores = torch.cat([padding, mlvl_scores], dim=1)
        if nms:
            det_bboxes, det_labels, det_kpts = multiclass_nms_kp(
                                                    mlvl_bboxes,
                                                    mlvl_scores,
                                                    mlvl_kpts,
                                                    cfg.score_thr,
                                                    cfg.nms,
                                                    cfg.max_per_img)
            return det_bboxes, det_labels, det_kpts
        else:
            return mlvl_bboxes, mlvl_scores, mlvl_kpts
