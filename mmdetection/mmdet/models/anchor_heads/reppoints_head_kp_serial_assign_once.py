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


@HEADS.register_module
class RepPointsHeadKpSerialAssignOnce(nn.Module):
    """RepPoint head.

    Args:
        in_channels (int): Number of channels in the input feature map.
        feat_channels (int): Number of channels of the feature map.
        point_feat_channels (int): Number of channels of points features.
        stacked_convs (int): How many conv layers are used.
        gradient_mul (float): The multiplier to gradients from
            points refinement and recognition.
        point_strides (Iterable): points strides.
        point_base_scale (int): bbox scale for assigning labels.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of points loss in refinement.
        use_grid_points (bool): If we use bounding box representation, the
        reppoints is represented as grid points on the bounding box.
        center_init (bool): Whether to use center point assignment.
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
                 loss_cls=dict(
                     type='FocalLoss',
                     use_sigmoid=True,
                     gamma=2.0,
                     alpha=0.25,
                     loss_weight=1.0),
                 loss_bbox_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 loss_kpt_init=dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_kpt_refine=dict(
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
        self.num_reppts = num_reppts
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        self.sampling = loss_cls['type'] not in ['FocalLoss']
        self.loss_cls = build_loss(loss_cls)
        self.loss_bbox_init = build_loss(loss_bbox_init)
        self.loss_bbox_refine = build_loss(loss_bbox_refine)
        self.loss_kpt_init = build_loss(loss_kpt_init)
        self.loss_kpt_refine = build_loss(loss_kpt_refine)
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
        # we use deformable conv to extract points features
        self.dcn_kernel = int(np.sqrt(num_reppts))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_reppts, \
            "The points number should be a square number."
        assert self.dcn_kernel % 2 == 1, \
            "The points number should be an odd square number."
        dcn_base = np.arange(-self.dcn_pad,
                             self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape(
            (-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
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
        keypts_out_dim = 2 * self.num_keypts
        reppts_out_dim = 2 * self.num_reppts
        self.cls_refine_dfmconv = DeformConv(self.feat_channels,
                                             self.point_feat_channels,
                                             self.dcn_kernel, 1, self.dcn_pad)
        self.cls_refine_out = nn.Conv2d(self.point_feat_channels,
                                        self.cls_out_channels, 1, 1, 0)
        self.keypts_init_conv = nn.Conv2d(self.feat_channels,
                                          self.point_feat_channels, 3,
                                          1, 1)
        self.keypts_init_out = nn.Conv2d(self.point_feat_channels,
                                         keypts_out_dim, 1, 1, 0)
        # self.reppts_init_conv = nn.Conv2d(self.feat_channels,
        #                                   self.point_feat_channels, 3,
        #                                   1, 1)
        self.reppts_init_out = nn.Conv2d(keypts_out_dim,
                                         reppts_out_dim, 1, 1, 0)
        self.keypts_refine_dfmconv = DeformConv(self.feat_channels,
                                                self.point_feat_channels,
                                                self.dcn_kernel, 1,
                                                self.dcn_pad)
        self.keypts_refine_out = nn.Conv2d(self.point_feat_channels,
                                           keypts_out_dim, 1, 1, 0)
        # self.reppts_refine_dfmconv = DeformConv(self.feat_channels,
        #                                         self.point_feat_channels,
        #                                         self.dcn_kernel, 1,
        #                                         self.dcn_pad)
        self.reppts_refine_out = nn.Conv2d(keypts_out_dim,
                                           reppts_out_dim, 1, 1, 0)

    def init_weights(self):
        for m in self.cls_convs:
            normal_init(m.conv, std=0.01)
        for m in self.reg_convs:
            normal_init(m.conv, std=0.01)
        bias_cls = bias_init_with_prob(0.01)
        normal_init(self.cls_refine_dfmconv, std=0.01)
        normal_init(self.cls_refine_out, std=0.01, bias=bias_cls)
        normal_init(self.keypts_init_conv, std=0.01)
        normal_init(self.keypts_init_out, std=0.01)
        # normal_init(self.reppts_init_conv, std=0.01)
        normal_init(self.reppts_init_out, std=0.01)
        normal_init(self.keypts_refine_dfmconv, std=0.01)
        normal_init(self.keypts_refine_out, std=0.01)
        # normal_init(self.reppts_refine_dfmconv, std=0.01)
        normal_init(self.reppts_refine_out, std=0.01)

    def points2bbox(self, pts, y_first=True):
        """
        Converting the points set into bounding box.
        :param pts: the input points sets (fields), each points
            set (fields) is represented as 2n scalar.
        :param y_first: if y_fisrt=True, the point set is represented as
            [y1, x1, y2, x2 ... yn, xn], otherwise the point set is
            represented as [x1, y1, x2, y2 ... xn, yn].
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
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
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
        else:
            raise NotImplementedError
        return bbox

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

    def gen_grid_from_reg(self, reg, previous_boxes):
        """
        Base on the previous bboxes and regression values, we compute the
            regressed bboxes and generate the grids on the bboxes.
        :param reg: the regression value to previous bboxes.
        :param previous_boxes: previous bboxes.
        :return: generate grids on the regressed bboxes.
        """
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] -
               previous_boxes[:, :2, ...]).clamp(min=1e-6)
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(
            reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        regressed_bbox = torch.cat([
            grid_left, grid_top, grid_left + grid_width, grid_top + grid_height
        ], 1)
        return grid_yx, regressed_bbox

    def forward_single(self, x):
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        # If we use center_init, the initial reppoints is from center points.
        # If we use bounding bbox representation, the initial reppoints is
        #   from regular grid placed on a pre-defined bbox.
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            reppts_init = dcn_base_offset / dcn_base_offset.max() * scale
            # bbox_init = x.new_tensor([-scale, -scale, scale,
            #                           scale]).view(1, 4, 1, 1)
        else:
            reppts_init = 0
            keypts_init = 0
        cls_feat = x
        pts_feat = x
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        # initialize reppoints
        keypts_out_init = self.keypts_init_out(
            self.relu(self.keypts_init_conv(pts_feat)))
        reppts_out_init = self.reppts_init_out(keypts_out_init)
        if self.use_grid_points:
            pass
            # reppts_out_init, bbox_out_init = self.gen_grid_from_reg(
            #     reppts_out_init, bbox_init.detach())
        else:
            reppts_out_init = reppts_out_init + reppts_init
            keypts_out_init = keypts_out_init + keypts_init
        # refine and classify reppoints
        reppts_out_init_grad_mul = self.gradient_mul * reppts_out_init \
            + (1 - self.gradient_mul) * reppts_out_init.detach()
        dcn_offset = reppts_out_init_grad_mul - dcn_base_offset
        cls_out = self.cls_refine_out(
            self.relu(self.cls_refine_dfmconv(cls_feat, dcn_offset)))
        keypts_out_refine = self.keypts_refine_out(
            self.relu(self.keypts_refine_dfmconv(pts_feat, dcn_offset)))
        reppts_out_refine = self.reppts_refine_out(keypts_out_refine)
        if self.use_grid_points:
            pass
            # pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
            #     pts_out_refine, bbox_out_init.detach())
        else:
            keypts_out_refine = keypts_out_refine + keypts_out_init.detach()
            reppts_out_refine = reppts_out_refine + reppts_out_init.detach()
        return (cls_out, keypts_out_init, keypts_out_refine,
                reppts_out_init, reppts_out_refine)

    def forward(self, feats, img_metas):
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

    def centers_to_bboxes(self, point_list):
        """Get bboxes according to center points. Only used in MaxIOUAssigner.
        """
        bbox_list = []
        for i_img, point in enumerate(point_list):
            bbox = []
            for i_lvl in range(len(self.point_strides)):
                scale = self.point_base_scale * self.point_strides[i_lvl] * 0.5
                bbox_shift = torch.Tensor([-scale, -scale, scale,
                                           scale]).view(1, 4).type_as(point[0])
                bbox_center = torch.cat(
                    [point[i_lvl][:, :2], point[i_lvl][:, :2]], dim=1)
                bbox.append(bbox_center + bbox_shift)
            bbox_list.append(bbox)
        return bbox_list

    def offset_to_pts(self, center_list, pred_list):
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
                yx_pts_shift = pts_shift.permute(1, 2, 0).view(
                    -1, 2 * num_points)
                y_pts_shift = yx_pts_shift[..., 0::2]
                x_pts_shift = yx_pts_shift[..., 1::2]
                xy_pts_shift = torch.stack([x_pts_shift, y_pts_shift], -1)
                xy_pts_shift = xy_pts_shift.view(*yx_pts_shift.shape[:-1], -1)
                pts = xy_pts_shift * self.point_strides[i_lvl] + pts_center
                pts_lvl.append(pts)
            pts_lvl = torch.stack(pts_lvl, 0)
            pts_list.append(pts_lvl)
        return pts_list

    def loss_single(self, cls_score,
                    kpt_pred_init, kpt_pred_refine,
                    rep_pred_init, rep_pred_refine,
                    labels, label_weights,
                    bbox_gt_init, bbox_weights_init,
                    bbox_gt_refine, bbox_weights_refine,
                    kpt_gt_init, kpt_weights_init,
                    kpt_gt_refine, kpt_weights_refine, stride,
                    num_total_samples_init, num_total_samples_refine):
        # classification loss
        labels = labels.reshape(-1)
        label_weights = label_weights.reshape(-1)
        cls_score = cls_score.permute(0, 2, 3,
                                      1).reshape(-1, self.cls_out_channels)
        loss_cls = self.loss_cls(
            cls_score,
            labels,
            label_weights,
            avg_factor=num_total_samples_refine)

        # bbox loss
        bbox_gt_init = bbox_gt_init.reshape(-1, 4)
        bbox_weights_init = bbox_weights_init.reshape(-1, 4)
        bbox_pred_init = self.points2bbox(
            rep_pred_init.reshape(-1, 2 * self.num_reppts), y_first=False)
        bbox_gt_refine = bbox_gt_refine.reshape(-1, 4)
        bbox_weights_refine = bbox_weights_refine.reshape(-1, 4)
        bbox_pred_refine = self.points2bbox(
            rep_pred_refine.reshape(-1, 2 * self.num_reppts), y_first=False)

        normalize_term = self.point_base_scale * stride
        loss_bbox_init = self.loss_bbox_init(
            bbox_pred_init / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            avg_factor=num_total_samples_init)
        loss_bbox_refine = self.loss_bbox_refine(
            bbox_pred_refine / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            avg_factor=num_total_samples_refine)

        # keypoint loss

        kpt_weights_init = kpt_weights_init.reshape(-1, self.num_keypts * 2)
        kpt_pred_init = kpt_pred_init.reshape(-1, self.num_keypts * 2)
        kpt_gt_init = kpt_gt_init.reshape(-1, self.num_keypts * 2)
        kpt_pos_num_init = kpt_weights_init.sum(1)
        kpt_weights_init[kpt_pos_num_init > 0] /= kpt_pos_num_init[
            kpt_pos_num_init > 0].unsqueeze(1)

        kpt_weights_refine = kpt_weights_refine.reshape(-1, self.num_keypts * 2)
        kpt_pred_refine = kpt_pred_refine.reshape(-1, self.num_keypts * 2)
        kpt_gt_refine = kpt_gt_refine.reshape(-1, self.num_keypts * 2)
        kpt_pos_num_refine = kpt_weights_refine.sum(1)
        kpt_weights_refine[kpt_pos_num_refine > 0] /= kpt_pos_num_refine[
            kpt_pos_num_refine > 0].unsqueeze(1)

        normalize_term = self.point_base_scale * stride
        loss_kpt_init = self.loss_kpt_init(
            kpt_pred_init / normalize_term,
            kpt_gt_init / normalize_term,
            kpt_weights_init,
            avg_factor=num_total_samples_init)
        loss_kpt_refine = self.loss_kpt_refine(
            kpt_pred_refine / normalize_term,
            kpt_gt_refine / normalize_term,
            kpt_weights_refine,
            avg_factor=num_total_samples_refine)
        return (loss_cls, loss_bbox_init, loss_bbox_refine,
                loss_kpt_init, loss_kpt_refine)

    def loss(self,
             cls_scores,
             keypts_preds_init,
             keypts_preds_refine,
             reppts_preds_init,
             reppts_preds_refine,
             gt_bboxes,
             gt_labels,
             gt_keypoints,
             img_metas,
             cfg,
             gt_bboxes_ignore=None):
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        assert len(featmap_sizes) == len(self.point_generators)
        label_channels = self.cls_out_channels if self.use_sigmoid_cls else 1

        # target for initial stage
        center_list, valid_flag_list = self.get_points(featmap_sizes,
                                                       img_metas)
        keypts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                          keypts_preds_init)
        reppts_coordinate_preds_init = self.offset_to_pts(center_list,
                                                          reppts_preds_init)
        if cfg.init.assigner['type'] == 'PointAssigner':
            # Assign target for center list
            candidate_list = center_list
        else:
            # transform center list to bbox list and
            #   assign target for bbox list
            bbox_list = self.centers_to_bboxes(center_list)
            candidate_list = bbox_list
        cls_reg_targets_init = point_target_kp(
            candidate_list,
            valid_flag_list,
            gt_bboxes,
            gt_keypoints,
            img_metas,
            cfg.init,
            gt_bboxes_ignore_list=gt_bboxes_ignore,
            gt_labels_list=gt_labels,
            label_channels=label_channels,
            sampling=self.sampling)
        (labels_list, label_weights_list,
         bbox_gt_list_init, candidate_list_init, bbox_weights_list_init,
         keypoint_gt_list_init, keypoint_weights_list_init,
         num_total_pos_init, num_total_neg_init) = cls_reg_targets_init
        num_total_samples_init = (
            num_total_pos_init +
            num_total_neg_init if self.sampling else num_total_pos_init)

        # # target for refinement stage
        # center_list, valid_flag_list = self.get_points(featmap_sizes,
        #                                                img_metas)
        # keypts_coordinate_preds_refine = self.offset_to_pts(
        #     center_list, keypts_preds_refine)
        # reppts_coordinate_preds_refine = self.offset_to_pts(
        #     center_list, reppts_preds_refine)
        # bbox_list = []
        # for i_img, center in enumerate(center_list):
        #     bbox = []
        #     for i_lvl in range(len(reppts_preds_refine)):
        #         bbox_preds_init = self.points2bbox(
        #             reppts_preds_init[i_lvl].detach())
        #         bbox_shift = bbox_preds_init * self.point_strides[i_lvl]
        #         bbox_center = torch.cat(
        #             [center[i_lvl][:, :2], center[i_lvl][:, :2]], dim=1)
        #         bbox.append(bbox_center +
        #                     bbox_shift[i_img].permute(1, 2, 0).reshape(-1, 4))
        #     bbox_list.append(bbox)
        # cls_reg_targets_refine = point_target_kp(
        #     bbox_list,
        #     valid_flag_list,
        #     gt_bboxes,
        #     gt_keypoints,
        #     img_metas,
        #     cfg.refine,
        #     gt_bboxes_ignore_list=gt_bboxes_ignore,
        #     gt_labels_list=gt_labels,
        #     label_channels=label_channels,
        #     sampling=self.sampling)
        # (labels_list, label_weights_list, bbox_gt_list_refine,
        #  candidate_list_refine, bbox_weights_list_refine,
        #  keypoint_gt_list_refine, keypoint_weights_list_refine,
        #  num_total_pos_refine, num_total_neg_refine) = cls_reg_targets_refine
        # num_total_samples_refine = (
        #     num_total_pos_refine +
        #     num_total_neg_refine if self.sampling else num_total_pos_refine)

        # compute loss
        (losses_cls, losses_bbox_init, losses_bbox_refine,
         losses_kpt_init, losses_kpt_refine) = multi_apply(
            self.loss_single,
            cls_scores,
            keypts_coordinate_preds_init,
            keypts_coordinate_preds_init,
            reppts_coordinate_preds_init,
            reppts_coordinate_preds_init,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_init,
            bbox_weights_list_init,
            keypoint_gt_list_init,
            keypoint_weights_list_init,
            keypoint_gt_list_init,
            keypoint_weights_list_init,
            self.point_strides,
            num_total_samples_init=num_total_samples_init,
            num_total_samples_refine=num_total_samples_init)
        loss_dict_all = {
            'loss_cls': losses_cls,
            'loss_bbox_init': losses_bbox_init,
            'loss_bbox_refine': losses_bbox_refine,
            'loss_kpt_init': losses_kpt_init,
            'loss_kpt_refine': losses_kpt_refine
        }
        return loss_dict_all

    def get_bboxes(self,
                   cls_scores,
                   keypts_preds_init,
                   keypts_preds_refine,
                   reppts_preds_init,
                   reppts_preds_refine,
                   img_metas,
                   cfg,
                   rescale=False,
                   nms=True):
        assert len(cls_scores) == len(keypts_preds_refine) \
                               == len(reppts_preds_refine)
        bbox_preds_refine = [
            self.points2bbox(reppts_pred_refine)
            for reppts_pred_refine in reppts_preds_refine
        ]
        kpt_preds_refine = [
            self.points2kpt(keypts_pred_refine)
            for keypts_pred_refine in keypts_preds_refine
        ]
        num_levels = len(cls_scores)
        mlvl_points = [
            self.point_generators[i].grid_points(cls_scores[i].size()[-2:],
                                                 self.point_strides[i])
            for i in range(num_levels)
        ]
        result_list = []
        for img_id in range(len(img_metas)):
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]
            bbox_pred_list = [
                bbox_preds_refine[i][img_id].detach()
                for i in range(num_levels)
            ]
            kpt_pred_list = [
                kpt_preds_refine[i][img_id].detach()
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
