import torch

from mmdet.core import bbox2result, bbox_mapping_back, multiclass_nms
from ..registry import DETECTORS
from .single_stage import SingleStageDetector
import numpy as np


@DETECTORS.register_module
class RepPointsDetectorKp(SingleStageDetector):
    """RepPoints: Point Set Representation for Object Detection.

        This detector is the implementation of:
        - RepPoints detector (https://arxiv.org/pdf/1904.11490)
    """

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super().__init__(backbone, neck, bbox_head, train_cfg, test_cfg,
                         pretrained)

    @property
    def with_keypoint(self):
        return True

    def forward_dummy(self, img):
        """Used for computing network flops.

        See `mmdetection/tools/get_flops.py`
        """
        x = self.extract_feat(img)
        outs = self.bbox_head(x, None)
        return outs

    def forward_train(self,
                      img,
                      img_metas,
                      gt_bboxes,
                      gt_labels,
                      gt_keypoints,
                      gt_bboxes_ignore=None):
        x = self.extract_feat(img)
        # outs = self.bbox_head(x)
        outs = self.bbox_head(x, img_metas)
        loss_inputs = outs + (gt_bboxes, gt_labels, gt_keypoints, img_metas, self.train_cfg)
        losses = self.bbox_head.loss(
            *loss_inputs, gt_bboxes_ignore=gt_bboxes_ignore)
        return losses

    def bbox2result_kp(self, bboxes, labels, kpts, num_classes):
        """Convert detection results to a list of numpy arrays.

        Args:
            bboxes (Tensor): shape (n, 5)
            labels (Tensor): shape (n, )
            num_classes (int): class number, including background class

        Returns:
            list(ndarray): bbox results of each class
        """
        if bboxes.shape[0] == 0:
            return ([
                np.zeros((0, 5), dtype=np.float32) for i in range(
                                                    num_classes - 1)
            ], )
        else:
            bboxes = bboxes.cpu().numpy()
            labels = labels.cpu().numpy()
            kpts = kpts.cpu().numpy()
            bboxes_in_cls = [bboxes[labels == i, :] for i in range(num_classes - 1)]
            bbox_scores = bboxes[:, 4]
            kpt_in_cls = [kpts[labels == i, :] for i in range(num_classes - 1)]
            return (bboxes_in_cls, bbox_scores, kpt_in_cls, )

    def simple_test(self, img, img_meta, rescale=False):
        x = self.extract_feat(img)
        # outs = self.bbox_head(x)
        outs = self.bbox_head(x, img_meta)
        bbox_inputs = outs + (img_meta, self.test_cfg, rescale)
        bbox_list = self.bbox_head.get_bboxes(*bbox_inputs)
        bbox_results = [
            self.bbox2result_kp(det_bboxes, det_labels, det_kpts,
                                self.bbox_head.num_classes)
            for det_bboxes, det_labels, det_kpts in bbox_list
        ]
        return bbox_results[0]

    def merge_aug_results(self, aug_bboxes, aug_scores, img_metas):
        """Merge augmented detection bboxes and scores.

        Args:
            aug_bboxes (list[Tensor]): shape (n, 4*#class)
            aug_scores (list[Tensor] or None): shape (n, #class)
            img_shapes (list[Tensor]): shape (3, ).

        Returns:
            tuple: (bboxes, scores)
        """
        recovered_bboxes = []
        for bboxes, img_info in zip(aug_bboxes, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        if aug_scores is None:
            return bboxes
        else:
            scores = torch.cat(aug_scores, dim=0)
            return bboxes, scores

    def aug_test(self, imgs, img_metas, rescale=False):
        # recompute feats to save memory
        feats = self.extract_feats(imgs)

        aug_bboxes = []
        aug_scores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            # TODO more flexible
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            det_bboxes, det_scores = self.bbox_head.get_bboxes(*bbox_inputs)[0]
            aug_bboxes.append(det_bboxes)
            aug_scores.append(det_scores)

        # after merging, bboxes will be rescaled to the original image size
        merged_bboxes, merged_scores = self.merge_aug_results(
            aug_bboxes, aug_scores, img_metas)
        det_bboxes, det_labels = multiclass_nms(merged_bboxes, merged_scores,
                                                self.test_cfg.score_thr,
                                                self.test_cfg.nms,
                                                self.test_cfg.max_per_img)

        if rescale:
            _det_bboxes = det_bboxes
        else:
            _det_bboxes = det_bboxes.clone()
            _det_bboxes[:, :4] *= img_metas[0][0]['scale_factor']
        bbox_results = bbox2result(_det_bboxes, det_labels,
                                   self.bbox_head.num_classes)
        return bbox_results
