import numpy as np
from pycocotools.coco import COCO

from .custom import CustomDataset
from .registry import DATASETS
from collections import defaultdict
from copy import deepcopy


@DATASETS.register_module
class CocoDataset(CustomDataset):

    CLASSES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic_light', 'fire_hydrant',
               'stop_sign', 'parking_meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports_ball', 'kite', 'baseball_bat',
               'baseball_glove', 'skateboard', 'surfboard', 'tennis_racket',
               'bottle', 'wine_glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot_dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted_plant', 'bed', 'dining_table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell_phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy_bear', 'hair_drier', 'toothbrush')

    def load_annotations(self, ann_file):
        num_joints = 17
        self.gt_class_keypoints_dict = {1: (0, 16)}
        self.flip_pairs = [[[1, 2], [3, 4], [5, 6], [7, 8],
                           [9, 10], [11, 12], [13, 14], [15, 16]]]
        self.flip_indices = np.arange(num_joints)
        for cat_list in self.flip_pairs:
            for pair in cat_list:
                k1, k2 = pair
                self.flip_indices[k1] = k2
                self.flip_indices[k2] = k1
        flip_indices_0 = deepcopy(self.flip_indices * 2).reshape(-1, 1)
        flip_indices_1 = deepcopy(self.flip_indices * 2 + 1).reshape(-1, 1)
        self.flip_indices = np.concatenate([flip_indices_0, flip_indices_1], 1).reshape(-1)

        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.getCatIds()
        self.cat2label = {
            cat_id: i + 1
            for i, cat_id in enumerate(self.cat_ids)
        }
        self.img_ids = self.coco.getImgIds()
        img_infos = []
        for i in self.img_ids:
            info = self.coco.loadImgs([i])[0]
            info['filename'] = info['file_name']
            img_infos.append(info)
        return img_infos

    def get_ann_info(self, idx):
        img_id = self.img_infos[idx]['id']
        ann_ids = self.coco.getAnnIds(imgIds=[img_id])
        ann_info = self.coco.loadAnns(ann_ids)
        return self._parse_ann_info(ann_info, self.with_mask,
                                    self.with_keypoint)

    def _filter_imgs(self, min_size=32, min_keypoint=0):
        """Filter annotations without enough keypoints, then filter
        images too small or without ground truths.
        """
        def visible_kps_in_ann(ann):
            # Each keypoints field has the format [x1, y1, v1, ...], where v is visibility
            return (np.array(ann["keypoints"][2::3]) > 0).sum()

        num_ann = len(self.coco.anns)
        anns = {}
        imgToAnns = defaultdict(list)
        for k, v in self.coco.anns.items():
            # import ipdb; ipdb.set_trace()
            if visible_kps_in_ann(v) >= min_keypoint:
                anns[v['id']] = v
                imgToAnns[v['image_id']].append(v)
        self.coco.anns = anns
        self.coco.imgToAnns = imgToAnns

        num_reserved_ann = len(self.coco.anns)
        num_filterd_ann = num_ann - num_reserved_ann
        print(f'Loaded {num_ann} instances, filtered {num_filterd_ann} '
              + f'instances with fewer than {min_keypoint} keypoints.')

        valid_inds = []
        ids_with_ann = set(_['image_id'] for _ in self.coco.anns.values())
        for i, img_info in enumerate(self.img_infos):
            if self.img_ids[i] not in ids_with_ann:
                continue
            if min(img_info['width'], img_info['height']) >= min_size:
                valid_inds.append(i)
        print(f'Totally {len(valid_inds)} images left.')
        return valid_inds

    def _parse_ann_info(self, ann_info, with_mask=True, with_keypoint=True):
        """Parse bbox and mask annotation.

        Args:
            ann_info (list[dict]): Annotation info of an image.
            with_mask (bool): Whether to parse mask annotations.

        Returns:
            dict: A dict containing the following keys: bboxes, bboxes_ignore,
                labels, masks, mask_polys, poly_lens.
        """
        gt_bboxes = []
        gt_labels = []
        gt_bboxes_ignore = []
        # Two formats are provided.
        # 1. mask: a binary map of the same size of the image.
        # 2. polys: each mask consists of one or several polys, each poly is a
        # list of float.
        if with_mask:
            gt_masks = []
            gt_mask_polys = []
            gt_poly_lens = []
        if with_keypoint:
            gt_keypoints = []
        for i, ann in enumerate(ann_info):
            if ann.get('ignore', False):
                continue
            x1, y1, w, h = ann['bbox']
            if ann['area'] <= 0 or w < 1 or h < 1:
                continue
            bbox = [x1, y1, x1 + w - 1, y1 + h - 1]
            if ann['iscrowd']:
                gt_bboxes_ignore.append(bbox)
            else:
                gt_bboxes.append(bbox)
                gt_labels.append(self.cat2label[ann['category_id']])
            if with_mask:
                gt_masks.append(self.coco.annToMask(ann))
                mask_polys = [
                    p for p in ann['segmentation'] if len(p) >= 6
                ]  # valid polygons have >= 3 points (6 coordinates)
                poly_lens = [len(p) for p in mask_polys]
                gt_mask_polys.append(mask_polys)
                gt_poly_lens.extend(poly_lens)
            if with_keypoint:
                keypoints = np.reshape(ann['keypoints'], (-1, 3))
                gt_keypoints.append(keypoints)

        if gt_bboxes:
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            gt_labels = np.array(gt_labels, dtype=np.int64)
        else:
            gt_bboxes = np.zeros((0, 4), dtype=np.float32)
            gt_labels = np.array([], dtype=np.int64)

        if gt_bboxes_ignore:
            gt_bboxes_ignore = np.array(gt_bboxes_ignore, dtype=np.float32)
        else:
            gt_bboxes_ignore = np.zeros((0, 4), dtype=np.float32)

        ann = dict(
            bboxes=gt_bboxes, labels=gt_labels, bboxes_ignore=gt_bboxes_ignore)

        if with_mask:
            ann['masks'] = gt_masks
            # poly format is not used in the current implementation
            ann['mask_polys'] = gt_mask_polys
            ann['poly_lens'] = gt_poly_lens
        if with_keypoint:
            ann['keypoints'] = np.array(gt_keypoints, dtype=np.float32)
        return ann
