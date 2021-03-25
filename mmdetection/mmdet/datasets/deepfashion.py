from .coco import CocoDataset
from .registry import DATASETS
import numpy as np
from copy import deepcopy
from pycocotools.coco import COCO

@DATASETS.register_module
class DeepFashionDataset(CocoDataset):
    CLASSES = (
        'Upper', 'Lower', 'Whole')

    def load_annotations(self, ann_file):
        self.gt_class_keypoints_dict = {
            1: [], 2: [], 3: []}
        self.flip_pairs = [
            [[0, 1], [2, 3], [6, 7]],
            [[4, 5], [6, 7]],
            [[0, 1], [2, 3], [4, 5], [6, 7]]
        ]

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

    @staticmethod
    def generate_target(joints, joints_vis, heatmap_size, g, sigma=1, target_type='grid'):
        '''
        :param joints:  [num_joints, 3]
        :param joints_vis: [num_joints, 3]
        :param target_type: grid, heatmap
        :return: target, target_weight(1: visible, 0: invisible)
        '''
        num_joints = joints.shape[0]
        target_weight = np.zeros((num_joints + 1, 1), dtype=np.float32)
        target_weight[1:, 0] = joints_vis[:, 0]

        target = np.zeros((num_joints + 1,
                           heatmap_size,
                           heatmap_size),
                          dtype=np.float32)

        tmp_size = sigma * 2

        joint_idx = np.argwhere(target_weight == 1)[:, 0].tolist()
        #for joint_id in range(num_joints):
        for joint_id in joint_idx:
            mu_x = int(joints[joint_id - 1][0] + 0.5)
            mu_y = int(joints[joint_id - 1][1] + 0.5)
            if mu_x < 0 or mu_y < 0 or mu_x >= heatmap_size or mu_y >= heatmap_size:
                target_weight[joint_id] = 0
                continue

            if target_type == 'heatmap':
                # Check that any part of the gaussian is in-bounds
                ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
                br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
                if ul[0] >= heatmap_size or ul[1] >= heatmap_size \
                        or br[0] < 0 or br[1] < 0:
                    # If not, just return the image as is
                    target_weight[joint_id] = 0
                    continue

                v = target_weight[joint_id]
                if v > 0.5:
                    # Usable gaussian range
                    g_x = max(0, -ul[0]), min(br[0], heatmap_size) - ul[0]
                    g_y = max(0, -ul[1]), min(br[1], heatmap_size) - ul[1]
                    # Image range
                    img_x = max(0, ul[0]), min(br[0], heatmap_size)
                    img_y = max(0, ul[1]), min(br[1], heatmap_size)

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            elif target_type == 'grid':
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][mu_y, mu_x] = 1
        return target, target_weight
