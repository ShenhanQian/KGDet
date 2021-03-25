from .coco import CocoDataset
from .registry import DATASETS
import numpy as np
from copy import deepcopy
from pycocotools.coco import COCO

@DATASETS.register_module
class DeepFashion2Dataset(CocoDataset):
    CLASSES = (
        'short_sleeved_shirt', 'long_sleeved_shirt',
        'short_sleeved_outwear', 'long_sleeved_outwear',
        'vest', 'sling', 'shorts', 'trousers', 'skirt',
        'short_sleeved_dress', 'long_sleeved_dress',
        'vest_dress', 'sling_dress')

    def load_annotations(self, ann_file):
        num_joints = 294
        self.gt_class_keypoints_dict = {
            1: (0, 25), 2: (25, 58), 3: (58, 89), 4: (89, 128), 5: (128, 143),
            6: (143, 158), 7: (158, 168), 8: (168, 182), 9: (182, 190),
            10: (190, 219), 11: (219, 256), 12: (256, 275), 13: (275, 294)}
        self.keypoint_groups = [
            [1, 26, 59, 90, 129, 144, 191, 220, 257, 276],
            [2, 27, 62, 91, 130, 192, 221, 258],
            [3, 28, 61, 92, 131, 193, 222, 259],
            [4, 29, 132, 147, 194, 223, 260, 279],
            [5, 30, 63, 94, 133, 195, 224, 261],
            [6, 31, 64, 95, 134, 196, 225, 262], [7, 32, 65, 96, 197, 226],
            [8, 66, 198], [9, 67, 199], [10, 68, 200], [11, 69, 201],
            [12, 41, 70, 105, 202, 235],
            [13, 42, 71, 106, 136, 151, 203, 236, 264, 283],
            [14, 43, 72, 107, 137, 152, 204, 237, 265, 284],
            [15, 44, 73, 108, 138, 153, 205, 238, 266, 285],
            [16, 45, 139, 154],
            [17, 46, 75, 110, 140, 155, 211, 244, 272, 291],
            [18, 47, 76, 111, 141, 156, 212, 245, 273, 292],
            [19, 48, 77, 112, 142, 157, 213, 246, 274, 293],
            [20, 49, 78, 113, 214, 247],
            [21, 79, 215], [22, 80, 216], [23, 81, 217], [24, 82, 218],
            [25, 58, 83, 122, 219, 256], [33, 97, 227], [34, 98, 228],
            [35, 99, 229], [36, 100, 230], [37, 101, 231], [38, 102, 232],
            [39, 103, 233], [40, 104, 234], [50, 114, 248], [51, 115, 249],
            [52, 116, 250], [53, 117, 251], [54, 118, 252], [55, 119, 253],
            [56, 120, 254], [57, 121, 255], [60, 93], [74, 109], [84, 123],
            [85, 124], [86, 125], [87, 126], [88, 127], [89, 128], [135, 263],
            [143, 275], [145, 277], [146, 278], [148, 280], [149, 281],
            [150, 282], [158, 294], [159, 169, 183], [160, 170, 184],
            [161, 171, 185], [162, 186], [163], [164], [165, 177],
            [166], [167], [168, 190], [172], [173], [174], [175], [176],
            [178], [179], [180], [181], [182], [187], [188], [189],
            [206, 239, 267, 286], [207, 240, 268, 287], [208, 241, 269, 288],
            [209, 242, 270, 289], [210, 243, 271, 290]]

        _flip_pairs = [
            [[2, 6], [3, 5], [7, 25], [8, 24], [9, 23], [10, 22], [11, 21],
                [12, 20], [13, 19], [14, 18], [15, 17]],
            [[2, 6], [3, 5], [7, 33], [8, 32], [9, 31], [10, 30], [11, 29],
                [12, 28], [13, 27], [14, 26], [15, 25], [16, 24], [17, 23],
                [18, 22], [19, 21]],
            [[2, 26], [3, 5], [4, 6], [7, 25], [8, 24], [9, 23], [10, 22],
                [11, 21], [12, 20], [13, 19], [14, 18], [15, 17], [16, 29],
                [27, 30], [28, 31]],
            [[2, 6], [3, 5], [4, 34], [7, 33], [8, 32], [9, 31], [10, 30],
                [11, 29], [12, 28], [13, 27], [14, 26], [15, 25], [16, 24],
                [17, 23], [18, 22], [19, 21], [20, 37], [35, 38], [36, 39]],
            [[2, 6], [3, 5], [7, 15], [8, 14], [9, 13], [10, 12]],
            [[2, 6], [3, 5], [7, 15], [8, 14], [9, 13], [10, 12]],
            [[1, 3], [4, 10], [5, 9], [6, 8]],
            [[1, 3], [4, 14], [5, 13], [6, 12], [7, 11], [8, 10]],
            [[1, 3], [4, 8], [5, 7]],
            [[2, 6], [3, 5], [7, 29], [8, 28], [9, 27], [10, 26], [11, 25],
                [12, 24], [13, 23], [14, 22], [15, 21], [16, 20], [17, 19]],
            [[2, 6], [3, 5], [7, 37], [8, 36], [9, 35], [10, 34], [11, 33],
                [12, 32], [13, 31], [14, 30], [15, 29], [16, 28], [17, 27],
                [18, 26], [19, 25], [20, 24], [21, 23]],
            [[2, 6], [3, 5], [7, 19], [8, 18], [9, 17], [10, 16], [11, 15],
                [12, 14]],
            [[2, 6], [3, 5], [7, 19], [8, 18], [9, 17], [10, 16], [11, 15],
                [12, 14]]
        ]
        self.flip_pairs = []
        for idx, _cat_pairs in enumerate(_flip_pairs):
            start_idx = self.gt_class_keypoints_dict[idx+1][0]
            cat_pairs = []
            for pair in _cat_pairs:
                x0 = pair[0] + start_idx - 1
                x1 = pair[1] + start_idx - 1
                cat_pairs.append([x0, x1])
            self.flip_pairs.append(cat_pairs)

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
                    #pdb.set_trace()

                    target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
                        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]

            elif target_type == 'grid':
                v = target_weight[joint_id]
                if v > 0.5:
                    target[joint_id][mu_y, mu_x] = 1
        return target, target_weight