import os
import numpy as np
import cv2


def visual_loss(bbox_pred, bbox_gt, bbox_weights, kps_pred, kps_gt, kps_weights):

    vis_dir = '/root/Code/RepPoints/work_dirs/vis'
    num_inst = (bbox_weights.sum(1) > 0).sum()
    if num_inst == 0:
        return

    valid_ind = (bbox_weights.sum(1) > 0)
    num_kp = kps_gt.size(1) // 2

    bbox_pred = bbox_pred[valid_ind]
    bbox_gt = bbox_gt[valid_ind]
    kps_pred = kps_pred[valid_ind]
    kps_gt = kps_gt[valid_ind]
    kps_weights = kps_weights[valid_ind]
    for i in range(num_inst):
        bbox_g = bbox_gt[i].detach().cpu()
        bbox_g = np.round(np.array(bbox_g))
        bbox_g = [int(x) for x in bbox_g]
        bbox_p = bbox_pred[i].detach().cpu()
        bbox_p = np.round(np.array(bbox_p))
        bbox_p = [int(x) for x in bbox_p]
        max_w = max(bbox_g[0::2])
        max_h = max(bbox_g[1::2])
        max_w = max(bbox_p[0::2] + [max_w])
        max_h = max(bbox_p[1::2] + [max_h])

        kps_g = kps_gt[i].detach().cpu()
        kps_g = np.round(np.array(kps_g))
        kps_g = [int(x) for x in kps_g]
        kps_p = kps_pred[i].detach().cpu()
        kps_p = np.round(np.array(kps_p))
        kps_p = [int(x) for x in kps_p]
        max_w = max(kps_g[0::2] + [max_w])
        max_h = max(kps_g[1::2] + [max_h])
        max_w = max(kps_p[0::2] + [max_w])
        max_h = max(kps_p[1::2] + [max_h])

        vis = np.zeros((max_h, max_w, 3), dtype=np.int16)

        x1, y1, x2, y2 = bbox_g
        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
        x1, y1, x2, y2 = bbox_p
        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 0, 255), 2)

        kps_w = kps_weights[i].detach().cpu()
        kps_w = np.array((kps_w > 0).byte())
        for j in range(num_kp):
            s = kps_w[2*j]
            if s > 0:
                x, y = kps_g[2*j+0:2*j+2]
                vis = cv2.circle(vis, (x, y), 2, (0, 255, 0), -1)
                x, y = kps_p[2*j+0:2*j+2]
                vis = cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

    out_file = '{:07d}.jpg'.format(i)
    # out_file = 'vis.jpg'
    out_path = os.path.join(vis_dir, out_file)
    cv2.imwrite(out_path, vis)
    return


def visual_after_nms(bbox_pred, kps_pred):
    import numpy as np
    import cv2
    import os

    vis_dir = '/root/Code/RepPoints/work_dirs/vis'
    # num_inst = bbox_pred.size(0)
    num_kps = kps_pred.size(1) // 3
    scores = bbox_pred[:, 4]
    bbox_pred = bbox_pred[:, :4]
    kps_pred = torch.cat([kps_pred[:, 0::3].unsqueeze(2), 
                          kps_pred[:, 1::3].unsqueeze(2)], dim=2
                         ).view(-1, 2*num_kps)
    sorted_indices = torch.argsort(scores, descending=True)
    ct = 0
    for ind in sorted_indices:
        s = scores[ind]
        if s < 0.9:
            continue
        bbox = bbox_pred[ind].detach().cpu()
        bbox = np.round(np.array(bbox))
        bbox = [int(x) for x in bbox]
        max_w = max(bbox[0::2])
        max_h = max(bbox[1::2])

        kps = kps_pred[ind].detach().cpu()
        kps = np.round(np.array(kps))
        kps = [int(x) for x in kps]
        max_w = max(kps[0::2] + [max_w])
        max_h = max(kps[1::2] + [max_h])

        vis = np.zeros((max_h, max_w, 3), dtype=np.int16)

        x1, y1, x2, y2 = bbox
        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for j in range(num_kps):
            x, y = kps[2*j+0:2*j+2]
            vis = cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        out_file = '{:07d}.jpg'.format(ct)
        # out_file = 'vis.jpg'
        out_path = os.path.join(vis_dir, out_file)
        cv2.imwrite(out_path, vis)
        ct += 1
        import ipdb; ipdb.set_trace()

    return


def visual_before_nms(bbox_pred, score_pred, kps_pred):
    import numpy as np
    import cv2
    import os

    vis_dir = '/root/Code/RepPoints/work_dirs/vis'
    # num_inst = bbox_pred.size(0)
    num_kps = kps_pred.size(1) // 3
    scores = score_pred[:, 1]
    kps_pred = torch.cat([kps_pred[:, 0::3].unsqueeze(2), 
                          kps_pred[:, 1::3].unsqueeze(2)], dim=2
                         ).view(-1, 2*num_kps)
    sorted_indices = torch.argsort(scores, descending=True)
    ct = 0
    for ind in sorted_indices:
        s = scores[ind]
        if s < 0.9:
            continue
        bbox = bbox_pred[ind].detach().cpu()
        bbox = np.round(np.array(bbox))
        bbox = [int(x) for x in bbox]
        max_w = max(bbox[0::2])
        max_h = max(bbox[1::2])

        kps = kps_pred[ind].detach().cpu()
        kps = np.round(np.array(kps))
        kps = [int(x) for x in kps]
        max_w = max(kps[0::2] + [max_w])
        max_h = max(kps[1::2] + [max_h])

        vis = np.zeros((max_h, max_w, 3), dtype=np.int16)

        x1, y1, x2, y2 = bbox
        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for j in range(num_kps):
            x, y = kps[2*j+0:2*j+2]
            vis = cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        out_file = '{:07d}.jpg'.format(ct)
        # out_file = 'vis.jpg'
        out_path = os.path.join(vis_dir, out_file)
        cv2.imwrite(out_path, vis)
        ct += 1
        import ipdb; ipdb.set_trace()

    return


def visual_get_bbox_single(score_pred, bbox_pred, kps_pred):
    import numpy as np
    import cv2
    import os

    vis_dir = '/root/Code/RepPoints/work_dirs/vis'
    # num_inst = bbox_pred.size(0)
    num_kps = kps_pred.size(0) // 2
    scores = score_pred.permute(1, 2, 0).reshape(-1).sigmoid()
    bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4)
    kps_pred = kps_pred.permute(1, 2, 0).reshape(-1, 2*num_kps)

    min_value = min(bbox_pred.min(), kps_pred.min())
    bbox_pred = (bbox_pred - min_value) * 100
    kps_pred = (kps_pred - min_value) * 100

    sorted_indices = torch.argsort(scores, descending=True)
    ct = 0
    for ind in sorted_indices:
        s = scores[ind]
        if s < 0.9:
            continue
        bbox = bbox_pred[ind].detach().cpu()
        bbox = np.round(np.array(bbox))
        bbox = [int(x) for x in bbox]
        max_w = max(bbox[0::2])
        max_h = max(bbox[1::2])

        kps = kps_pred[ind].detach().cpu()
        kps = np.round(np.array(kps))
        kps = [int(x) for x in kps]
        max_w = max(kps[0::2] + [max_w])
        max_h = max(kps[1::2] + [max_h])

        vis = np.zeros((max_h, max_w, 3), dtype=np.int16)

        x1, y1, x2, y2 = bbox
        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for j in range(num_kps):
            x, y = kps[2*j+0:2*j+2]
            vis = cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        out_file = '{:07d}.jpg'.format(ct)
        # out_file = 'vis.jpg'
        out_path = os.path.join(vis_dir, out_file)
        cv2.imwrite(out_path, vis)
        ct += 1
        import ipdb; ipdb.set_trace()

    return


def visual_get_bbox(score_preds, bbox_preds, kps_preds):

    num_lvl = len(score_preds)
    num_image = score_preds[0].size(0)
    for lvl in range(num_lvl):
        for idx in range(num_image):
            score_pred = score_preds[lvl][idx]
            bbox_pred = bbox_preds[lvl][idx]
            kps_pred = kps_preds[lvl][idx]
            visual_get_bbox_single(score_pred, bbox_pred, kps_pred)
    return


def visual_gt_heatmap(img_metas, heatmap_gts):
    import numpy as np
    import cv2
    import os

    vis_dir = '/root/Code/RepPoints/work_dirs/vis'
    num_img = len(img_metas)
    import ipdb; ipdb.set_trace()
    for i_img in range(num_img):
        img_meta = img_metas[i_img]
        heatmap_gt = heatmap_gts[i_img]

        

        img_metas, heatmap_gts
        s = scores[ind]
        if s < 0.9:
            continue
        bbox = bbox_pred[ind].detach().cpu()
        bbox = np.round(np.array(bbox))
        bbox = [int(x) for x in bbox]
        max_w = max(bbox[0::2])
        max_h = max(bbox[1::2])

        kps = kps_pred[ind].detach().cpu()
        kps = np.round(np.array(kps))
        kps = [int(x) for x in kps]
        max_w = max(kps[0::2] + [max_w])
        max_h = max(kps[1::2] + [max_h])

        vis = np.zeros((max_h, max_w, 3), dtype=np.int16)

        x1, y1, x2, y2 = bbox
        vis = cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)

        for j in range(num_kps):
            x, y = kps[2*j+0:2*j+2]
            vis = cv2.circle(vis, (x, y), 2, (0, 0, 255), -1)

        out_file = '{:07d}.jpg'.format(ct)
        # out_file = 'vis.jpg'
        out_path = os.path.join(vis_dir, out_file)
        cv2.imwrite(out_path, vis)
        ct += 1
        import ipdb; ipdb.set_trace()

    return