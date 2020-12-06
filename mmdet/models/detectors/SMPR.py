import torch

from mmdet.core import bbox_mapping_back, kpt_mapping_back, multiclass_nms_kpt, kpt2result

from ..registry import DETECTORS
from .single_stage_kpt import SingleStageDetector_kpt


@DETECTORS.register_module
class SMPR(SingleStageDetector_kpt):

    def __init__(self,
                 backbone,
                 neck,
                 bbox_head,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None):
        super(SMPR, self).__init__(backbone, neck, bbox_head, train_cfg,
                                   test_cfg, pretrained)

    def merge_aug_results(self, aug_bboxes, aug_kpts, aug_scores, aug_rescores, img_metas):
        recovered_bboxes = []
        recovered_kpts = []
        for bboxes, kpts, img_info in zip(aug_bboxes, aug_kpts, img_metas):
            img_shape = img_info[0]['img_shape']
            scale_factor = img_info[0]['scale_factor']
            flip = img_info[0]['flip']
            bboxes = bbox_mapping_back(bboxes, img_shape, scale_factor, flip)
            kpts = kpt_mapping_back(kpts, img_shape, scale_factor, flip)
            recovered_bboxes.append(bboxes)
            recovered_kpts.append(kpts)
        bboxes = torch.cat(recovered_bboxes, dim=0)
        kpts = torch.cat(recovered_kpts, dim=0)
        if aug_scores is None:
            return bboxes, kpts
        else:
            scores = torch.cat(aug_scores, dim=0)
            rescores = torch.cat(aug_rescores, dim=0)

            return bboxes, kpts, scores, rescores
    
    def aug_test(self, imgs, img_metas, rescale=False):
        feats = self.extract_feats(imgs)
        aug_bboxes = []
        aug_kpts = []
        aug_scores = []
        aug_rescores = []
        for x, img_meta in zip(feats, img_metas):
            # only one image in the batch
            outs = self.bbox_head(x)
            bbox_inputs = outs + (img_meta, self.test_cfg, False, False)
            # det_bboxes, det_kpts, det_scores, det_rescores
            det = self.bbox_head.get_bboxes(*bbox_inputs)
            for det_bboxes, det_kpts, det_scores, det_rescores in det:
                aug_bboxes.append(det_bboxes)
                aug_kpts.append(det_kpts)
                aug_scores.append(det_scores)
                aug_rescores.append(det_rescores)
        # merged_bboxes.shape = torch.Size([1720, 4])
        # merged_kpts.shape = torch.Size([1720, 34])
        # merged_scores.shape = torch.Size([1720, 2])
        merged_bboxes, merged_kpts, merged_scores, merged_rescores = self.merge_aug_results(
            aug_bboxes, aug_kpts, aug_scores, aug_rescores, img_metas)
        
        # det_bboxes.shape = torch.Size([10, 5])
        det_bboxes, det_kpts, det_labels = multiclass_nms_kpt(merged_bboxes, merged_kpts, merged_scores,
                                                self.test_cfg.score_thr,
                                                self.test_cfg.nms,
                                                self.test_cfg.max_per_img,
                                                score_factors=merged_rescores)
        # import ipdb; ipdb.set_trace()
        kpt_result = kpt2result(det_bboxes, det_kpts, det_labels, self.bbox_head.num_classes)
        return kpt_result
