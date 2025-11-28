import math

import torch
import torch.nn as nn
import torchvision
from loss import BboxRegression, ClsScoreRegression
from torchvision import models
from utils import assign_label, compute_offsets, generate_proposal


class FeatureExtractor(nn.Module):
    """
    Image feature extraction with MobileNet.
    """

    def __init__(self, reshape_size=224, pooling=False, verbose=False):
        super().__init__()

        self.mobilenet = models.mobilenet_v2(pretrained=True)
        self.mobilenet = nn.Sequential(
            *list(self.mobilenet.children())[:-1]
        )  # Remove the last classifier

        # average pooling
        if pooling:
            self.mobilenet.add_module(
                "LastAvgPool", nn.AvgPool2d(math.ceil(reshape_size / 32.0))
            )  # input: N x 1280 x 7 x 7

        for i in self.mobilenet.named_parameters():
            i[1].requires_grad = True  # fine-tune all

    def forward(self, img, verbose=False):
        """
        Inputs:
        - img: Batch of resized images, of shape Nx3x224x224

        Outputs:
        - feat: Image feature, of shape Nx1280 (pooled) or Nx1280x7x7
        """
        num_img = img.shape[0]

        img_prepro = img

        feat = []
        process_batch = 500
        for b in range(math.ceil(num_img / process_batch)):
            feat.append(
                self.mobilenet(img_prepro[b * process_batch : (b + 1) * process_batch])
                .squeeze(-1)
                .squeeze(-1)
            )  # forward and squeeze
        feat = torch.cat(feat)

        if verbose:
            print("Output feature shape: ", feat.shape)

        return feat


class FastRCNN(nn.Module):
    def __init__(
        self,
        in_dim=1280,
        hidden_dim=256,
        num_classes=20,
        roi_output_w=2,
        roi_output_h=2,
        drop_ratio=0.3,
    ):
        super().__init__()

        assert num_classes != 0
        self.num_classes = num_classes
        self.roi_output_w, self.roi_output_h = roi_output_w, roi_output_h
        self.feat_extractor = FeatureExtractor()
        ##############################################################################
        # TODO: Declare the cls & bbox heads (in Fast R-CNN).                        #
        # The cls & bbox heads share a sequential module with a Linear layer,        #
        # followed by a Dropout (p=drop_ratio), a ReLU nonlinearity and another      #
        # Linear layer.                                                              #
        # The cls head is a Linear layer that predicts num_classes + 1 (background). #
        # The bbox head is a Linear layer that predicts offsets(dim=4).              #
        # HINT: The dimension of the two Linear layers are in_dim -> hidden_dim and  #
        # hidden_dim -> hidden_dim.                                                  #
        ##############################################################################
        # Replace "pass" statement with your code

        # 共享的Sequential模块
        self.shared_layers = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.Dropout(p=drop_ratio),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
        )

        # 分类头：预测num_classes + 1个类别（包含背景）
        self.cls_head = nn.Linear(hidden_dim, self.num_classes + 1)

        # 边界框回归头：预测4个边界框偏移量
        self.bbox_head = nn.Linear(hidden_dim, 4)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################

    def forward(self, images, bboxes, bbox_batch_ids, proposals, proposal_batch_ids):
        """
        Training-time forward pass for our two-stage Faster R-CNN detector.

        Inputs:
        - images: Tensor of shape (B, 3, H, W) giving input images
        - bboxes: Tensor of shape (N, 5) giving ground-truth bounding boxes
        and category labels, from the dataloader, where N is the total number
        of GT boxes in the batch
        - bbox_batch_ids: Tensor of shape (N, ) giving the index (in the batch)
        of the image that each GT box belongs to
        - proposals: Tensor of shape (M, 4) giving the proposals for input images,
        where M is the total number of proposals in the batch
        - proposal_batch_ids: Tensor of shape (M, ) giving the index of the image
        that each proposals belongs to

        Outputs:
        - total_loss: Torch scalar giving the overall training loss.
        """
        w_cls = 1  # for cls_scores
        w_bbox = 1  # for offsets
        total_loss = None
        ##############################################################################
        # TODO: Implement the forward pass of Fast R-CNN.                            #
        # A few key steps are outlined as follows:                                   #
        # i) Extract image feature.                                                  #
        # ii) Perform RoI Pool on proposals and meanpool the feature in the spatial  #
        #     dimension.                                                             #
        # iii) Pass the RoI feature through the shared-fc layer. Predict             #
        #      classification scores and box offsets                                 #
        # iv) Assign the proposals with targets of each image.                       #
        #     HINT: Use `assign_label` in utils.py.                                  #
        #     NOTE: foreground ids are in [0, self.num_classes-1], and background id #
        #           is self.num_classes. (see dataset.py)                            #
        # v) Compute the cls_loss between the predicted class_prob and GT_class      #
        #    (For poistive & negative proposals)                                     #
        #    Compute the bbox_loss between the offsets and GT_offsets                #
        #    (For positive proposals)                                                #
        #    Compute the total_loss which is formulated as:                          #
        #    total_loss = w_cls*cls_loss + w_bbox*bbox_loss.                         #
        ##############################################################################
        # Replace "pass" statement with your code
        B, _, H, W = images.shape

        # extract image feature
        feature_map = self.feat_extractor(images)

        # perform RoI Pool & mean pool
        rois_list = []
        for i, (proposal, batch_id) in enumerate(zip(proposals, proposal_batch_ids)):
            # 将每个proposal与对应的batch index组合，格式: [batch_index, x1, y1, x2, y2]
            roi = torch.cat([batch_id.unsqueeze(0).float(), proposal])
            rois_list.append(roi)

        if rois_list:
            rois = torch.stack(rois_list)  # shape: (M, 5)
        else:
            return torch.tensor(0.0).to(images.device)

        # 特征图大小 / 输入图像大小 = 7 / 224 ≈ 0.03125
        spatial_scale = feature_map.shape[2] / images.shape[2]

        pooled_features = torchvision.ops.roi_pool(
            feature_map,
            rois,
            output_size=(self.roi_output_h, self.roi_output_w),
            spatial_scale=1.0,
        )  # shape: (M, 1280, 2, 2)

        spatial_pooled = torch.nn.functional.adaptive_avg_pool2d(
            pooled_features, (1, 1)
        )  # (M, 1280, 1, 1)
        roi_features_flat = spatial_pooled.view(spatial_pooled.size(0), -1)  # (M, 1280)

        # forward heads, get predicted cls scores & offsets
        shared_features = self.shared_layers(roi_features_flat)
        cls_scores = self.cls_head(shared_features)
        bbox_offsets = self.bbox_head(shared_features)

        # assign targets with proposals
        pos_masks, neg_masks, GT_labels, GT_bboxes = [], [], [], []
        for img_idx in range(B):
            # get the positive/negative proposals and corresponding
            # GT box & class label of this image
            # 获取当前图像的proposals
            img_proposal_mask = proposal_batch_ids == img_idx
            img_proposals = proposals[img_proposal_mask]

            # 获取当前图像的GT boxes
            img_bbox_mask = bbox_batch_ids == img_idx
            img_bboxes = bboxes[img_bbox_mask]

            if len(img_proposals) > 0 and len(img_bboxes) > 0:
                # 使用assign_label函数分配标签
                pos_mask, neg_mask, assigned_labels, assigned_bboxes = assign_label(
                    proposals=img_proposals,
                    bboxes=img_bboxes,
                    background_id=self.num_classes,
                    pos_thresh=0.5,
                    neg_thresh=0.5,
                    pos_fraction=0.25,
                )
                # 将结果添加到列表中
                pos_masks.append(pos_mask)
                neg_masks.append(neg_mask)
                GT_labels.append(assigned_labels)
                GT_bboxes.append(assigned_bboxes)
            else:
                # 如果没有proposals或GT boxes，创建空的mask和标签
                false_mask = torch.zeros(
                    len(img_proposals), dtype=torch.bool, device=images.device
                )
                true_mask = torch.ones(
                    len(img_proposals), dtype=torch.bool, device=images.device
                )
                empty_labels = torch.full(
                    (len(img_proposals),),
                    self.num_classes,
                    dtype=torch.long,
                    device=images.device,
                )
                empty_bboxes = torch.zeros(
                    (len(img_proposals), 4), device=images.device
                )

                pos_masks.append(false_mask)
                neg_masks.append(true_mask)
                GT_labels.append(empty_labels)
                #GT_bboxes.append(empty_bboxes)

        pos_mask = torch.cat(pos_masks)
        neg_mask = torch.cat(neg_masks)
        GT_labels = torch.cat(GT_labels)
        GT_bboxes = torch.cat(GT_bboxes)

        # compute loss
        cls_mask = pos_mask | neg_mask  # 正样本和负样本都要计算分类损失

        if cls_mask.sum() > 0:
            # 使用ClsScoreRegression函数计算分类损失
            # 只对分配了标签的proposal计算分类损失
            cls_loss = ClsScoreRegression(
                cls_scores[cls_mask],  # 预测的类别分数
                GT_labels[cls_mask],  # GT类别标签
                B
            )
        else:
            # 如果没有正负样本，分类损失为0
            cls_loss = torch.tensor(0.0, device=images.device)

        if pos_mask.sum() > 0:
            # 获取正样本对应的proposals和GT框
            pos_proposals = proposals[pos_mask]  # 正样本的候选框
            pos_GT_bboxes = GT_bboxes   # 正样本对应的真实框（来自assign_label）

            # 计算真实的偏移量
            true_offsets = compute_offsets(pos_proposals, pos_GT_bboxes)

            # 使用BboxRegression函数计算边界框回归损失
            # 注意：bbox_offsets包含所有proposal的预测，我们只取正样本的
            bbox_loss = BboxRegression(
                bbox_offsets[pos_mask],  # 正样本的预测偏移量
                true_offsets,  # 计算得到的真实偏移量
                B
            )
        else:
            # 如果没有正样本，边界框回归损失为0
            bbox_loss = torch.tensor(0.0, device=images.device)

        # Compute the total_loss which is formulated as:
        # total_loss = w_cls*cls_loss + w_bbox*bbox_loss
        total_loss = w_cls * cls_loss + w_bbox * bbox_loss

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return total_loss

    def inference(
        self, images, proposals, proposal_batch_ids, thresh=0.5, nms_thresh=0.7
    ):
        """ "
        Inference-time forward pass for our two-stage Faster R-CNN detector

        Inputs:
        - images: Tensor of shape (B, 3, H, W) giving input images
        - proposals: Tensor of shape (M, 4) giving the proposals for input images,
        where M is the total number of proposals in the batch
        - proposal_batch_ids: Tensor of shape (M, ) giving the index of the image
        that each proposals belongs to
        - thresh: Threshold value on confidence probability. HINT: You can convert the
        classification score to probability using a softmax nonlinearity.
        - nms_thresh: IoU threshold for NMS

        We can output a variable number of predicted boxes per input image.
        In particular we assume that the input images[i] gives rise to P_i final
        predicted boxes.

        Outputs:
        - final_proposals: List of length (B,) where final_proposals[i] is a Tensor
        of shape (P_i, 4) giving the coordinates of the final predicted boxes for
        the input images[i]
        - final_conf_probs: List of length (B,) where final_conf_probs[i] is a
        Tensor of shape (P_i, 1) giving the predicted probabilites that the boxes
        in final_proposals[i] are objects (vs background)
        - final_class: List of length (B,), where final_class[i] is an int64 Tensor
        of shape (P_i, 1) giving the predicted category labels for each box in
        final_proposals[i].
        """
        final_proposals, final_conf_probs, final_class = None, None, None
        ##############################################################################
        # TODO: Predicting the final proposal coordinates `final_proposals`,         #
        # confidence scores `final_conf_probs`, and the class index `final_class`.   #
        # The overall steps are similar to the forward pass, but now you cannot      #
        # decide the activated nor negative proposals without GT boxes.              #
        # You should apply post-processing (thresholding and NMS) to all proposals.  #
        # A few key steps are outlined as follows:                                   #
        # i) Extract image feature.                                                  #
        # ii) Perform RoI Pooling on proposals, then apply mean pooling on the RoI   #
        #     features in the spatial dimensions.                                    #
        #     HINT: Use torchvision.ops.roi_pool.                                    #
        # iii) Pass the RoI features through the shared-fc layer. Predict            #
        #      classification scores and box offsets.                                #
        # iv) Obtain the predicted class, confidence score, and bounding box for     #
        #     each proposal. Note that the confidence score is the probability of    #
        #     the predicted class (apply softmax to classification scores first).    #
        #     HINT: Use generate_proposal from utils.py to get predicted boxes from  #
        #     offsets.                                                               #
        # v) Apply post-processing to filter and obtain the final predictions for    #
        #    each image in the batch:                                                #
        #    - Thresholding: Filter out predictions with confidence lower than       #
        #      threshold.                                                            #
        #    - Non-Maximum Suppression (NMS): Apply NMS to the remaining predictions #
        #      for each image (do not distinguish between different classes).        #
        #      HINT: Use torchvision.ops.nms.                                        #
        ##############################################################################
        # Replace "pass" statement with your code
        B = images.shape[0]

        # extract image feature
        feature_map = self.feat_extractor(images)

        # perform RoI Pool & mean pool
        rois_list = []
        for i, (proposal, batch_id) in enumerate(zip(proposals, proposal_batch_ids)):
            # 将每个proposal与对应的batch index组合，格式: [batch_index, x1, y1, x2, y2]
            roi = torch.cat([batch_id.unsqueeze(0).float(), proposal])
            rois_list.append(roi)

        if rois_list:
            rois = torch.stack(rois_list)  # shape: (M, 5)
        else:
            # 返回每个图像的空结果
            final_proposals = [
                torch.empty((0, 4), device=images.device) for _ in range(B)
            ]
            final_conf_probs = [
                torch.empty((0, 1), device=images.device) for _ in range(B)
            ]
            final_class = [
                torch.empty((0, 1), dtype=torch.int64, device=images.device)
                for _ in range(B)
            ]
            return final_proposals, final_conf_probs, final_class

        # 特征图大小 / 输入图像大小 = 7 / 224 ≈ 0.03125
        spatial_scale = feature_map.shape[2] / images.shape[2]

        pooled_features = torchvision.ops.roi_pool(
            feature_map,
            rois,
            output_size=(self.roi_output_h, self.roi_output_w),
            spatial_scale=1.0,
        )  # shape: (M, 1280, 2, 2)

        spatial_pooled = torch.nn.functional.adaptive_avg_pool2d(
            pooled_features, (1, 1)
        )  # (M, 1280, 1, 1)
        roi_features_flat = spatial_pooled.view(spatial_pooled.size(0), -1)  # (M, 1280)

        # forward heads, get predicted cls scores & offsets
        shared_features = self.shared_layers(roi_features_flat)
        cls_scores = self.cls_head(shared_features)
        bbox_offsets = self.bbox_head(shared_features)

        # get predicted boxes & class label & confidence probability
        predicted_boxes = generate_proposal(proposals, bbox_offsets)
        probabilities = nn.functional.softmax(cls_scores, dim=-1)
        confidences_probability, predicted_classes = torch.max(probabilities[:,:-1], dim=-1)

        final_proposals = []
        final_conf_probs = []
        final_class = []
        # post-process to get final predictions
        for img_idx in range(B):
            # filter by threshold
            # 获取属于当前图像的proposal的mask
            img_mask = proposal_batch_ids == img_idx

            if not img_mask.any():
                # 如果没有属于该图像的proposal，添加空tensor
                final_proposals.append(torch.empty((0, 4), device=images.device))
                final_conf_probs.append(torch.empty((0, 1), device=images.device))
                final_class.append(
                    torch.empty((0, 1), dtype=torch.int64, device=images.device)
                )
                continue

            # 提取当前图像的预测结果
            img_boxes = predicted_boxes[img_mask]
            img_confidences = confidences_probability[img_mask]
            img_classes = predicted_classes[img_mask]
            # Thresholding: 过滤掉置信度低于阈值或预测为背景的预测
            keep_mask = img_confidences >= thresh
            # remove background predictions (background id == self.num_classes)
            #class_mask = img_classes != self.num_classes
            #keep_mask = keep_mask & class_mask

            if not keep_mask.any():
                # 如果没有满足阈值条件的预测，添加空tensor
                final_proposals.append(torch.empty((0, 4), device=images.device))
                final_conf_probs.append(torch.empty((0, 1), device=images.device))
                final_class.append(
                    torch.empty((0, 1), dtype=torch.int64, device=images.device)
                )
                continue

            img_boxes = img_boxes[keep_mask]
            img_confidences = img_confidences[keep_mask]
            img_classes = img_classes[keep_mask]

            # nms
            keep_indices = torchvision.ops.nms(img_boxes, img_confidences, nms_thresh)

            # 获取最终预测结果
            img_final_boxes = img_boxes[keep_indices]
            img_final_confidences = img_confidences[keep_indices]
            img_final_classes = img_classes[keep_indices]

            # 添加到最终结果列表
            final_proposals.append(img_final_boxes)
            final_conf_probs.append(img_final_confidences.unsqueeze(1))
            final_class.append(img_final_classes.unsqueeze(1).to(torch.int64))
            print(final_proposals)

        ##############################################################################
        #                               END OF YOUR CODE                             #
        ##############################################################################
        return final_proposals, final_conf_probs, final_class
