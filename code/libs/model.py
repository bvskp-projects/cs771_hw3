import math
import torch
import torchvision

from torchvision.models import resnet
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.ops.boxes import batched_nms

import torch
from torch import nn

# point generator
from .point_generator import PointGenerator

# input / output transforms
from .transforms import GeneralizedRCNNTransform

# loss functions
from .losses import sigmoid_focal_loss, giou_loss


class FCOSClassificationHead(nn.Module):
    """
    A classification head for FCOS with convolutions and group norms

    Args:
        in_channels (int): number of channels of the input feature.
        num_classes (int): number of classes to be predicted
        num_convs (Optional[int]): number of conv layer. Default: 3.
        prior_probability (Optional[float]): probability of prior. Default: 0.01.
    """

    def __init__(self, in_channels, num_classes, num_convs=3, prior_probability=0.01):
        super().__init__()
        self.num_classes = num_classes

        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        # A separate background category is not needed, as later we will consider
        # C binary classfication problems here (using sigmoid focal loss)
        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1, padding=1
        )
        torch.nn.init.normal_(self.cls_logits.weight, std=0.01)
        # see Sec 3.3 in "Focal Loss for Dense Object Detection'
        torch.nn.init.constant_(
            self.cls_logits.bias, -math.log((1 - prior_probability) / prior_probability)
        )

    def forward(self, x):
        """
        Fill in the missing code here. The head will be applied to all levels
        of the feature pyramid, and predict a single logit for each location on
        every feature location.

        Without pertumation, the results will be a list of tensors in increasing
        depth order, i.e., output[0] will be the feature map with highest resolution
        and output[-1] will the feature map with lowest resolution. The list length is
        equal to the number of pyramid levels. Each tensor in the list will be
        of size N x C x H x W, storing the classification logits (scores).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        cls_logits_list = list() # list to contain the classification logits for each FPN layer

        for features in x:
            cls_logits = self.conv(features)
            cls_logits = self.cls_logits(cls_logits)

            # Change output shape from (N, num_classes, H, W) to (N, H*W, num_classes)
            N, K, H, W = cls_logits.shape
            cls_logits = cls_logits.view(N, self.num_classes, H, W) # N, num_classes, H, W
            cls_logits = cls_logits.permute(0, 2, 3, 1) # N, H, W, num_classes
            cls_logits = cls_logits.reshape(N, -1, self.num_classes) # N, HW, num_classes

            cls_logits_list.append(cls_logits)

        return cls_logits_list # list of length num_fpn_heads with (N, HW, num_classes) logits for each level


class FCOSRegressionHead(nn.Module):
    """
    A regression head for FCOS with convolutions and group norms.
    This head predicts
    (a) the distances from each location (assuming foreground) to a box
    (b) a center-ness score

    Args:
        in_channels (int): number of channels of the input feature.
        num_convs (Optional[int]): number of conv layer. Default: 3.
    """

    def __init__(self, in_channels, num_convs=3):
        super().__init__()
        conv = []
        for _ in range(num_convs):
            conv.append(
                nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1)
            )
            conv.append(nn.GroupNorm(16, in_channels))
            conv.append(nn.ReLU(inplace=True))
        self.conv = nn.Sequential(*conv)

        # regression outputs must be positive
        self.bbox_reg = nn.Sequential(
            nn.Conv2d(in_channels, 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True)
        )
        self.bbox_ctrness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1, padding=1
        )

        self.apply(self.init_weights)
        # The following line makes sure the regression head output a non-zero value.
        # If your regression loss remains the same, try to uncomment this line.
        # It helps the initial stage of training
        # torch.nn.init.normal_(self.bbox_reg[0].bias, mean=1.0, std=0.1)

    def init_weights(self, m):
        if isinstance(m, nn.Conv2d):
            torch.nn.init.normal_(m.weight, std=0.01)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        """
        Fill in the missing code here. The logic is rather similar to
        FCOSClassificationHead. The key difference is that this head bundles both
        regression outputs and the center-ness scores.

        Without pertumation, the results will be two lists of tensors in increasing
        depth order, corresponding to regression outputs and center-ness scores.
        Again, the list length is equal to the number of pyramid levels.
        Each tensor in the list will be of size N x 4 x H x W (regression)
        or N x 1 x H x W (center-ness).

        Some re-arrangement of the outputs is often preferred for training / inference.
        You can choose to do it here, or in compute_loss / inference.
        """
        bbox_regression_list = list()
        bbox_centerness_list = list()

        for features in x:
            bbox_feature = self.conv(features)
            bbox_regression = self.bbox_reg(bbox_feature) # ReLU is already applied here!
            bbox_centerness = self.bbox_ctrness(bbox_feature)

            # Change bbox output shape from (N, 4, H, W) to (N, H*W, 4)
            N, _, H, W = bbox_regression.shape
            bbox_regression = bbox_regression.view(N, 4, H, W)
            bbox_regression = bbox_regression.permute(0, 2, 3, 1) # N, H, W, 4
            bbox_regression = bbox_regression.reshape(N, -1, 4) # N, HW, 4 
            bbox_regression_list.append(bbox_regression)

            # Change centerness output shape from (N, 1, H, W) to (N, H*W, 1)
            N, _, H, W = bbox_centerness.shape
            bbox_centerness = bbox_centerness.view(N, 1, H, W)
            bbox_centerness = bbox_centerness.permute(0, 2, 3, 1) # N, H, W, 1
            bbox_centerness = bbox_centerness.reshape(N, -1, 1) # N, HW, 1 
            bbox_centerness_list.append(bbox_centerness)
        
        # Concatenate lists of bbox coords and centerness vals
        return bbox_regression_list, bbox_centerness_list # lists of len num_fpn_heads, (N, HW, 4 OR 1) for each fpn level

class FCOS(nn.Module):
    """
    Implementation of Fully Convolutional One-Stage (FCOS) object detector,
    as desribed in the journal paper: https://arxiv.org/abs/2006.09214

    Args:
        backbone (string): backbone network, only ResNet is supported now
        backbone_freeze_bn (bool): if to freeze batch norm in the backbone
        backbone_out_feats (List[string]): output feature maps from the backbone network
        backbone_out_feats_dims (List[int]): backbone output features dimensions
        (in increasing depth order)

        fpn_feats_dim (int): output feature dimension from FPN in increasing depth order
        fpn_strides (List[int]): feature stride for each pyramid level in FPN
        num_classes (int): number of output classes of the model (excluding the background)
        regression_range (List[Tuple[int, int]]): box regression range on each level of the pyramid
        in increasing depth order. E.g., [[0, 32], [32 64]] means that the first level
        of FPN (highest feature resolution) will predict boxes with width and height in range of [0, 32],
        and the second level in the range of [32, 64].

        img_min_size (List[int]): minimum sizes of the image to be rescaled before feeding it to the backbone
        img_max_size (int): maximum size of the image to be rescaled before feeding it to the backbone
        img_mean (Tuple[float, float, float]): mean values used for input normalization.
        img_std (Tuple[float, float, float]): std values used for input normalization.

        train_cfg (Dict): dictionary that specifies training configs, including
            center_sampling_radius (int): radius of the "center" of a groundtruth box,
            within which all anchor points are labeled positive.

        test_cfg (Dict): dictionary that specifies test configs, including
            score_thresh (float): Score threshold used for postprocessing the detections.
            nms_thresh (float): NMS threshold used for postprocessing the detections.
            detections_per_img (int): Number of best detections to keep after NMS.
            topk_candidates (int): Number of best detections to keep before NMS.

        * If a new parameter is added in config.py or yaml file, they will need to be defined here.
    """

    def __init__(
        self,
        backbone,
        backbone_freeze_bn,
        backbone_out_feats,
        backbone_out_feats_dims,
        fpn_feats_dim,
        fpn_strides,
        num_classes,
        regression_range,
        img_min_size,
        img_max_size,
        img_mean,
        img_std,
        train_cfg,
        test_cfg,
    ):
        super().__init__()
        assert backbone in ("resnet18", "resnet34", "resnet50", "resnet101", "resnet152")
        self.backbone_name = backbone
        self.backbone_freeze_bn = backbone_freeze_bn
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.regression_range = regression_range

        return_nodes = {}
        for feat in backbone_out_feats:
            return_nodes.update({feat: feat})

        # backbone network
        backbone_model = resnet.__dict__[backbone](weights="IMAGENET1K_V1")
        self.backbone = create_feature_extractor(
            backbone_model, return_nodes=return_nodes
        )

        # feature pyramid network (FPN)
        self.fpn = FeaturePyramidNetwork(
            backbone_out_feats_dims,
            out_channels=fpn_feats_dim,
            extra_blocks=LastLevelP6P7(fpn_feats_dim, fpn_feats_dim)
        )

        # point generator will create a set of points on the 2D image plane
        self.point_generator = PointGenerator(
            img_max_size, fpn_strides, regression_range
        )

        # classification and regression head
        self.cls_head = FCOSClassificationHead(fpn_feats_dim, num_classes)
        self.reg_head = FCOSRegressionHead(fpn_feats_dim)

        # image batching, normalization, resizing, and postprocessing
        self.transform = GeneralizedRCNNTransform(
            img_min_size, img_max_size, img_mean, img_std
        )

        # other params for training / inference
        self.center_sampling_radius = train_cfg["center_sampling_radius"]
        self.score_thresh = test_cfg["score_thresh"]
        self.nms_thresh = test_cfg["nms_thresh"]
        self.detections_per_img = test_cfg["detections_per_img"]
        self.topk_candidates = test_cfg["topk_candidates"]

    """
    We will overwrite the train function. This allows us to always freeze
    all batchnorm layers in the backbone, as we won't have sufficient samples in
    each mini-batch to aggregate the bachnorm stats.
    """
    @staticmethod
    def freeze_bn(module):
        if isinstance(module, nn.BatchNorm2d):
            module.eval()

    def train(self, mode=True):
        self.training = mode
        for module in self.children():
            module.train(mode)
        # additionally fix all bn ops (affine params are still allowed to update)
        if self.backbone_freeze_bn:
            self.apply(self.freeze_bn)
        return self

    """
    The behavior of the forward function changes depending on if the model is
    in training or evaluation mode.

    During training, the model expects both the input images
    (list of tensors within the range of [0, 1]),
    as well as a targets (list of dictionary), containing the following keys
        - boxes (``FloatTensor[N, 4]``): the ground-truth boxes in
          ``[x1, y1, x2, y2]`` format, with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the class label for each ground-truth box
        - other keys such as image_id are not used here
    The model returns a Dict[Tensor] during training, containing the classification, regression
    and centerness losses, as well as a final loss as a summation of all three terms.

    During inference, the model requires only the input tensors, and returns the post-processed
    predictions as a List[Dict[Tensor]], one for each input image. The fields of the Dict are as
    follows:
        - boxes (``FloatTensor[N, 4]``): the predicted boxes in ``[x1, y1, x2, y2]`` format,
          with ``0 <= x1 < x2 <= W`` and ``0 <= y1 < y2 <= H``.
        - labels (Int64Tensor[N]): the predicted labels for each image
        - scores (Tensor[N]): the scores for each prediction

    See also the comments for compute_loss / inference.
    """

    def forward(self, images, targets):
        # sanity check
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    torch._assert(
                        isinstance(boxes, torch.Tensor),
                        "Expected target boxes to be of type Tensor.",
                    )
                    torch._assert(
                        len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                        f"Expected target boxes of shape [N, 4], got {boxes.shape}.",
                    )

        # record the original image size, this is needed to decode the box outputs
        original_image_sizes = []
        for img in images:
            val = img.shape[-2:]
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.transform(images, targets)

        # get the features from the backbone
        # the result will be a dictionary {feature name : tensor}
        features = self.backbone(images.tensors)

        # send the features from the backbone into the FPN
        # the result is converted into a list of tensors (list length = #FPN levels)
        # this list stores features in increasing depth order, each of size N x C x H x W
        # (N: batch size, C: feature channel, H, W: height and width)
        fpn_features = self.fpn(features)
        fpn_features = list(fpn_features.values())

        # classification / regression heads
        cls_logits = self.cls_head(fpn_features)
        reg_outputs, ctr_logits = self.reg_head(fpn_features)

        # 2D points (corresponding to feature locations) of shape H x W x 2
        points, strides, reg_range = self.point_generator(fpn_features)

        # training / inference
        if self.training:
            # training: generate GT labels, and compute the loss
            losses = self.compute_loss(
                targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
            )
            # return loss during training
            return losses

        else:
            # inference: decode / postprocess the boxes
            detections = self.inference(
                points, strides, cls_logits, reg_outputs, ctr_logits, images.image_sizes
            )
            # rescale the boxes to the input image resolution
            detections = self.transform.postprocess(
                detections, images.image_sizes, original_image_sizes
            )
            # return detectrion results during inference
            return detections

    """
    Fill in the missing code here. This is probably the most tricky part
    in this assignment. Here you will need to compute the object label for each point
    within the feature pyramid. If a point lies around the center of a foreground object
    (as controlled by self.center_sampling_radius), its regression and center-ness
    targets will also need to be computed.

    Further, three loss terms will be attached to compare the model outputs to the
    desired targets (that you have computed), including
    (1) classification (using sigmoid focal for all points)
    (2) regression loss (using GIoU and only on foreground points)
    (3) center-ness loss (using binary cross entropy and only on foreground points)

    Some of the implementation details that might not be obvious
    * The output regression targets are divided by the feature stride (Eq 1 in the paper)
    * All losses are normalized by the number of positive points (Eq 2 in the paper)

    The output must be a dictionary including the loss values
    {
        "cls_loss": Tensor (1)
        "reg_loss": Tensor (1)
        "ctr_loss": Tensor (1)
        "final_loss": Tensor (1)
    }
    where the final_loss is a sum of the three losses and will be used for training.
    """

    def compute_loss(
        self, targets, points, strides, reg_range, cls_logits, reg_outputs, ctr_logits
    ):
        return losses

    """
    Fill in the missing code here. The inference is also a bit involved. It is
    much easier to think about the inference on a single image
    (a) Loop over every pyramid level
        (1) compute the object scores
        (2) filter out boxes with object scores (self.score_thresh)
        (3) select the top K boxes (self.topk_candidates)
        (4) decode the boxes
        (5) clip boxes outside of the image boundaries (due to padding) / remove small boxes
    (b) Collect all candidate boxes across all pyramid levels
    (c) Run non-maximum suppression to remove any duplicated boxes
    (d) keep the top K boxes after NMS (self.detections_per_img)

    Some of the implementation details that might not be obvious
    * As the output regression target is divided by the feature stride during training,
    you will have to multiply the regression outputs by the stride at inference time.
    * Most of the detectors will allow two overlapping boxes from different categories
    (e.g., one from "shirt", the other from "person"). That means that
        (a) one can decode two same boxes of different categories from one location;
        (b) NMS is only performed within each category.
    * Regression range is not used, as the range is not enforced during inference.
    * image_shapes is needed to remove boxes outside of the images.
    * Output labels should be offseted by +1 to compensate for the input label transform

    The output must be a list of dictionary items (one for each image) following
    [
        {
            "boxes": Tensor (N x 4) with each row in (x1, y1, x2, y2)
            "scores": Tensor (N, )
            "labels": Tensor (N, )
        },
    ]
    """



    def inference(
        self, points, strides, cls_logits, reg_outputs, ctr_logits, image_shapes
    ):
        # points: list of possible points at each fpn layer: [(80, 80, 2), (40, 40, 2), (20, 20, 2), (10, 10, 2), (5, 5, 2)]
        # strides: tensor of the different FPN strides [8, 16, 32, 64, 128]
        # cls_logits: list of tensors: (N, HW_, num_classes) for each FPN layer: [[N, 6400, 20], [N, 1600, 20], [N, 400, 20], [N, 100, 20], [N, 25, 20]] = (N, 8525, num_classes)
        # reg_outputs: list of tensors: (N, HW_, 4) for each FPN layer: [[N, 6400, 4], [N, 1600, 4], [N, 400, 4], [N, 100, 4], [N, 25, 4]] = (N, 8525, 4)
        # ctr_logits: list of tensors: (N, HW_, 1) for each FPN layer: [[N, 6400, 1], [N, 1600, 1], [N, 400, 1], [N, 100, 1], [N, 25, 1]] = (N, 8525, 1)
        # image_shapes: shapes of each of the input images [(640, 451), ... * N]

        detections = list() # list of detections for each image

        num_images = len(image_shapes)


        for i in range(num_images):
            # Get a list of all the bboxes/logits/center scores at each fpn level for ONE image
            bbox_coords_per_image = [fpn_level_bboxes[i] for fpn_level_bboxes in reg_outputs]
            cls_logits_per_image = [fpn_level_cls_logits[i] for fpn_level_cls_logits in cls_logits]
            center_logits_per_image = [fpn_level_centerness[i] for fpn_level_centerness in ctr_logits]
            image_shape = image_shapes[i]

            # Lists that contain the predictions, scores, and bboxes for every detection in this image at the different fpn levels
            final_image_bboxes_list = list()
            final_image_confidence_scores_list = list()
            final_image_classes_list = list()

            for bbox_coords_in_fpn_level, cls_logits_in_fpn_level, center_logits_in_fpn_level, points_in_fpn_level, stride_in_fpn_level in zip(bbox_coords_per_image, cls_logits_per_image, center_logits_per_image, points, strides):
                num_classes = cls_logits_in_fpn_level.shape[-1] # get the number of classes

                # remove lowest scoring boxes, keep only topk scoring predictions
                final_image_confidence_scores = torch.sqrt(torch.sigmoid(cls_logits_in_fpn_level) * torch.sigmoid(center_logits_in_fpn_level))
                final_image_confidence_scores = final_image_confidence_scores.flatten() # flatten to 1D tensor to easily do thresholding

                indexesToKeep = final_image_confidence_scores > self.score_thresh
                final_image_confidence_scores = final_image_confidence_scores[indexesToKeep]

                # keep only topk scoring predictions
                topKIndexes = torch.where(indexesToKeep)[0] # indexes of every True value
                # compute number of topk predictions to get, either there are enough high scoring candidates to get the max number (self.topk_candidates), or keep all the ones we have
                num_topK = min(self.topk_candidates, topKIndexes.size(0)) 
                final_image_confidence_scores, indexes = final_image_confidence_scores.topk(num_topK) # get the final topk scores and indexes for that image at that fpn level
                topKIndexes = topKIndexes[indexes]
                
                # Get the final point indexes; de-flatten the indexes by dividing by the number of classes (undoes the class logits * center logits flatten operation above)
                point_indexes = torch.div(topKIndexes, num_classes, rounding_mode="floor")

                # Get the final labels: remainder of topkindexes will get you the class value (due to the flatten operation above)
                classes_per_fpn_level = topKIndexes % num_classes

                # Get the final boxes
                box_offsets = bbox_coords_in_fpn_level[point_indexes]
                box_points = points_in_fpn_level.reshape((-1, 2))[point_indexes] # get the associated centers for the topk scores: format (# point locations in this fpn level, 2)

                center_x = box_points[...,1]
                center_y = box_points[...,0]

                l = torch.mul(box_offsets[...,0], stride_in_fpn_level) # need to multiply by correct stride!!
                t = torch.mul(box_offsets[...,1], stride_in_fpn_level)
                r = torch.mul(box_offsets[...,2], stride_in_fpn_level)
                b = torch.mul(box_offsets[...,3], stride_in_fpn_level)

                final_image_bboxes = torch.stack((center_x - l, center_y - t, center_x + r, center_y + b), dim=-1) # (# point locations in fpn level, 4)

                # clip final bboxes to be within the image dimensions
                box_dims = final_image_bboxes.dim()
                boxes_x = final_image_bboxes[...,0::2]
                boxes_y = final_image_bboxes[...,1::2]
                final_image_bboxes = torch.stack((boxes_x.clamp(min=0, max=image_shape[1]), boxes_y.clamp(min=0, max=image_shape[0])), dim=box_dims).reshape(final_image_bboxes.shape)

                # append computed bbox coords, labels, and scores to lists
                final_image_bboxes_list.append(final_image_bboxes)
                final_image_confidence_scores_list.append(final_image_confidence_scores)
                final_image_classes_list.append(classes_per_fpn_level)


            # Create final output tensors
            final_bboxes = torch.cat(final_image_bboxes_list, dim=0)
            final_scores = torch.cat(final_image_confidence_scores_list, dim=0)
            final_classes = torch.cat(final_image_classes_list, dim=0)

            # Do non-max suppression SEPARATELY FOR EACH CLASS
            toKeep = batched_nms(final_bboxes, final_scores, final_classes, self.nms_thresh)

            toKeep = toKeep[: self.detections_per_img] # keep the first __ number of detections
            detections.append({"boxes": final_bboxes[toKeep], "scores": final_scores[toKeep], "labels": final_classes[toKeep] + 1})

        return detections
    
