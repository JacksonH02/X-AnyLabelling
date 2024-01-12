import numpy as np

from .__base__.yolo_seg import YOLO_Seg


class YOLOv8_Seg(YOLO_Seg):

    def get_box_results(self, bboxes):

        predictions = np.squeeze(bboxes).T
        num_classes = bboxes.shape[1] - self.num_masks - 4
        scores = np.max(predictions[:, 4: 4+num_classes], axis=1)
        predictions = predictions[scores > self.conf_threshold, :]
        scores = scores[scores > self.conf_threshold]

        if len(scores) == 0:
            return [], [], [], np.array([])

        box_predictions = predictions[..., :num_classes+4]
        mask_predictions = predictions[..., num_classes+4:]

        # Get the class with the highest confidence
        class_ids = np.argmax(box_predictions[:, 4:], axis=1)

        # Get bounding boxes for each object
        boxes = self.extract_boxes(box_predictions)

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        indices = self.numpy_nms(boxes, scores, self.iou_threshold)

        return boxes[indices], scores[indices], class_ids[indices], mask_predictions[indices]