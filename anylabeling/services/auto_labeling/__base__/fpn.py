import cv2
import numpy as np
import os
import logging
import traceback
from PyQt5 import QtCore
from PyQt5.QtCore import QCoreApplication
from anylabeling.views.labeling.shape import Shape
from anylabeling.views.labeling.utils.opencv import qt_img_to_rgb_cv_img
from ..model import Model
from anylabeling.app_info import __preferred_device__
from ..engines.build_onnx_engine import OnnxBaseModel
from ..types import AutoLabelingResult
from scipy.ndimage import zoom


class FPN(Model):
    class Meta:
        required_config_names = [
            "type",
            "name",
            "display_name",
            "model_path",
            "classes",
        ]
        widgets = ["button_run"]
        output_modes = {
            "polygon": QCoreApplication.translate("Model", "Polygon"),
        }
        default_output_mode = "polygon"

    def __init__(self, model_config, on_message) -> None:
        # Run the parent class's init method
        super().__init__(model_config, on_message)
        model_name = self.config['type']
        self.input_size = self.config["input_size"]
        self.pixel_mean = np.array([123.675, 116.28, 103.53])
        self.pixel_std = np.array([58.395, 57.12, 57.375])
        model_abs_path = self.get_model_abs_path(self.config, "model_path")
        if not model_abs_path or not os.path.isfile(model_abs_path):
            raise FileNotFoundError(
                QCoreApplication.translate(
                    "Model",
                    f"Could not download or initialize {model_name} model."
                )
            )
        self.net = OnnxBaseModel(model_abs_path, __preferred_device__)
        self.classes = self.config["classes"]
        self.input_shape = self.net.get_input_shape()[-2:]

        self.filter_classes = self.config.get("filter_classes", None)

        if self.filter_classes:
            self.filter_classes = [
                i for i, item in enumerate(self.classes)
                if item in self.filter_classes
            ]

    def post_process(self, masks):
        """
        Post process masks
        """
        # Find contours
        masks = masks[0]
        masks = masks.argmax(axis=0)
        shapes = []
        for i in range(len(self.classes)):
            masks_cur = np.zeros(masks.shape)
            masks_cur[masks == (i+1)] = 255
            # masks[masks == i] = 255
            masks_cur = masks_cur.astype(np.uint8)
            contours, _ = cv2.findContours(
                masks_cur, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
            )
            # Refine contours
            approx_contours = []
            for contour in contours:
                # Approximate contour
                epsilon = 0.001 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                approx_contours.append(approx)
            filtered_approx_contours = []

            # Remove too big contours ( >90% of image size)
            if len(approx_contours) >= 1:
                image_size = masks.shape[0] * masks.shape[1]
                areas = [cv2.contourArea(contour) for contour in approx_contours]
                filtered_approx_contours = [
                    contour
                    for contour, area in zip(approx_contours, areas)
                    if area < image_size * 0.9
                ]
            # Remove small contours (area < 20% of average area)
            if len(filtered_approx_contours) >= 1:
                areas = [cv2.contourArea(contour) for contour in approx_contours]
                avg_area = np.mean(areas)

                filtered_approx_contours = [
                    contour
                    for contour, area in zip(filtered_approx_contours, areas)
                    if area > avg_area * 0.05
                ]

            approx_contours = filtered_approx_contours
            if self.output_mode == "polygon":
                for approx in approx_contours:
                    # Scale points
                    points = approx.reshape(-1, 2)
                    points[:, 0] = points[:, 0]
                    points[:, 1] = points[:, 1]
                    points = points.tolist()
                    if len(points) < 3:
                        continue
                    points.append(points[0])

                    # Create shape
                    shape = Shape(flags={})
                    for point in points:
                        point[0] = int(point[0])
                        point[1] = int(point[1])
                        shape.add_point(QtCore.QPointF(point[0], point[1]))
                    shape.shape_type = "polygon"
                    shape.closed = True

                    shape.fill_color = '#' + f"{i+1}".zfill(6)
                    shape.line_color = '#' + f"{i+1}".zfill(6)
                    shape.line_width = 2
                    shape.label = self.classes[i]
                    shape.selected = False
                    shapes.append(shape)



        return shapes

    def transform_masks(self, masks, original_size, transform_matrix):
        """Transform masks
        Transform the masks back to the original image size.
        """
        output_masks = []
        for batch in range(masks.shape[0]):
            batch_masks = []
            for mask_id in range(masks.shape[1]):
                mask = masks[batch, mask_id]
                mask = cv2.warpAffine(
                    mask,
                    transform_matrix[:2],
                    (original_size[1], original_size[0]),
                    flags=cv2.INTER_LINEAR,
                )
                batch_masks.append(mask)
            output_masks.append(batch_masks)
        return np.array(output_masks)

    def preprocess(self, input_image):
        """
        Calculate embedding and metadata for a single image.
        """
        # Normalization
        # input_image = input_image.astype(np.float32)
        input_image = (input_image - self.pixel_mean) / self.pixel_std
        original_size = input_image.shape[:2]

        # Calculate a transformation matrix to convert to self.input_size
        scale_x = self.input_size[1] / input_image.shape[1]
        scale_y = self.input_size[0] / input_image.shape[0]
        scale = min(scale_x, scale_y)
        transform_matrix = np.array(
            [
                [scale, 0, 0],
                [0, scale, 0],
                [0, 0, 1],
            ]
        )
        cv_image = cv2.warpAffine(
            input_image,
            transform_matrix[:2],
            (self.input_size[1], self.input_size[0]),
            flags=cv2.INTER_LINEAR,
        )
        cv_image = cv_image.astype(np.float32)
        cv_image = np.expand_dims(cv_image, axis=0)
        cv_image = cv_image.transpose(0, 3, 1, 2)
        return {
            "image_resized": cv_image,
            "original_size": original_size,
            "transform_matrix": transform_matrix,
        }
        # image = letterbox(input_image, self.input_shape, stride=self.stride)[0]
        # image = image.transpose((2, 0, 1)) # HWC to CHW
        # image = np.ascontiguousarray(image).astype('float32')
        # image /= 255  # 0 - 255 to 0.0 - 1.0
        # if len(image.shape) == 3:
        #     image = image[None]
        # return image

    def predict_shapes(self, image, filename=None):
        """
        Predict shapes from image
        """
        if image is None:
            return []

        try:
            image = qt_img_to_rgb_cv_img(image, filename)
        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            return []
        shapes = []
        image_metas = self.preprocess(image)
        blob = image_metas['image_resized']
        original_size = image_metas['original_size']
        trans_matrix = image_metas['transform_matrix']
        try:
            predictions = self.net.get_ort_inference(blob)

            if len(predictions.shape) == 4:
                predictions = predictions[0]
            else:
                predictions = predictions[0]
                # Transform the masks back to the original image size.
            # post process the predict result
            predictions = zoom(predictions, (1, 4, 4), mode='nearest')
            # predictions = predictions.argmax(axis=0)
            # predictions = np.expand_dims(predictions, axis=0)
            predictions = np.expand_dims(predictions, axis=0)
            inv_transform_matrix = np.linalg.inv(trans_matrix)
            transformed_masks = self.transform_masks(
                predictions, original_size, inv_transform_matrix
            )

            shapes = self.post_process(transformed_masks)

        except Exception as e:  # noqa
            logging.warning("Could not inference model")
            logging.warning(e)
            traceback.print_exc()
            return AutoLabelingResult([], replace=False)

        result = AutoLabelingResult(shapes, replace=False)
        return result
    def unload(self):
        del self.net