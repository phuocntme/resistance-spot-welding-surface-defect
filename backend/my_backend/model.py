"""YOLO + SAM ML backend for Label Studio polygon annotation."""

from __future__ import annotations

import gc
import json
import logging
import os
import shutil
from typing import Any

import cv2
import numpy as np
import torch
from numpy.typing import NDArray
from segment_anything import SamPredictor, sam_model_registry
from ultralytics import YOLO

from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.utils import get_image_local_path

from config import get_settings

__all__ = ["YoloSamBackend"]

logger = logging.getLogger(__name__)

DEVICE: str = "cuda" if torch.cuda.is_available() else "cpu"


class YoloSamBackend(LabelStudioMLBase):
    """Label Studio ML backend using YOLO for detection and SAM for segmentation."""

    def __init__(self, **kwargs: Any) -> None:
        if "model_dir" not in kwargs or kwargs["model_dir"] is None:
            kwargs["model_dir"] = os.path.dirname(os.path.abspath(__file__))

        super().__init__(**kwargs)

        settings = get_settings()

        # Label config mapping
        self.image_key: str = settings.image_key
        self.output_from_name: str = settings.output_from_name
        self.output_to_name: str = settings.output_to_name

        logger.info("Device: %s | Loading models...", DEVICE)

        try:
            self.yolo = YOLO(settings.yolo_checkpoint).to(DEVICE)

            sam = sam_model_registry[settings.sam_model_type](
                checkpoint=settings.sam_checkpoint
            )
            sam.to(device=DEVICE)
            self.sam_predictor = SamPredictor(sam)

            logger.info("YOLO and SAM models loaded successfully")
        except Exception:
            logger.exception("Failed to load models")
            raise

    def predict(
        self,
        tasks: list[dict[str, Any]],
        context: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Main prediction method called by Label Studio."""
        predictions: list[dict[str, Any]] = []

        if not context and "context" in kwargs:
            context = kwargs["context"]
        if context is None:
            context = {}

        interactive_items: list[dict[str, Any]] = []
        if isinstance(context.get("result"), list):
            interactive_items = context["result"]

        is_interactive = len(interactive_items) > 0

        logger.debug(
            "predict called | tasks=%d | interactive_items=%d",
            len(tasks),
            len(interactive_items),
        )

        for task in tasks:
            task_results: list[dict[str, Any]] = []
            image: NDArray[np.uint8] | None = None

            try:
                image_ref = task.get("data", {}).get(self.image_key)
                if not image_ref:
                    predictions.append({"result": task_results})
                    continue

                image_path = get_image_local_path(image_ref)
                image = cv2.imread(image_path)
                if image is None:
                    predictions.append({"result": task_results})
                    continue

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                h, w, _ = image.shape

                self.sam_predictor.set_image(image)

                if is_interactive:
                    task_results = self._process_interactive(interactive_items, h, w)
                else:
                    task_results = self._process_auto_detect(image, h, w)

                predictions.append({"result": task_results})

            finally:
                del image
                self._cleanup_memory()

        return predictions

    def _process_interactive(
        self,
        interactive_items: list[dict[str, Any]],
        h: int,
        w: int,
    ) -> list[dict[str, Any]]:
        """Handle interactive annotation (keypoint clicks and rectangle boxes)."""
        results: list[dict[str, Any]] = []

        for res in interactive_items:
            region_type = str(res.get("type", "")).lower()
            val = res.get("value", {})

            logger.debug("Interactive item: type=%s", region_type)

            if region_type in ("keypointlabels", "keypoint"):
                result = self._handle_keypoint(val, h, w)
                if result:
                    results.append(result)

            elif region_type in ("rectanglelabels", "rectangle"):
                result = self._handle_rectangle(val, h, w, region_id=res.get("id"))
                if result:
                    results.append(result)

            else:
                logger.debug("Unsupported region type: %s", region_type)

        logger.debug("Interactive polygons generated: %d", len(results))
        return results

    def _handle_keypoint(
        self,
        val: dict[str, Any],
        h: int,
        w: int,
    ) -> dict[str, Any] | None:
        """Process a keypoint click into a SAM point prompt."""
        labels = val.get("keypointlabels", []) or val.get("labels", [])
        if not labels:
            return None

        px = val["x"] * w / 100
        py = val["y"] * h / 100

        logger.debug("SAM point prompt: (%.1f, %.1f) | Label: %s", px, py, labels)

        point_coords = np.array([[px, py]], dtype=np.float32)
        point_labels = np.array([1])

        masks, _, _ = self.sam_predictor.predict(
            point_coords=point_coords,
            point_labels=point_labels,
            multimask_output=False,
        )
        mask = masks[0].astype(np.uint8)

        poly_points = self._mask_to_polygon(mask, h, w)
        if not poly_points:
            return None

        return self._build_polygon_result(poly_points, labels, w, h)

    def _handle_rectangle(
        self,
        val: dict[str, Any],
        h: int,
        w: int,
        region_id: str | None = None,
    ) -> dict[str, Any] | None:
        """Process a rectangle box into a SAM box prompt."""
        labels = val.get("rectanglelabels", []) or val.get("labels", [])
        if not labels:
            return None

        x_px = val["x"] * w / 100
        y_px = val["y"] * h / 100
        w_px = val["width"] * w / 100
        h_px = val["height"] * h / 100
        box = np.array([x_px, y_px, x_px + w_px, y_px + h_px], dtype=np.float32)

        logger.debug("SAM box prompt: %s | Label: %s", box, labels)

        masks, _, _ = self.sam_predictor.predict(box=box, multimask_output=False)
        mask = masks[0].astype(np.uint8)

        poly_points = self._mask_to_polygon(mask, h, w)
        if not poly_points:
            return None

        return self._build_polygon_result(poly_points, labels, w, h, region_id=region_id)

    def _process_auto_detect(
        self,
        image: NDArray[np.uint8],
        h: int,
        w: int,
    ) -> list[dict[str, Any]]:
        """Run YOLO auto-detection + SAM segmentation."""
        settings = get_settings()
        results: list[dict[str, Any]] = []

        yolo_results = self.yolo(image, conf=settings.yolo_confidence_threshold)

        if len(yolo_results[0].boxes) == 0:
            return results

        boxes = yolo_results[0].boxes.xyxy.cpu().numpy()
        class_ids = yolo_results[0].boxes.cls.cpu().numpy()
        names = yolo_results[0].names

        for i, box in enumerate(boxes):
            label_name = names[int(class_ids[i])]

            masks, _, _ = self.sam_predictor.predict(box=box, multimask_output=False)
            mask = masks[0].astype(np.uint8)

            poly_points = self._mask_to_polygon(mask, h, w)
            if poly_points:
                results.append(
                    self._build_polygon_result(poly_points, [label_name], w, h)
                )

        return results

    def _build_polygon_result(
        self,
        poly_points: list[list[float]],
        labels: list[str],
        w: int,
        h: int,
        region_id: str | None = None,
    ) -> dict[str, Any]:
        """Build a single polygon annotation result dict."""
        result: dict[str, Any] = {
            "from_name": self.output_from_name,
            "to_name": self.output_to_name,
            "type": "polygonlabels",
            "original_width": w,
            "original_height": h,
            "image_rotation": 0,
            "value": {
                "points": poly_points,
                "polygonlabels": labels,
            },
        }
        if region_id is not None:
            result["id"] = region_id
        return result

    def _mask_to_polygon(
        self,
        mask: NDArray[np.uint8],
        h: int,
        w: int,
    ) -> list[list[float]] | None:
        """Convert a SAM binary mask to polygon points in Label Studio percentage format."""
        settings = get_settings()

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return None

        largest = max(contours, key=cv2.contourArea)
        epsilon = settings.polygon_simplify_epsilon * cv2.arcLength(largest, True)
        approx = cv2.approxPolyDP(largest, epsilon, True)

        return [[p[0][0] / w * 100, p[0][1] / h * 100] for p in approx]

    def _cleanup_memory(self) -> None:
        """Release GPU memory and run garbage collection."""
        self.sam_predictor.reset_image()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        self._cleanup_old_jobs()

    def _cleanup_old_jobs(self) -> None:
        """Remove old job directories, keeping only the N most recent."""
        settings = get_settings()
        keep = settings.job_cleanup_keep_recent
        base_dir = os.path.dirname(os.path.abspath(__file__))
        state_file = os.path.join(base_dir, ".auto_polygon_state.json")

        job_dirs = [
            (int(name), os.path.join(base_dir, name))
            for name in os.listdir(base_dir)
            if os.path.isdir(os.path.join(base_dir, name)) and name.isdigit()
        ]

        if len(job_dirs) <= keep:
            return

        job_dirs.sort(key=lambda x: x[0])
        to_remove = job_dirs[:-keep]

        removed_ids: list[str] = []
        for job_id, path in to_remove:
            try:
                shutil.rmtree(path)
                removed_ids.append(str(job_id))
            except OSError:
                logger.warning("Failed to remove job directory: %s", path, exc_info=True)

        if removed_ids and os.path.exists(state_file):
            try:
                with open(state_file) as f:
                    state = json.load(f)
                state["processed_job_ids"] = [
                    jid
                    for jid in state.get("processed_job_ids", [])
                    if jid not in removed_ids
                ]
                with open(state_file, "w") as f:
                    json.dump(state, f, indent=2)
            except (OSError, json.JSONDecodeError):
                logger.warning("Failed to update state file", exc_info=True)

        if removed_ids:
            logger.info("Cleaned up %d old job dirs, kept %d", len(removed_ids), keep)
