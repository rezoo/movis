from typing import Hashable, Optional

import cv2
import numpy as np

from ..effect import Effect
from ..enum import Direction
from ..imgproc import alpha_composite
from ..transform import Transform, TransformValue
from .layer import Layer


class LayerItem:
    def __init__(
            self, layer: Layer, name: str = 'layer', transform: Optional[Transform] = None,
            offset: float = 0.0, start_time: float = 0.0, end_time: Optional[float] = None,
            visible: bool = True):
        self.layer: Layer = layer
        self.name: str = name
        self.transform: Transform = transform if transform is not None else Transform()
        self.offset: float = offset
        self.start_time: float = start_time
        self.end_time: float = end_time if end_time is not None else layer.duration
        self.visible: bool = visible
        self._effects: list[Effect] = []

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    def add_effect(self, effect: Effect) -> Effect:
        self._effects.append(effect)
        return effect

    def get_key(self, layer_time: float) -> tuple[Hashable, Hashable, Hashable]:
        if not self.visible:
            return (None, None, None)
        transform_key = self.transform.get_current_value(layer_time)
        layer_key = self.layer.get_key(layer_time) if hasattr(self.layer, 'get_key') else layer_time

        def get_effect_key(e: Effect) -> Optional[Hashable]:
            return e.get_key(layer_time) if hasattr(e, 'get_key') else layer_time

        effects_key = None if len(self._effects) == 0 else tuple([get_effect_key(e) for e in self._effects])
        return (transform_key, layer_key, effects_key)

    def _composite(
        self, bg_image: np.ndarray, time: float,
        parent: tuple[int, int] = (0, 0),
        preview_level: int = 1,
    ) -> np.ndarray:
        # Retrieve layer image
        layer_time = time - self.offset
        if layer_time < self.start_time or self.end_time <= layer_time:
            return bg_image
        fg_image = self(time)
        if fg_image is None:
            return bg_image

        # Get affine matrix and transform layer image
        p = self.transform.get_current_value(layer_time)
        result = _get_fixed_affine_matrix(fg_image, p, preview_level=preview_level)
        if result is None:
            return bg_image
        affine_matrix_fixed, (W, H), (offset_x, offset_y) = result
        fg_image_transformed = cv2.warpAffine(
            fg_image, affine_matrix_fixed, dsize=(W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        # Composite bg_image and fg_image
        bg_image = alpha_composite(
            bg_image, fg_image_transformed,
            position=(offset_x - parent[0], offset_y - parent[1]),
            opacity=p.opacity)
        return bg_image

    def __call__(self, time: float) -> Optional[np.ndarray]:
        layer_time = time - self.offset
        if not self.visible:
            return None
        frame = self.layer(layer_time)
        if frame is None:
            return None
        for effect in self._effects:
            frame = effect(frame, layer_time)
        return frame

    def __repr__(self) -> str:
        return f"LayerItem(name={self.name!r}, layer={self.layer!r}, transform={self.transform!r}, " \
            f"offset={self.offset}, visible={self.visible})"


def _get_fixed_affine_matrix(
    fg_image: np.ndarray, p: TransformValue,
    preview_level: int = 1
) -> Optional[tuple[np.ndarray, tuple[int, int], tuple[int, int]]]:
    h, w = fg_image.shape[:2]

    T1, SR = _get_T1(p), _get_SR(p)
    T2 = _get_T2(p, (w, h), p.origin_point)
    M = T1 @ SR @ T2
    P = np.array([
        [1 / preview_level, 0, 0],
        [0, 1 / preview_level, 0],
        [0, 0, 1]], dtype=np.float64)
    affine_matrix = (P @ M)[:2]

    corners_layer = np.array([
        [0, 0, 1],
        [0, h, 1],
        [w, 0, 1],
        [w, h, 1]], dtype=np.float64)
    corners_global = corners_layer @ affine_matrix.transpose()
    min_coords = np.ceil(corners_global.min(axis=0))
    max_coords = np.floor(corners_global.max(axis=0))
    WH = (max_coords - min_coords).astype(np.int32)
    W, H = WH[0], WH[1]
    if W == 0 or H == 0:
        return None
    offset_x, offset_y = int(min_coords[0]), int(min_coords[1])

    Pf = np.array([
        [1 / preview_level, 0, - offset_x],
        [0, 1 / preview_level, - offset_y],
        [0, 0, 1]], dtype=np.float64)
    affine_matrix_fixed = (Pf @ M)[:2]
    return affine_matrix_fixed, (W, H), (offset_x, offset_y)


def _get_T1(p: TransformValue) -> np.ndarray:
    return np.array([
        [1, 0, p.position[0] + p.anchor_point[0]],
        [0, 1, p.position[1] + p.anchor_point[1]],
        [0, 0, 1]], dtype=np.float64)


def _get_SR(p: TransformValue) -> np.ndarray:
    cos_t = np.cos((2 * np.pi * p.rotation) / 360)
    sin_t = np.sin((2 * np.pi * p.rotation) / 360)
    SR = np.array([
        [p.scale[0] * cos_t, - p.scale[0] * sin_t, 0],
        [p.scale[1] * sin_t, p.scale[1] * cos_t, 0],
        [0, 0, 1]], dtype=np.float64)
    return SR


def _get_T2(p: TransformValue, size: tuple[int, int], origin_point: Direction) -> np.ndarray:
    center_point = Direction.to_vector(
        origin_point, (float(size[0]), float(size[1])))
    T2 = np.array([
        [1, 0, - p.anchor_point[0] - center_point[0]],
        [0, 1, - p.anchor_point[1] - center_point[1]],
        [0, 0, 1]], dtype=np.float64)
    return T2
