from typing import Hashable, Optional, Union

import cv2
import numpy as np

from ..effect import Effect
from ..enum import Direction
from ..imgproc import BlendingMode, alpha_composite
from ..transform import Transform, TransformValue
from .layer import Layer


class Component:

    def __init__(
            self, name: str, layer: Layer, transform: Optional[Transform] = None,
            offset: float = 0.0, start_time: float = 0.0, end_time: float = 0.0,
            visible: bool = True,
            blending_mode: Union[BlendingMode, str] = BlendingMode.NORMAL,
            origin_point: Union[Direction, str] = Direction.CENTER,
            alpha_matte: Optional["Component"] = None):
        self.name: str = name
        self.layer: Layer = layer
        self.transform: Transform = transform if transform is not None else Transform()
        self.offset: float = offset
        self.start_time: float = start_time
        self.end_time: float = end_time if end_time == 0.0 else self.layer.duration
        self.visible: bool = visible
        mode = BlendingMode.from_string(blending_mode) if isinstance(blending_mode, str) else blending_mode
        self.blending_mode: BlendingMode = mode
        self.origin_point = Direction.from_string(origin_point) if isinstance(origin_point, str) else origin_point
        self._alpha_matte: Optional[Component] = alpha_matte
        self._effects: list[Effect] = []

    @property
    def duration(self) -> float:
        return self.end_time - self.start_time

    @property
    def composition_start_time(self) -> float:
        return self.offset + self.start_time

    @property
    def composition_end_time(self) -> float:
        return self.offset + self.end_time

    def enable_alpha_matte(self, alpha_matte: "Component") -> "Component":
        self._alpha_matte = alpha_matte
        return alpha_matte

    def disable_alpha_matte(self) -> Optional["Component"]:
        if self._alpha_matte is None:
            return None
        prev_alpha_matte = self._alpha_matte
        self._alpha_matte = None
        return prev_alpha_matte

    def add_effect(self, effect: Effect) -> Effect:
        self._effects.append(effect)
        return effect

    def get_key(self, layer_time: float) -> tuple[Hashable, Hashable, Hashable, Hashable]:
        if not self.visible:
            return (None, None, None, None)
        transform_key = self.transform.get_current_value(layer_time)
        layer_key = self.layer.get_key(layer_time) if hasattr(self.layer, 'get_key') else layer_time
        alpha_matte_key = self._alpha_matte.get_key(layer_time) \
            if self._alpha_matte is not None and hasattr(self._alpha_matte, 'get_key') else None

        def get_effect_key(e: Effect) -> Optional[Hashable]:
            return e.get_key(layer_time) if hasattr(e, 'get_key') else layer_time

        effects_key = None if len(self._effects) == 0 else tuple([get_effect_key(e) for e in self._effects])
        return (transform_key, layer_key, alpha_matte_key, effects_key)

    def _composite(
        self, bg_image: np.ndarray, time: float,
        parent: tuple[int, int] = (0, 0),
        alpha_matte_mode: bool = False,
    ) -> np.ndarray:
        layer_time = time - self.offset
        if layer_time < self.start_time or self.end_time <= layer_time:
            return bg_image
        fg_image = self(time)
        if fg_image is None:
            return bg_image
        p = self.transform.get_current_value(layer_time)
        result = self._get_fixed_affine_matrix(fg_image, p)
        if result is None:
            return bg_image
        affine_matrix_fixed, (W, H), (offset_x, offset_y) = result
        fg_image_transformed = cv2.warpAffine(
            fg_image, affine_matrix_fixed, dsize=(W, H),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        bg_image = alpha_composite(
            bg_image, fg_image_transformed,
            position=(offset_x - parent[0], offset_y - parent[1]),
            opacity=p.opacity, blending_mode=self.blending_mode,
            alpha_matte_mode=alpha_matte_mode)
        return bg_image

    def _get_fixed_affine_matrix(
        self, fg_image: np.ndarray, p: TransformValue,
    ) -> Optional[tuple[np.ndarray, tuple[int, int], tuple[int, int]]]:
        h, w = fg_image.shape[:2]

        T1 = _get_T1(p)
        SR = _get_SR(p)
        T2 = _get_T2(p, (w, h), self.origin_point)
        SR_T2 = SR @ T2
        affine_matrix = (T1 @ SR_T2)[:2]

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

        T1_fixed = _get_T1(p, offset=(offset_x, offset_y))
        affine_matrix_fixed = (T1_fixed @ SR_T2)[:2]
        return affine_matrix_fixed, (W, H), (offset_x, offset_y)

    def __call__(self, time: float) -> Optional[np.ndarray]:
        layer_time = time - self.offset
        if not self.visible:
            return None
        frame = self.layer(layer_time)
        if frame is None:
            return None
        for effect in self._effects:
            frame = effect(frame, layer_time)

        if self._alpha_matte is not None:
            p = self.transform.get_current_value(layer_time)
            result = self._get_fixed_affine_matrix(frame, p)
            if result is None:
                return frame
            _, _, (offset_x, offset_y) = result
            frame = self._alpha_matte._composite(
                frame, time, parent=(offset_x, offset_y), alpha_matte_mode=True)
        return frame

    def __repr__(self) -> str:
        return f"LayerItem(name={self.name!r}, layer={self.layer!r}, transform={self.transform!r}, " \
            f"offset={self.offset}, visible={self.visible}, blending_mode={self.blending_mode})"


def _get_T1(p: TransformValue, offset: tuple[int, int] = (0, 0)) -> np.ndarray:
    return np.array([
        [1, 0, p.position[0] + p.anchor_point[0] - offset[0]],
        [0, 1, p.position[1] + p.anchor_point[1] - offset[1]],
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
