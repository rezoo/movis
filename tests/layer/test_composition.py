import os
import tempfile

import numpy as np

import pytest
import movis as mv
from movis.layer import Composition


def test_create_composition():
    scene = Composition(size=(640, 480), duration=1.0)
    img = scene(0.0)
    assert img.shape == (480, 640, 4)
    assert img.dtype == np.uint8
    img = scene(1.0)
    assert img is None


def test_composition_add_layer():
    scene = Composition(size=(640, 480), duration=1.0)

    assert len(scene.layers) == 0
    scene.add_layer(mv.layer.Rectangle(size=(256, 256), color="#ffffff"))
    assert len(scene.layers) == 1
    assert isinstance(scene.layers[0], mv.layer.LayerItem)
    img = scene(0.0)
    assert img.shape == (480, 640, 4)
    assert img.dtype == np.uint8
    img = scene(1.0)
    assert img is None


def test_composition_add_layer_with_dict():
    scene = Composition(size=(640, 480), duration=1.0)

    assert len(scene) == 0
    scene['layer'] = mv.layer.Rectangle(size=(256, 256), color='#ffffff')
    assert len(scene) == 1
    scene['layer_item'] = mv.layer.LayerItem(
        mv.layer.Rectangle(size=(100, 100), color="#ffffff"),
        transform=mv.Transform(position=(0, 0)),
    )
    assert len(scene) == 2


def test_composition_add_layer_with_name():
    scene = Composition(size=(640, 480), duration=1.0)

    item = scene.add_layer(mv.layer.Rectangle((256, 256), color='#ffffff'), name='layer')
    assert len(scene.layers) == 1
    assert 'layer' in scene
    item2 = scene['layer']
    assert item is item2


def test_composition_with_start_time():
    scene = Composition(size=(640, 480), duration=1.0)

    scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=1.0),
        name='layer', start_time=0.5)
    assert 0 < len(scene)
    img = scene(0.0)
    assert img.max() == 0
    img = scene(0.5)
    assert 0 < img.max()


def test_composition_with_end_time():
    scene = Composition(size=(640, 480), duration=1.0)

    scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=1.0),
        name='layer', end_time=0.5)
    assert 0 < len(scene)
    img = scene(0.0)
    assert 0 < img.max()
    img = scene(0.5)
    assert img.max() == 0


def test_composition_with_offset():
    scene = Composition(size=(640, 480), duration=1.0)

    scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=0.5),
        name='layer', offset=0.5)
    assert 0 < len(scene)
    img = scene(0.0)
    assert img.max() == 0
    img = scene(0.5)
    assert 0 < img.max()


def test_composition_keys():
    scene = Composition(size=(640, 480), duration=1.0)
    scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=0.5),
        name='layer', offset=0.5)
    keys = list(scene.keys())
    assert len(keys) == 1
    assert keys[0] == 'layer'


def test_composition_values():
    scene = Composition(size=(640, 480), duration=1.0)
    item = scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=0.5),
        name='layer')
    values = list(scene.values())
    assert len(values) == 1
    assert values[0] is item


def test_composition_items():
    scene = Composition(size=(640, 480), duration=1.0)
    layer_item = scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=0.5),
        name='layer')
    items = list(scene.items())
    assert len(items) == 1
    assert items[0][0] == 'layer'
    assert items[0][1] is layer_item


def test_composition_properties():
    scene = Composition(size=(640, 480), duration=1.0)
    scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=0.5),
        name='layer')
    assert scene.size == (640, 480)
    assert scene.duration == 1.0
    assert scene.preview_level == 1


def test_composition_pop_layer():
    scene = Composition(size=(640, 480), duration=1.0)
    item1 = scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=0.5),
        name='layer')
    assert len(scene) == 1
    item2 = scene.pop_layer('layer')
    assert item1 is item2
    assert len(scene) == 0

    with pytest.raises(KeyError):
        scene.pop_layer('layer')


def test_composition_del():
    scene = Composition(size=(640, 480), duration=1.0)
    scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=0.5),
        name='layer')
    assert len(scene) == 1
    del scene['layer']
    assert len(scene) == 0

    with pytest.raises(KeyError):
        del scene['layer']


def test_composition_clear():
    scene = Composition(size=(640, 480), duration=1.0)
    scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=0.5),
        name='layer')
    assert len(scene) == 1
    scene.clear()
    assert len(scene) == 0

    with pytest.raises(KeyError):
        del scene['layer']


def test_composition_get_key():
    scene = Composition(size=(640, 480), duration=1.0)
    scene.add_layer(
        mv.layer.Rectangle((128, 128), color='#ffffff', duration=0.5),
        name='layer')
    assert len(scene) == 1
    key1 = scene.get_key(0.0)
    key2 = scene.get_key(0.25)
    assert key1 == key2
    key3 = scene.get_key(0.5)
    key4 = scene.get_key(0.75)
    assert key2 != key3
    assert key3 == key4
    key5 = scene.get_key(1.0)
    assert key5 is None


def test_composition_preview():
    scene = Composition(size=(640, 480), duration=1.0)
    scene.add_layer(
        mv.layer.Rectangle((256, 256), color='#ffffff', duration=1.0),
        name='layer')

    with scene.preview(level=1):
        assert scene.preview_level == 1
        img = scene(0.0)
        assert img.shape == (480, 640, 4)
        assert img.dtype == np.uint8

    with scene.preview(level=2):
        assert scene.preview_level == 2
        img = scene(0.0)
        assert img.shape == (240, 320, 4)
        assert img.dtype == np.uint8
    img = scene(0.0)
    assert img.shape == (480, 640, 4)
    assert img.dtype == np.uint8

    with scene.preview(level=4):
        assert scene.preview_level == 4
        img = scene(0.0)
        assert img.shape == (120, 160, 4)
        assert img.dtype == np.uint8
    img = scene(0.0)
    assert img.shape == (480, 640, 4)
    assert img.dtype == np.uint8


def test_composition_preview_level():
    scene = Composition(size=(640, 480), duration=1.0)
    scene.add_layer(
        mv.layer.Rectangle((256, 256), color='#ffffff', duration=1.0),
        name='layer')
    scene.preview_level = 1
    img = scene(0.0)
    assert img.shape == (480, 640, 4)
    assert img.dtype == np.uint8

    scene.preview_level = 2
    img = scene(0.0)
    assert img.shape == (240, 320, 4)
    assert img.dtype == np.uint8


def test_composition_write_video():
    scene = Composition(size=(640, 480), duration=1.0)
    scene.add_layer(
        mv.layer.Rectangle((256, 256), color='#ffffff', duration=1.0),
        name='layer')

    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    file_name = temp_file.name
    temp_file.close()

    try:
        scene.write_video(file_name, fps=3.0)
    finally:
        os.remove(file_name)


def test_composition_get_coords():
    scene = Composition(size=(32, 16), duration=1.0)
    item = scene.add_layer(
        mv.layer.Image.from_color((32, 16), color='white'),
        scale=1.0,
    )
    coords = item.get_composition_coords(
       time=0.0, layer_coords=np.array([[0, 0], [32, 16]], dtype=float))
    assert np.all(coords == np.array([[0, 0], [32, 16]], dtype=float))

    scene = Composition(size=(32, 16), duration=1.0)
    item = scene.add_layer(
        mv.layer.Image.from_color((32, 16), color='white'),
        scale=0.5,
    )
    coords = item.get_composition_coords(
        time=0.0, layer_coords=np.array([[0, 0], [32, 16]], dtype=float))
    assert np.all(coords == np.array([[8, 4], [24, 12]], dtype=float))
