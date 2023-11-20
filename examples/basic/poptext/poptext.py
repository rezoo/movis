import json

import movis as mv


def main():
    scene = mv.layer.Composition(size=(1080, 1900), duration=4.0)
    scene.add_layer(
        mv.layer.Image("background.png"),
    )

    scene.add_layer(
        mv.layer.Audio("output.ogg"),
        name="soundclip",
    )

    timings = []

    with open("timings.json") as f:
        timings = json.load(f)

    last_timing = {"name": None, "time_seconds": 0}

    for timing in timings:
        scene.add_layer(
            mv.layer.Text(
                timing["name"],
                font_size=92,
                font_family="Helvetica",
                color="#ffffff",
            ),
            name=timing["name"],
            offset=last_timing["time_seconds"],
            end_time=timing["time_seconds"] - last_timing["time_seconds"],
            blending_mode="normal",
        )

        last_timing = timing

        scene[timing["name"]].add_effect(mv.effect.DropShadow(offset=10.0))
        scene[timing["name"]].transform.rotation.init_value = -10.0
        scene[timing["name"]].scale.enable_motion().extend(
            keyframes=[0.0, 0.25],
            values=[0.75, 1.0],
            easings=["ease_in"],
        )

    scene.write_video("output.mp4")


if __name__ == '__main__':
    main()
