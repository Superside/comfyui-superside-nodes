"""Play a video backwards: the last frame becomes the first.

Local work, no API call and nothing to pay for. ComfyUI hands a VIDEO over as
components - a frame batch, a frame rate and optionally an audio waveform - so
reversing is a flip of the frame batch, and the node rebuilds a VIDEO from the
result rather than returning a bare image batch. That matters: an image batch
would drop the frame rate and the audio, and a SaveVideo downstream would then
have to be told the fps again by hand.

The audio gets its own choice, because there is no single right answer. Flipped
audio keeps sound locked to picture - useful for a reversed action - but speech
becomes gibberish, so a run that only wants the visuals backwards can keep the
original track or drop it.
"""

import logging

import torch

logger = logging.getLogger(__name__)

AUDIO_MODES = ["reverse", "keep", "drop"]


class SupersideReverseVideoNode:
    """Reverse a video, audio included or not."""

    CATEGORY = "Superside"
    RETURN_TYPES = ("VIDEO", "STRING")
    RETURN_NAMES = ("video", "info")
    FUNCTION = "reverse"
    DISPLAY_NAME = "Reverse Video"
    DESCRIPTION = (
        "Play a video backwards - the last frame becomes the first. Returns a "
        "VIDEO, so the frame rate and the audio survive and SaveVideo connects "
        "straight in. Runs locally: no API key, no cost. `audio` picks whether "
        "the sound is reversed with the picture, left as it was, or dropped."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {
                    "tooltip": "The video to play backwards.",
                }),
            },
            "optional": {
                "audio": (AUDIO_MODES, {
                    "default": "reverse",
                    "tooltip": "What happens to the sound.\n"
                               "  reverse - flipped with the picture, so it "
                               "stays in sync. Speech turns to gibberish.\n"
                               "  keep - the original track over reversed "
                               "picture, so sound and image no longer line up.\n"
                               "  drop - no audio at all.",
                }),
            },
        }

    @staticmethod
    def _reverse_audio(audio, mode):
        """Flip the waveform along time, or pass it through, or drop it."""
        if audio is None or mode == "drop":
            return None
        if mode == "keep":
            return audio

        waveform = audio.get("waveform")
        if waveform is None:
            return audio
        # ComfyUI audio is [batch, channels, samples]: time is the last axis,
        # so only that one is flipped - reversing channels would swap left
        # and right, which is not what reversing a video means.
        flipped = dict(audio)
        flipped["waveform"] = torch.flip(waveform, dims=[-1])
        return flipped

    def reverse(self, video, audio="reverse"):
        try:
            from comfy_api.latest import InputImpl, Types
        except ImportError as exc:
            raise RuntimeError(
                "This node needs ComfyUI's native VIDEO type "
                "(comfy_api.latest), which this install does not have. "
                "Please update ComfyUI."
            ) from exc

        components = video.get_components()

        images = components.images
        if images is None or len(images) == 0:
            raise ValueError("The video has no frames to reverse.")

        frame_count = int(images.shape[0])
        reversed_images = torch.flip(images, dims=[0])

        reversed_audio = self._reverse_audio(components.audio, audio)

        # An alpha channel does not survive the rebuild: VideoFromComponents
        # writes only images and audio when it encodes. Say so rather than
        # quietly handing back a video that lost its transparency.
        if getattr(components, "alpha", None) is not None:
            logger.warning(
                "The source video carries an alpha channel, which is dropped "
                "when the reversed video is encoded."
            )

        # Carry the source's bit depth and colour space across, so reversing an
        # HDR clip does not quietly flatten it to 8-bit sRGB.
        kwargs = {}
        for name, getter in (("bit_depth", "get_bit_depth"),
                             ("color_space", "get_color_space")):
            try:
                value = getattr(video, getter)()
            except Exception:
                continue
            if value is not None:
                kwargs[name] = value

        out = InputImpl.VideoFromComponents(
            Types.VideoComponents(
                images=reversed_images,
                audio=reversed_audio,
                frame_rate=components.frame_rate,
            ),
            **kwargs,
        )

        rate = float(components.frame_rate) if components.frame_rate else 0.0
        lines = [
            "frames       %d, reversed" % frame_count,
            "frame rate   %.3f fps" % rate,
            "duration     %.2f s" % (frame_count / rate if rate else 0.0),
            "audio        %s" % (
                "none in the source" if components.audio is None
                else {"reverse": "reversed with the picture",
                      "keep": "left as it was, no longer in sync",
                      "drop": "dropped"}[audio]),
        ]
        if kwargs:
            lines.append("kept         %s" % ", ".join(
                "%s=%s" % (k, v) for k, v in kwargs.items()))
        logger.info("Reverse Video: %d frames at %.3f fps", frame_count, rate)
        return (out, "\n".join(lines))
