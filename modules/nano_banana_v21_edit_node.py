"""Nano Banana 2.1 Edit: google/nano-banana-2.1/edit.

Same input shape as Nano Banana 2 - up to six reference images, one prompt, the
same aspect-ratio and resolution controls - so this subclasses that node and
only declares what differs. Note the endpoint lives under google/ rather than
fal-ai/, unlike the 2.0 and Pro ones.
"""

import logging

from .nano_banana_v2_edit_node import SupersideNanoBananaV2EditNode

logger = logging.getLogger(__name__)


class SupersideNanoBananaV21EditNode(SupersideNanoBananaV2EditNode):
    """Edit images with Nano Banana 2.1."""

    ENDPOINT = "google/nano-banana-2.1/edit"

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("images", "description")
    FUNCTION = "generate"
    DISPLAY_NAME = "Nano Banana 2.1 Edit"
    DESCRIPTION = (
        "Edit images using google/nano-banana-2.1/edit with up to 6 image "
        "inputs for context-aware image editing."
    )

    def generate(self, prompt, api_key, **kwargs):
        """Same flow as 2.0, against the 2.1 endpoint.

        The parent hard-codes its own endpoint and reports failures under the
        2.0 name, so both are overridden here rather than inherited - an error
        that names the wrong model sends anyone debugging it to the wrong place.
        """
        try:
            client = self.get_client(api_key)
            arguments = self.prepare_arguments(client, prompt, **kwargs)
            result = self.call_api(client, self.ENDPOINT, arguments)
            images = self.process_images(result)
            return (images[0], result.get("description", ""))
        except Exception as exc:
            logger.error("Nano Banana 2.1 edit failed: %s", exc)
            raise RuntimeError("Nano Banana 2.1 edit failed: %s" % exc) from exc
