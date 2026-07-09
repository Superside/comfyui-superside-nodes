# ComfyUI Superside Nodes

Custom ComfyUI nodes wrapping [fal.ai](https://fal.ai) models for image editing, image-to-video, upscaling, vision/language, and region selection - built for Superside production workflows.

## Install

1. Clone this repo into your ComfyUI `custom_nodes` directory.

   **Windows:**
   ```
   cd ComfyUI_windows_portable\ComfyUI\custom_nodes
   git clone https://github.com/Superside/comfyui-superside-nodes.git
   ```

   **Mac:**
   ```
   cd ComfyUI/custom_nodes
   git clone https://github.com/Superside/comfyui-superside-nodes.git
   ```

2. Install dependencies (into whichever Python environment ComfyUI itself runs on):
   ```
   pip install -r requirements.txt
   ```
3. Restart ComfyUI. All nodes appear under the **Superside** category in the node menu.

### Video walkthrough

[![Watch the installation walkthrough](https://img.youtube.com/vi/wcp9fhMJnXA/maxresdefault.jpg)](https://youtu.be/wcp9fhMJnXA)

## API key

**There is no config file and no environment variable for the fal.ai API key.** Every node has a required `api_key` text input - paste your key directly into that widget on each node. Ask your project lead for the key; it is never stored in this repository.

If `api_key` is left empty, the node fails immediately with a clear error instead of silently using someone else's key or a stale config file.

## Node reference

Every node's display name in ComfyUI's search/menu is prefixed with **"Superside "** (e.g. `Superside Seedream V5 Pro Edit`, `Superside Bria Background Standardizer`) - type "Superside" in the node search to see the whole set.

Nodes are grouped by task below. For every node: **Inputs** lists required inputs first, then optional ones (with defaults); **Outputs** lists the return values in order.

### Image editing & generation

#### Seedream V5 Pro Edit (`SupersideSeedreamV5ProEditNode`)
Grounded, region-precise editing with ByteDance Seedream V5 Pro - changes one element while keeping the rest of the frame intact. Up to 10 reference images.
- **Inputs:** `prompt`, `image_1`, `api_key` · optional: `image_2`-`image_10`, `size_mode` (preset/custom), `image_size` (preset, default `auto_2K`), `width`/`height` (custom mode), `output_format` (jpeg/png), `num_images` (1-6), `enable_safety_checker`, `sync_mode`
- **Outputs:** `images` (IMAGE), `info` (STRING - result URL)

#### Seedream V4.5 Edit (`SupersideSeedreamV45EditNode`)
Broader multi-reference editing (up to 10 images) with higher max resolution and multi-image output.
- **Inputs:** `prompt`, `image_1`, `api_key` · optional: `image_2`-`image_10`, `size_mode`, `image_size` (up to `auto_4K`), `width`/`height`, `num_images`, `max_images`, `seed` (-1 = random), `enable_safety_checker`, `sync_mode`
- **Outputs:** `images` (IMAGE), `info` (STRING)

#### Nano Banana Pro Edit (`SupersideNanoBananaProEditNode`)
Context-aware image editing, up to 6 reference images, up to 4K output.
- **Inputs:** `prompt`, `image_1`, `api_key` · optional: `image_2`-`image_6`, `num_images`, `aspect_ratio`, `output_format`, `resolution` (1K/2K/4K), `sync_mode`
- **Outputs:** `images` (IMAGE), `description` (STRING)

#### Nano Banana V2 Edit (`SupersideNanoBananaV2EditNode`)
Same family as Pro, with extra controls: seed, safety tolerance, web search grounding, reasoning depth.
- **Inputs:** `prompt`, `image_1`, `api_key` · optional: `image_2`-`image_6`, `num_images`, `seed`, `aspect_ratio`, `output_format`, `safety_tolerance`, `sync_mode`, `resolution` (0.5K-4K), `limit_generations`, `enable_web_search`, `thinking_level`
- **Outputs:** `images` (IMAGE), `description` (STRING)

#### GPT Image 2 Edit (`SupersideGPTImage2EditNode`)
OpenAI GPT Image 2 editing - mask-based inpainting, preset/aspect-ratio/custom sizing up to 4K. Uses fal's queued execution path internally (polls until complete).
- **Inputs:** `prompt`, `image_1`, `api_key` · optional: `image_2`-`image_6`, `mask_image`, `size_mode` (preset/aspect_ratio/custom), `image_size`, `aspect_ratio`, `resolution`, `width`/`height` (custom), `quality`, `num_images`, `output_format`, `sync_mode`
- **Outputs:** `images` (IMAGE), `info` (STRING)

#### Grok Imagine Image Quality Edit (`SupersideGrokImagineImageQualityEditNode`)
xAI Grok Imagine editing, up to 3 reference images, returns the model's revised prompt.
- **Inputs:** `prompt`, `image_1`, `api_key` · optional: `image_2`, `image_3`, `aspect_ratio`, `resolution` (1k/2k), `output_format`, `num_images`, `sync_mode`
- **Outputs:** `images` (IMAGE), `revised_prompt` (STRING)

#### Flux Kontext Max Multi-Image Node (`SupersideFluxKontextMaxMultiImageNode`)
FLUX.1 Kontext [Max] context-aware generation from up to 4 images.
- **Inputs:** `prompt`, `api_key` · optional: `image_1`-`image_4`, `seed`, `guidance_scale`, `num_images`, `safety_tolerance` (1-6), `output_format`, `aspect_ratio`
- **Outputs:** `IMAGE`

#### Juggernaut Flux Pro Image-to-Image (`SupersideJuggernautFluxProImg2ImgNode`)
High-realism image-to-image stylization.
- **Inputs:** `image`, `prompt`, `api_key` · optional: `strength`, `num_inference_steps`, `seed`, `guidance_scale`, `num_images`, `enable_safety_checker`
- **Outputs:** `IMAGE`

#### Wan 2.5 Image-to-Image (`SupersideWan25ImageToImageNode`)
Single or dual-reference editing with Wan 2.5.
- **Inputs:** `prompt`, `image_1`, `api_key` · optional: `image_2`, `negative_prompt`, `image_size`, `num_images` (1-4), `seed`
- **Outputs:** `IMAGE`

### Background tools

#### Bria Background Standardizer - Hex Color (`SupersideBriaBackgroundStandardizerNode`)
Cuts out the subject with Bria RMBG 2.0 (`fal-ai/bria/background/remove`) and composites it **locally** onto an exact solid hex color - no generative model touches the subject or the background pixels. Use this to batch-homogenize backgrounds (e.g. avatar sets) without any quality drift.
- **Inputs:** `image`, `hex_color` (e.g. `#F5F5F5`), `api_key` · optional: `edge_feather` (0-15px, softens the cutout edge), `sync_mode`
- **Outputs:** `image` (IMAGE), `info` (STRING - resolved hex + source cutout URL)

#### Bria Replace Background (`SupersideBriaReplaceBackgroundNode`)
Prompt-driven background replacement with realistic lighting/perspective. Use this when you want a *scene*, not an exact flat color (for that, use the Standardizer above).
- **Inputs:** `image`, `prompt`, `api_key` · optional: `negative_prompt`, `steps_num`, `seed` (-1 = random), `sync_mode`
- **Outputs:** `image` (IMAGE), `info` (STRING - result URL)

### Image-to-video

#### Kling 2.1 Image-to-Video (`SupersideKling21ImageToVideoNode`)
Three quality tiers in one node.
- **Inputs:** `prompt`, `image`, `model_tier` (master/pro/standard), `api_key` · optional: `tail_image` (end-frame, Pro tier only), `duration` (5/10s), `negative_prompt`, `cfg_scale`
- **Outputs:** video URL (STRING)

#### Kling 2.5 Turbo Pro Image-to-Video (`SupersideKling25TurboProImageToVideoNode`)
Top-tier cinematic single-tier model, better motion fluidity than 2.1.
- **Inputs:** `prompt`, `image`, `api_key` · optional: `duration` (5/10s), `negative_prompt`, `cfg_scale`
- **Outputs:** video URL (STRING)

#### Seedance Lite Image-to-Video (`SupersideSeedanceLiteImageToVideoNode`)
Cost-efficient tier, up to 4 reference images.
- **Inputs:** `prompt`, `reference_image_1`, `api_key` · optional: `reference_image_2`-`4`, `aspect_ratio`, `resolution` (480p/720p), `duration`, `camera_fixed`, `seed`, `enable_safety_checker`
- **Outputs:** video URL (STRING)

#### Seedance Pro Image-to-Video (`SupersideSeedanceProImageToVideoNode`)
Higher quality tier, up to 1080p, with end-frame control.
- **Inputs:** `prompt`, `image`, `api_key` · optional: `end_image`, `aspect_ratio`, `resolution` (480p/720p/1080p), `duration`, `camera_fixed`, `seed`, `enable_safety_checker`
- **Outputs:** video URL (STRING)

#### Wan 2.5 Image-to-Video (`SupersideWan25ImageToVideoNode`)
Supports audio-driven video generation and prompt expansion.
- **Inputs:** `prompt`, `image`, `api_key` · optional: `audio_url` (WAV/MP3, 3-30s), `resolution` (480p/720p/1080p), `duration` (5/10s), `negative_prompt`, `enable_prompt_expansion`, `seed`
- **Outputs:** video URL (STRING)

### Upscaling

#### Ideogram Upscale (`SupersideIdeogramUpscaleNode`)
Prompt-guided upscaling with resemblance/detail sliders.
- **Inputs:** `image`, `api_key` · optional: `prompt`, `resemblance`, `detail`, `expand_prompt`, `seed`
- **Outputs:** `IMAGE`

#### PASD Upscaler Node (`SupersidePASDUpscalerNode`)
Pixel-aware stable diffusion super-resolution with ControlNet guidance and wavelet color correction.
- **Inputs:** `image`, `api_key` · optional: `scale`, `steps`, `guidance_scale`, `conditioning_scale`, `prompt`, `negative_prompt`
- **Outputs:** `IMAGE`

#### SeedVR2 Upscale Image (`SupersideSeedVR2UpscaleImageNode`)
Seamless upscaler with target-resolution or scale-factor mode.
- **Inputs:** `image`, `api_key` · optional: `upscale_mode` (target/factor), `upscale_factor`, `target_resolution` (720p-2160p), `seed`, `noise_scale`
- **Outputs:** `IMAGE`

#### SeedVR Upscale Video (`SupersideSeedVRUpscaleVideoNode`)
Video upscaling with temporal consistency (takes a video URL, not an IMAGE tensor).
- **Inputs:** `video_url`, `api_key` · optional: `upscale_factor`, `seed`
- **Outputs:** `video_url` (STRING)

#### Topaz Upscale Image (`SupersideTopazUpscaleImageNode`)
10 Topaz model variants (Standard, CGI, High Fidelity, Recovery, Redefine, Wonder, etc.) with face enhancement, denoise, sharpen, and creative-recovery controls.
- **Inputs:** `image`, `api_key` · optional: `model` (10 variants), `upscale_factor`, `crop_to_fill`, `output_format`, `subject_detection`, `face_enhancement(+creativity/strength)`, `sharpen`, `denoise`, `fix_compression`, `strength`, `creativity`, `texture`, `prompt`, `autoprompt`, `detail`
- **Outputs:** `IMAGE`

### Vision & language

#### Any LLM Text (`SupersideAnyLLMTextNode`)
Text-only chat/completion across many models via OpenRouter on fal.ai (Gemini, Claude 4.6, GPT, Llama, Grok, Kimi).
- **Inputs:** `prompt`, `api_key` · optional: `system_prompt`, `model` (16 options, default `google/gemini-2.5-flash`), `reasoning`, `temperature`, `max_tokens`
- **Outputs:** `output` (STRING), `reasoning` (STRING)

#### Any LLM Vision (`SupersideAnyLLMVisionNode`)
Multi-image (up to 6) vision Q&A across many models, with auto-rescale for large images.
- **Inputs:** `prompt`, `api_key` · optional: `image_1`-`image_6`, `system_prompt`, `model` (21 options), `reasoning`, `priority` (latency/throughput), `auto_rescale_images`, `max_image_dimension`, `temperature`, `max_tokens`
- **Outputs:** `output` (STRING), `reasoning` (STRING)

#### Florence-2 Detailed Caption (`SupersideFlorence2CaptionNode`)
Fixed-purpose auto-caption generator (no prompt needed).
- **Inputs:** `image`, `api_key`
- **Outputs:** `STRING` (caption)

### Region selection

Both region selectors below share the same output contract, so they're interchangeable in downstream masking/inpainting workflows.

#### Florence-2 Smart Region Selector (`SupersideFlorence2RegionSelectorNode`)
Single-region selection (face/upper body/lower body/full body/custom object) using Florence-2 segmentation, with a grounding fallback.
- **Inputs:** `image`, `region_type`, `api_key` · optional: `custom_text` (only when `region_type=object`), `selection_mode` (largest/merge_all), `padding_percent`, `return_rect_mask`
- **Outputs:** `mask` (MASK), `mask_image` (IMAGE), `info` (STRING, JSON), `center_x`, `center_y`, `crop_width`, `crop_height` (INT)

#### SAM 3 Smart Region Selector (`SupersideSAM3RegionSelectorNode`)
Broader vocabulary than Florence (garments, vehicle parts, accessories) plus multi-mask/scoring modes.
- **Inputs:** `image`, `region_type` (19 presets incl. `object`), `api_key` · optional: `custom_text`, `selection_mode` (largest/first/merge_all), `padding_percent`, `return_rect_mask`, `return_multiple_masks`, `max_masks`, `include_scores`, `include_boxes`
- **Outputs:** `mask` (MASK), `mask_image` (IMAGE), `info` (STRING, JSON), `center_x`, `center_y`, `crop_width`, `crop_height` (INT)

### Utility (no API key needed)

These two nodes make no fal.ai calls, so they don't have an `api_key` input.

#### Prompt Box (`SupersidePromptBoxNode`)
A simple text box - write a prompt, connect the STRING output anywhere. Displays the text in the node UI.
- **Inputs:** `prompt`
- **Outputs:** `prompt` (STRING)

#### Prompt Splitter (`SupersidePromptSplitterNode`)
Splits one prompt into up to 10 separate STRING outputs using a separator symbol - useful for feeding individual prompts into multi-image nodes.
- **Inputs:** `prompt`, `separator` (default `*`)
- **Outputs:** `text_1` ... `text_10` (STRING)

## Package layout

```
comfyui-superside-nodes/
├── __init__.py                # Node registration (NODE_CLASS_MAPPINGS, etc.)
├── modules/
│   ├── base_node.py            # SupersideFalNode, ImageProcessingMixin, APIClientMixin, API_KEY_INPUT_SPEC
│   └── <28 node files>
├── web/js/show_text.js        # Read-only result-text display widget for select nodes
├── requirements.txt
└── README.md
```

## Architecture notes

- **`SupersideFalNode.get_client(api_key)`** builds a `fal_client.SyncClient(key=api_key)` scoped to that single call - no global environment mutation, so multiple nodes with different keys never interfere with each other.
- **`ImageProcessingMixin`** handles tensor→PNG upload and API-response→tensor conversion.
- **`APIClientMixin.call_api(client, endpoint, arguments)`** picks synchronous vs. queued execution automatically based on the endpoint (GPT Image 2 uses the queued path; everything else is synchronous).
- All nodes are registered under the **Superside** category.
