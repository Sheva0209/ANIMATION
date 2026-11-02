"""
scene_generator.py
Gemini Imagen API - generates custom scene images with Sheva character (img2img)
for scenes marked with 'generated_scene' in the script.
"""

import os
import re
import base64
import yaml
from pathlib import Path
from PIL import Image
import io
BASE_DIR = Path(__file__).parent.parent
POSES_DIR = BASE_DIR / "poses"


def load_config() -> dict:
    with open(BASE_DIR / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_reference_image(pose_name: str) -> bytes:
    """Load a reference character image as bytes."""
    for p in POSES_DIR.iterdir():
        if p.stem.lower() == pose_name.lower() and p.suffix.lower() in [".png", ".jpg", ".jpeg"]:
            return p.read_bytes()
    raise FileNotFoundError(f"Reference pose not found: {pose_name}")


def generate_scene_image(
    scene_prompt: str,
    scene_id: int,
    safe_name: str,
    config: dict | None = None,
) -> str:
    """
    Generate a custom scene image using Gemini Imagen with character reference.

    Args:
        scene_prompt: Description of the scene to generate (e.g. "Sheva di pasar beli ikan")
        scene_id: Scene ID for output filename
        safe_name: Video safe name for organizing output
        config: Config dict

    Returns:
        Path to the generated image file (PNG with transparent-ish bg)
    """
    if config is None:
        config = load_config()

    from google import genai
    from google.genai import types

    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY not set in environment")

    client = genai.Client(api_key=api_key)
    img_cfg = config["gemini_image"]
    char_desc = img_cfg["character_description"]

    # Output dir
    out_dir = BASE_DIR / img_cfg["output_dir"] / safe_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"scene_{scene_id:02d}.png"

    # Build full prompt with character consistency instruction
    full_prompt = (
        f"{char_desc}, {scene_prompt}. "
        f"Keep the character design consistent: same outfit, same hair color and style, "
        f"same goth-alt aesthetic. Anime illustration style, clean lines, vibrant colors."
    )

    print(f"[SceneGen] Generating scene {scene_id}: '{scene_prompt[:60]}...'")
    print(f"[SceneGen] Full prompt: {full_prompt[:100]}...")

    # Load reference image for style consistency
    ref_pose = img_cfg.get("reference_pose", "sheva_waving and say hi")
    try:
        ref_bytes = _load_reference_image(ref_pose)

        # Use Gemini Flash Image for style-referenced generation
        response = client.models.generate_content(
            model="gemini-2.5-flash-image",
            contents=[
                types.Part.from_bytes(
                    data=ref_bytes,
                    mime_type="image/png",
                ),
                types.Part.from_text(
                    text=(
                        f"Using this character as strict style reference, generate an anime illustration: "
                        f"{full_prompt}. "
                        f"Keep exact same art style, character design, outfit, and color palette as the reference image."
                    )
                ),
            ],
            config=types.GenerateContentConfig(
                response_modalities=["IMAGE", "TEXT"],
            ),
        )

        # Extract image from response
        img_data = None
        for part in response.candidates[0].content.parts:
            if part.inline_data is not None:
                img_data = part.inline_data.data
                break

        if img_data is None:
            raise ValueError("No image data in Gemini response -- model may not support image output")

        # Save image
        img = Image.open(io.BytesIO(img_data))
        img.save(str(out_path), "PNG")
        print(f"[SceneGen] Scene {scene_id} saved: {out_path} ({img.size})")
        return str(out_path)

    except Exception as e:
        err_str = str(e)
        if "429" in err_str or "RESOURCE_EXHAUSTED" in err_str or "quota" in err_str.lower():
            print(f"[SceneGen] Gemini quota exhausted -- switching to OpenAI gpt-image-1 fallback...")
            return _generate_openai_image(full_prompt, scene_id, out_path)
        print(f"[SceneGen] WARNING: Gemini img2img failed ({e}), trying Gemini text-to-image fallback...")
        try:
            return _generate_text_to_image(client, full_prompt, scene_id, out_path, img_cfg)
        except Exception as e2:
            if "429" in str(e2) or "RESOURCE_EXHAUSTED" in str(e2):
                print(f"[SceneGen] Gemini t2i also quota-limited -- switching to OpenAI gpt-image-1...")
                return _generate_openai_image(full_prompt, scene_id, out_path)
            raise


def _generate_text_to_image(client, prompt: str, scene_id: int, out_path: Path, img_cfg: dict) -> str:
    """Fallback: text-to-image via Gemini Flash."""
    from google.genai import types

    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=prompt,
        config=types.GenerateContentConfig(
            response_modalities=["IMAGE", "TEXT"],
        ),
    )

    img_data = None
    for part in response.candidates[0].content.parts:
        if part.inline_data is not None:
            img_data = part.inline_data.data
            break

    if img_data is None:
        raise ValueError(f"Gemini returned no image for scene {scene_id}")

    img = Image.open(io.BytesIO(img_data))
    img.save(str(out_path), "PNG")
    print(f"[SceneGen] Text-to-image scene {scene_id} saved: {out_path} ({img.size})")
    return str(out_path)


def _generate_openai_image(prompt: str, scene_id: int, out_path: Path) -> str:
    """
    Fallback: generate scene image using OpenAI gpt-image-1.
    Uses the existing OPENAI_API_KEY from environment.
    """
    from openai import OpenAI
    import base64

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set -- cannot use OpenAI fallback")

    client = OpenAI(api_key=api_key)
    print(f"[SceneGen] OpenAI gpt-image-1: generating scene {scene_id}...")

    # Enhance prompt for character consistency
    openai_prompt = (
        f"Anime illustration: {prompt}. "
        f"Character: girl with short dark purple hair with blue highlights, fair skin, "
        f"black hoodie with tribal print, black cargo pants with chain, black combat boots, "
        f"ear piercings, cross necklace, goth-alt aesthetic. "
        f"Clean anime art style, vibrant colors, high quality."
    )

    response = client.images.generate(
        model="chatgpt-image-latest",
        prompt=openai_prompt,
        n=1,
        size="1024x1024",
        quality="medium",
    )

    # chatgpt-image-latest returns base64
    img_b64 = response.data[0].b64_json
    img_data = base64.b64decode(img_b64)
    img = Image.open(io.BytesIO(img_data))
    img.save(str(out_path), "PNG")
    print(f"[SceneGen] chatgpt-image scene {scene_id} saved: {out_path} ({img.size})")
    return str(out_path)



def process_generated_scenes(script: dict, safe_name: str, config: dict | None = None) -> dict:
    """
    Process all scenes in script that have 'generated_scene' field.
    Generates images and updates script with 'generated_image_path'.

    Returns updated script dict.
    """
    if config is None:
        config = load_config()

    generated_count = 0
    max_gen = config["llm"].get("max_generated_scenes", 3)

    for scene in script["scenes"]:
        if not scene.get("generated_scene"):
            continue
        if generated_count >= max_gen:
            print(f"[SceneGen] Max generated scenes ({max_gen}) reached, skipping remaining")
            break

        try:
            img_path = generate_scene_image(
                scene_prompt=scene["generated_scene"],
                scene_id=scene["id"],
                safe_name=safe_name,
                config=config,
            )
            scene["generated_image_path"] = img_path
            generated_count += 1
        except Exception as e:
            print(f"[SceneGen] ERROR generating scene {scene['id']}: {e}")
            scene["generated_image_path"] = None

    return script


if __name__ == "__main__":
    import sys
    import json
    from dotenv import load_dotenv

    load_dotenv(BASE_DIR / ".env")
    prompt = " ".join(sys.argv[1:]) or "Sheva di perpustakaan kampus membaca buku dengan ekspresi serius"
    cfg = load_config()
    path = generate_scene_image(prompt, scene_id=99, safe_name="test", config=cfg)
    print(f"Generated: {path}")
