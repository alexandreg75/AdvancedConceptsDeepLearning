from __future__ import annotations

import os
from pathlib import Path
from PIL import Image

from pipeline_utils import (
    DEFAULT_MODEL_ID,
    load_text2img,
    get_device,
    make_generator,
    to_img2img,
)


TP2_DIR = Path(__file__).resolve().parent
OUT_DIR = TP2_DIR / "outputs"
IN_DIR = TP2_DIR / "inputs"


def save(img: Image.Image, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(path))


def run_baseline() -> None:
    model_id = DEFAULT_MODEL_ID
    scheduler_name = "EulerA"
    seed = 42
    steps = 30
    guidance = 7.5

    prompt = (
        "ultra-realistic product photo of a minimalist white sneaker on a neutral studio background, "
        "studio lighting, soft shadow, very sharp, high-end e-commerce style"
    )
    negative = "text, watermark, logo, low quality, blurry, deformed"

    pipe = load_text2img(model_id, scheduler_name)
    device = get_device()
    g = make_generator(seed, device)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=512,
        width=512,
        generator=g,
    )

    img = out.images[0]
    save(img, OUT_DIR / "baseline.png")
    print("OK saved TP2/outputs/baseline.png")
    print(
        "CONFIG:",
        {"model_id": model_id, "scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance},
    )


def run_text2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42

    # Prompt unique e-commerce (identique pour tous les runs)
    prompt = (
        "ultra-realistic product photo of a minimalist white sneaker on a neutral studio background, "
        "studio lighting, soft shadow, very sharp, high-end e-commerce style"
    )
    negative = "text, watermark, logo, low quality, blurry, deformed"

    plan = [
        # name, scheduler, steps, guidance
        ("run01_baseline", "EulerA", 30, 7.5),
        ("run02_steps15", "EulerA", 15, 7.5),
        ("run03_steps50", "EulerA", 50, 7.5),
        ("run04_guid4", "EulerA", 30, 4.0),
        ("run05_guid12", "EulerA", 30, 12.0),
        ("run06_ddim", "DDIM", 30, 7.5),
    ]

    for name, scheduler_name, steps, guidance in plan:
        pipe = load_text2img(model_id, scheduler_name)
        device = get_device()
        g = make_generator(seed, device)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512,
            generator=g,
        )

        img = out.images[0]
        save(img, OUT_DIR / f"t2i_{name}.png")
        print("T2I", name, {"scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance})

    print("T2I done. Files in TP2/outputs: t2i_run01...t2i_run06")


def run_img2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42
    scheduler_name = "EulerA"
    steps = 30
    guidance = 7.5

    # Image source produit (à fournir)
    init_path = IN_DIR / "my_product.jpg"
    if not init_path.is_file():
        raise FileNotFoundError(f"Missing init image: {init_path}")

    # Prompt e-commerce : on garde l'identité produit, on améliore la photo
    prompt = (
        "ultra-realistic high-end e-commerce product photo, clean studio lighting, "
        "neutral background, soft shadow, very sharp, premium look"
    )
    negative = "text, watermark, logo, low quality, blurry, deformed"

    strengths = [
        ("run07_strength035", 0.35),
        ("run08_strength060", 0.60),
        ("run09_strength085", 0.85),
    ]

    pipe_t2i = load_text2img(model_id, scheduler_name)
    pipe_i2i = to_img2img(pipe_t2i)

    device = get_device()
    init_image = Image.open(init_path).convert("RGB")

    # Important : pour être strictement reproductible, on recrée un generator par run
    for name, strength in strengths:
        g = make_generator(seed, device)

        out = pipe_i2i(
            prompt=prompt,
            image=init_image,
            strength=strength,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            generator=g,
        )
        img = out.images[0]
        save(img, OUT_DIR / f"i2i_{name}.png")
        print(
            "I2I",
            name,
            {"scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance, "strength": strength},
        )

    # Optionnel : sauvegarde une copie légère de l'input pour comparer (si c'est OK niveau taille)
    # Sinon, tu mets juste une capture dans le rapport.
    print(f"I2I done. Source used: {init_path}")


def main() -> None:
    # Tu peux commenter/décommenter selon l'exo que tu exécutes
    run_baseline()
    run_text2img_experiments()
    run_img2img_experiments()


if __name__ == "__main__":
    main()
