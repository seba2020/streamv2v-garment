"""Minimal Phase 1 smoke test: garment IP-Adapter + clothing-mask blending
on a single input image. No camera, no websockets, no frontend -- just
"does the pipeline run, and does the garment show up".

Usage:
    python tools/test_garment_phase1.py \
        --person path/to/person.jpg \
        --garment path/to/garment.jpg \
        --output out.png
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from PIL import Image

from utils.wrapper import StreamV2VWrapper


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--person", required=True, help="Path to the source person/frame image")
    parser.add_argument("--garment", required=True, help="Path to the reference garment image")
    parser.add_argument("--output", default="phase1_output.png")
    parser.add_argument("--model", default="Jiali/stable-diffusion-1.5")
    parser.add_argument("--prompt", default="a person wearing the reference garment")
    parser.add_argument("--ip-adapter-scale", type=float, default=0.6)
    parser.add_argument("--acceleration", default="xformers", choices=["none", "xformers"])
    parser.add_argument(
        "--num-frames",
        type=int,
        default=6,
        help="Feed the same image N times (StreamV2V's temporal feature bank settles after a few frames)",
    )
    args = parser.parse_args()

    print("Loading model (first run also downloads IP-Adapter + segmentation weights)...")
    wrapper = StreamV2VWrapper(
        model_id_or_path=args.model,
        t_index_list=[30, 35, 40, 45],
        mode="img2img",
        width=512,
        height=512,
        acceleration=args.acceleration,
        use_cached_attn=True,
        use_ip_adapter=True,
        ip_adapter_scale=args.ip_adapter_scale,
        use_clothes_mask=True,
        seed=1,
    )
    wrapper.prepare(prompt=args.prompt, num_inference_steps=50, guidance_scale=1.0)
    wrapper.set_ip_adapter_image(Image.open(args.garment).convert("RGB"))

    person_image = Image.open(args.person).convert("RGB").resize((512, 512))

    output = None
    for i in range(args.num_frames):
        output = wrapper(image=person_image, prompt=args.prompt)
        print(f"frame {i + 1}/{args.num_frames} done")

    output.save(args.output)
    print(f"Saved: {args.output}")


if __name__ == "__main__":
    main()
