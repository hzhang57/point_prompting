import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import CogVideoXVideoToVideoPipeline, CogVideoXDPMScheduler
from diffusers.utils import load_video, export_to_video
from PIL import Image

model_id = "THUDM/CogVideoX-5b"
max_frames = 8
input_size = (720, 480)

pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

raw_video = load_video("/kaggle/working/point_prompting/input.mp4")
frame_stride = max(len(raw_video) // max_frames, 1)
input_video = raw_video[::frame_stride][:max_frames]
input_video = [
    frame.resize(input_size, Image.Resampling.LANCZOS)
    for frame in input_video
]

prompt = "A cinematic video, same motion, cyberpunk city style"
negative_prompt = "low quality, blurry, distorted, artifacts"

generator = torch.Generator(device="cuda").manual_seed(42)

video = pipe(
    video=input_video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    strength=0.55,          # SDEdit强度：越大改动越大
    guidance_scale=6.0,
    num_inference_steps=25,
    generator=generator,
).frames[0]

export_to_video(video, "/kaggle/working/point_prompting/output.mp4", fps=8)
