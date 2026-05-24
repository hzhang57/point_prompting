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
gpu_count = torch.cuda.device_count()
cuda_capability = torch.cuda.get_device_capability(0) if gpu_count else (0, 0)
torch_dtype = torch.bfloat16 if cuda_capability[0] >= 8 else torch.float16
load_kwargs = {
    "torch_dtype": torch_dtype,
}

if gpu_count >= 2:
    load_kwargs.update(
        device_map="balanced",
        max_memory={i: "14GiB" for i in range(gpu_count)},
    )

pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
    model_id,
    **load_kwargs,
)

pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
if gpu_count < 2:
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

generator = torch.Generator(device="cpu").manual_seed(42)

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
