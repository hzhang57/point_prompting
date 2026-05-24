import torch
from diffusers import CogVideoXVideoToVideoPipeline, CogVideoXDPMScheduler
from diffusers.utils import load_video, export_to_video

model_id = "THUDM/CogVideoX-5b"

pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
)

pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()

input_video = load_video("input.mp4")

prompt = "A cinematic video, same motion, cyberpunk city style"
negative_prompt = "low quality, blurry, distorted, artifacts"

generator = torch.Generator(device="cuda").manual_seed(42)

video = pipe(
    video=input_video,
    prompt=prompt,
    negative_prompt=negative_prompt,
    strength=0.55,          # SDEdit强度：越大改动越大
    guidance_scale=6.0,
    num_inference_steps=50,
    generator=generator,
).frames[0]

export_to_video(video, "/kaggle/working/point_prompting/input.mp4", fps=8)