import inspect
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
os.environ.setdefault("PYTORCH_ALLOC_CONF", "expandable_segments:True")

import torch
from diffusers import CogVideoXVideoToVideoPipeline, CogVideoXDPMScheduler
from diffusers.utils import load_video, export_to_video
from PIL import Image


def log(message):
    print(f"[cogvideo] {message}", flush=True)


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

log(f"CUDA devices: {gpu_count}")
log(f"Using dtype: {torch_dtype}")
if gpu_count >= 2:
    log(f"Using balanced device_map with max_memory: {load_kwargs['max_memory']}")
else:
    log("Using model CPU offload because fewer than 2 GPUs are available")

log(f"Loading pipeline: {model_id}")
pipe = CogVideoXVideoToVideoPipeline.from_pretrained(
    model_id,
    **load_kwargs,
)
log("Pipeline loaded")

pipe.scheduler = CogVideoXDPMScheduler.from_config(pipe.scheduler.config)
if gpu_count < 2:
    pipe.enable_model_cpu_offload()
pipe.vae.enable_tiling()
pipe.vae.enable_slicing()
log("Scheduler and VAE memory settings configured")
if hasattr(pipe, "hf_device_map") and pipe.hf_device_map:
    log(f"Device map: {pipe.hf_device_map}")

log("Loading input video")
raw_video = load_video("/kaggle/working/point_prompting/input.mp4")
frame_stride = max(len(raw_video) // max_frames, 1)
input_video = raw_video[::frame_stride][:max_frames]
log(f"Loaded {len(raw_video)} frames; sampling {len(input_video)} frames with stride {frame_stride}")
input_video = [
    frame.resize(input_size, Image.Resampling.LANCZOS)
    for frame in input_video
]
log(f"Resized sampled frames to {input_size[0]}x{input_size[1]}")

prompt = "A cinematic video, same motion, cyberpunk city style"
negative_prompt = "low quality, blurry, distorted, artifacts"

generator = torch.Generator(device="cpu").manual_seed(42)
num_inference_steps = 25

def log_step(pipe, step_index, timestep, callback_kwargs):
    log(f"Inference step {step_index + 1}/{num_inference_steps}")
    return callback_kwargs


pipe_kwargs = {
    "video": input_video,
    "prompt": prompt,
    "negative_prompt": negative_prompt,
    "strength": 0.55,          # SDEdit强度：越大改动越大
    "guidance_scale": 6.0,
    "num_inference_steps": num_inference_steps,
    "generator": generator,
}

if "callback_on_step_end" in inspect.signature(pipe.__call__).parameters:
    pipe_kwargs["callback_on_step_end"] = log_step
else:
    log("Step callback is not supported by this Diffusers version")

log("Starting video generation")
video = pipe(
    **pipe_kwargs,
).frames[0]
log(f"Generation finished with {len(video)} output frames")

log("Exporting output video")
export_to_video(video, "/kaggle/working/point_prompting/output.mp4", fps=8)
log("Saved output video to /kaggle/working/point_prompting/output.mp4")
