import os, math, random, torch
from PIL import Image
from datasets import load_dataset, Image as DSImage
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from huggingface_hub import login
from diffusers import StableDiffusionPipeline, DDPMScheduler
from diffusers.models.attention_processor import LoRAAttnProcessor

# --- config ---
MODEL="runwayml/stable-diffusion-v1-5"
DATASET="PeterBrendan/AdImageNet"
OUT="./ads-lora-sd15"; RES=512; BATCH=1; ACC=8; STEPS=8000
LR=1e-4; LORA_R=16; SAVE_EVERY=1000; SEED=42
random.seed(SEED); torch.manual_seed(SEED)

# --- HF auth (optional if you've run `huggingface-cli login`) ---
tok=os.environ.get("HUGGINGFACE_TOKEN")
if tok: login(token=tok)

# --- data helpers ---
CAPTION_COL="text"; IMAGE_COL="image"
def load_ads():
    ds=load_dataset(DATASET, split="train")
    if IMAGE_COL not in ds.column_names:
        if "file_name" in ds.column_names:
            ds=ds.cast_column("file_name", DSImage()); ds=ds.rename_column("file_name", IMAGE_COL)
        else: raise ValueError("No image column found.")
    if CAPTION_COL not in ds.column_names:
        ds=ds.add_column(CAPTION_COL, ["ad"]*len(ds))
    return ds
class AdsSet(Dataset):
    def __init__(s, ds, size=RES):
        s.ds=ds; s.t=transforms.Compose([
            transforms.Resize((size,size), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(), transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])])
    def __len__(s): return len(s.ds)
    def __getitem__(s,i):
        ex=s.ds[i]; img=ex[IMAGE_COL]
        if isinstance(img,str): img=Image.open(img).convert("RGB")
        else: img=img.convert("RGB")
        return {"pixel_values": s.t(img), "caption": ex.get(CAPTION_COL,"ad")}

# --- device/dtype ---
device=torch.device("mps" if torch.backends.mps.is_available() else "cpu")
dtype=torch.float16 if device.type=="mps" else torch.float16
torch.backends.mps.allow_fp16_reduced_precision=True

# --- load data/model ---
train=AdsSet(load_ads()); loader=DataLoader(train,batch_size=BATCH,shuffle=True,num_workers=2,drop_last=True)
pipe = StableDiffusionPipeline.from_pretrained(MODEL, dtype=dtype, safety_checker=None)
pipe.scheduler=DDPMScheduler.from_config(pipe.scheduler.config)
unet=pipe.unet

# --- add LoRA to all attention processors ---
from diffusers.models.attention_processor import (
    AttnProcessor2_0,
    LoRAAttnProcessor,
    LoRAAttnProcessor2_0,
)
from diffusers.loaders import AttnProcsLayers

lora_attn_procs = {}
for name, proc in pipe.unet.attn_processors.items():
    # cross-attn on attn2, self-attn on attn1
    cross_dim = None if name.endswith("attn1.processor") else pipe.unet.config.cross_attention_dim

    # hidden size per block
    if name.startswith("mid_block"):
        h = pipe.unet.config.block_out_channels[-1]
    elif name.startswith("up_blocks"):
        i = int(name.split(".")[1])
        h = list(reversed(pipe.unet.config.block_out_channels))[i]
    elif name.startswith("down_blocks"):
        i = int(name.split(".")[1])
        h = pipe.unet.config.block_out_channels[i]
    else:
        h = pipe.unet.config.block_out_channels[0]

    # choose correct LoRA class and use **positional** args
    if isinstance(proc, AttnProcessor2_0):
        lora_attn_procs[name] = LoRAAttnProcessor2_0(h, cross_dim, LORA_R)
    else:
        lora_attn_procs[name] = LoRAAttnProcessor(h, cross_dim, LORA_R)

pipe.unet.set_attn_processor(lora_attn_procs)

# collect trainable LoRA params cleanly
trainable_loras = AttnProcsLayers(pipe.unet.attn_processors)
opt = torch.optim.AdamW(trainable_loras.parameters(), lr=LR, betas=(0.9,0.999), weight_decay=1e-2)

# Apply them all at once
pipe.unet.set_attn_processor(lora_attn_procs)
unet = pipe.unet  # keep rest of your code the same


# collect LoRA params
lora_params=[p for m in unet.attn_processors.values() for p in m.parameters() if isinstance(m, LoRAAttnProcessor)]
opt=torch.optim.AdamW(lora_params, lr=LR, betas=(0.9,0.999), weight_decay=1e-2)
acc_count=0; global_step=0; opt.zero_grad(set_to_none=True)

# --- helpers ---
def enc_text(prompts):
    tok=pipe.tokenizer(prompts, padding="max_length", truncation=True,
                       max_length=pipe.tokenizer.model_max_length, return_tensors="pt").to(device)
    with torch.no_grad(): enc=pipe.text_encoder(**tok).last_hidden_state
    return enc
def vae_enc(x):
    x=x.to(device, dtype=dtype)
    with torch.no_grad(): return pipe.vae.encode(x).latent_dist.sample()*0.18215

# --- train ---
while global_step<STEPS:
    for b in loader:
        if global_step>=STEPS: break
        lat=vae_enc(b["pixel_values"])
        noise=torch.randn_like(lat)
        t=torch.randint(0, pipe.scheduler.config.num_train_timesteps, (lat.size(0),), device=device).long()
        noisy=pipe.scheduler.add_noise(lat, noise, t)
        pred=unet(noisy, t, encoder_hidden_states=enc_text(b["caption"])).sample
        loss=torch.nn.functional.mse_loss(pred, noise); (loss/ACC).backward(); acc_count+=1
        if acc_count%ACC==0:
            opt.step(); opt.zero_grad(set_to_none=True); global_step+=1; acc_count=0
            if global_step%100==0: print(f"step {global_step}/{STEPS} loss {loss.item():.4f}")
            if global_step%SAVE_EVERY==0 or global_step==STEPS:
                os.makedirs(OUT, exist_ok=True); unet.save_attn_procs(OUT)

# --- quick smoke test ---
pipe.unet.load_attn_procs(OUT); pipe.to(device)
img=pipe("clean product ad, centered item, whitespace, CTA",
         num_inference_steps=25, guidance_scale=6.0).images[0]
img.save(os.path.join(OUT,"sample.png")); print("Saved LoRA + sample to", OUT)
