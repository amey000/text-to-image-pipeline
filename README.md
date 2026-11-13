# Text-to-Image Pipeline (LISA)

Fine-tunes **Stable Diffusion v1.5** on **COCO 2017** captions with **LoRA**, then generates images through an LLM-augmented pipeline:

1. **Prompt expansion:** GPT-4 rewrites a short prompt (e.g. *"a bowl on a table"*) into a richer, spatially explicit description.
2. **Generation:** Stable Diffusion + the trained LoRA (DPM-Solver, 30 steps, guidance 8.0).
3. **Evaluation:** a **CLIP score** (`openai/clip-vit-base-patch32`) for text–image alignment, plus a **GPT-3.5 "judge"** that critiques whether the spatial details were realized.

Everything runs in **Google Colab** with a GPU, using Google Drive for storage. An interactive ipywidgets dashboard ties it together.

## Repository contents

| File | Purpose |
| --- | --- |
| `Text_to_Image_pipeline.ipynb` | Main Colab notebook: data prep → training → dashboard. |
| `Text_to_Image_Pipeline.py` | Colab export of the notebook (contains `!` shell magics, so run it in Colab, not plain Python). |
| `train_text_to_image_lora.py` | Hugging Face Diffusers LoRA training script (Apache 2.0), launched by the pipeline. |
| `requirements.txt` | Python dependencies (`pip install -r requirements.txt`). |
| `SETUP_NOTES.md` | Extended setup notes and troubleshooting. |

## Prerequisites

- Google Colab with a **GPU runtime** (T4 is enough; V100/A100 are faster).
- **OpenAI API key** for GPT-4 expansion and GPT-3.5 judging.
- **Hugging Face access token** for downloading Stable Diffusion v1.5.
- **COCO 2017 dataset.** It is **not included** in this repo (~26 GB). Download it from [cocodataset.org](https://cocodataset.org/#download):
  - *2017 Train images*
  - *2017 Train/Val annotations* (needs `captions_train2017.json`)

Upload it to Google Drive with this layout, or edit `DRIVE_IMAGES_DIR` / `DRIVE_ANNOTATIONS` at the top of the pipeline:

```
MyDrive/
└── coco2017/
    ├── train2017/                   # .jpg images
    └── annotations/
        └── captions_train2017.json
```

## How to run

Open `Text_to_Image_pipeline.ipynb` in Colab, set **Runtime → Change runtime type → GPU**, upload `train_text_to_image_lora.py` to the Colab working directory, then run the three stages in order.

**Stage 1: Data preparation** (`prepare_colab_training_subset`)
Mounts Drive, copies a 5,000-image subset to Colab's local disk for speed, and writes the `metadata.jsonl` (file name + caption) that the trainer expects.

**Stage 2: LoRA training** (`launch_lora_training`)
Installs `diffusers`, `transformers`, `accelerate`, `peft` and `bitsandbytes`, then runs the training below. Outside Colab, install everything up front with `pip install -r requirements.txt`.

| Setting | Value |
| --- | --- |
| Base model | `stable-diffusion-v1-5/stable-diffusion-v1-5` |
| Resolution | 512 |
| Batch size / grad. accumulation | 1 / 4 |
| Steps | 500 (checkpoint every 250) |
| Learning rate | 1e-4, cosine schedule |
| Precision | fp16 |

Weights are saved to `MyDrive/lisa_model_output`. Expect roughly 30–60 minutes on a T4.

**Stage 3: Dashboard** (`create_dashboard`)
Enter your OpenAI key, Hugging Face token and LoRA path (defaults to the Stage 2 output), type a prompt, and generate. The dashboard shows the expanded prompt, the image, the CLIP score (above ~25 indicates strong alignment), and the GPT judge's critique. A toggle lets you compare against the base model without the LoRA.

## Keeping your keys safe

Values typed into notebook widgets are **saved inside the `.ipynb` file**. Before committing or sharing the notebook:

- Prefer **Colab Secrets** (🔑 icon in the sidebar) named `OPENAI_API_KEY` and `HF_TOKEN`, or environment variables, over typing keys into widgets.
- Otherwise, clear outputs and widget state (**Edit → Clear all outputs**) before saving.

## Troubleshooting

- **"Training data not found"**: Stage 1 hasn't been run, or the COCO files aren't at the configured Drive paths.
- **"T4 GPU not detected" / CUDA errors**: the Colab runtime isn't set to GPU.
- **Why not `runwayml/stable-diffusion-v1-5`?** That repo has been taken down from Hugging Face. This project uses the mirror `stable-diffusion-v1-5/stable-diffusion-v1-5`, which has the same weights. LoRAs trained on either one are interchangeable.
- **OpenAI errors**: check your API key and credit balance.

## Credits

- LoRA training script: [Hugging Face Diffusers](https://github.com/huggingface/diffusers/tree/main/examples/text_to_image) (Apache 2.0).
- Dataset: [COCO 2017](https://cocodataset.org) (CC BY 4.0 annotations).
- Base model: Stable Diffusion v1.5 (CreativeML OpenRAIL-M).
