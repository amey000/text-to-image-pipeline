# LISA: Language-Informed Semantic Adaptation
# Text-to-Image LoRA Training & Reasoning Pipeline

This project provides a complete workflow for fine-tuning Stable Diffusion (v1.5) on the COCO dataset using LoRA, followed by an LLM-augmented inference pipeline that expands user prompts for better spatial and semantic accuracy.

The system is designed to run in a Google Colab environment using Google Drive for storage.

========================================================================
1. PROJECT STRUCTURE
========================================================================

The project consists of two main Python scripts:

1. train_text_to_image_lora.py
   - The core training script provided by Hugging Face Diffusers.
   - Handles the actual finetuning mathematics, gradient application, and model saving.
   - It is called automatically by the pipeline script; you usually do not need to run this manually.

2. Text_to_Image_Pipeline.py
   - The master controller script.
   - Handles Data Preparation (moving images from Drive to local Colab SSD).
   - Triggers the Training process.
   - Contains the Inference logic (LISA) using OpenAI GPT-4 for prompt expansion.
   - Launches an interactive UI Dashboard for generation and evaluation.

========================================================================
2. IMPORTANT: DATASET REQUIREMENTS
========================================================================

**MANUAL DOWNLOAD REQUIRED:** This pipeline **does not** download the COCO dataset automatically. You are responsible for downloading the dataset and uploading it to your Google Drive before running any scripts.

1. Download the **COCO 2017** dataset (specifically the "Train images" and "Train/Val annotations").
2. Upload the files to your Google Drive.
3. Ensure the folder structure matches the configuration below (or update the paths in `Text_to_Image_Pipeline.py`).

**Required Directory Structure:**
/content/drive/MyDrive/
└── coco2017/
    ├── train2017/                     # Folder containing the .jpg images
    └── annotations/
        └── captions_train2017.json    # The COCO captions JSON file

*If your data is located elsewhere, you MUST edit the `DRIVE_IMAGES_DIR` and `DRIVE_ANNOTATIONS` variables at the top of `Text_to_Image_Pipeline.py`.*

========================================================================
3. PREREQUISITES
========================================================================

Hardware:
- NVIDIA GPU (T4, V100, or A100).
- Google Colab environment is highly recommended.

Accounts & Keys:
- **OpenAI API Key**: Required for GPT-4 prompt expansion and GPT-3.5 evaluation.
- **Hugging Face Access Token**: Must have Write permissions to download Stable Diffusion v1.5.

========================================================================
4. INSTALLATION & SETUP
========================================================================

1. Upload both `train_text_to_image_lora.py` and `Text_to_Image_Pipeline.py` to your Colab workspace (root directory).

2. Open `Text_to_Image_Pipeline.py`. This script is designed to be run in stages (or cells if converted to a Notebook).

3. The script automatically installs the required libraries:
   - diffusers, transformers, accelerate, peft, bitsandbytes, openai

========================================================================
5. HOW TO RUN
========================================================================

The `Text_to_Image_Pipeline.py` handles the workflow in three distinct stages:

--- STAGE 1: Data Preparation ---
Run the function `prepare_colab_training_subset()`.
- This mounts Google Drive.
- It copies a subset (default 5000) of images from your manually uploaded folder in Drive to Colab local storage (SSD) for faster training.
- It creates a `metadata.jsonl` file required by the training script.

--- STAGE 2: Training (LoRA) ---
Run the function `launch_lora_training()`.
- This executes `accelerate launch` to start training.
- Configuration: 500 Steps, Batch Size 1, Resolution 512, Mixed Precision FP16.
- Output: The trained LoRA weights will be saved to `/content/drive/MyDrive/lisa_model_output`.
- Duration: Approx 30-60 mins on a T4 GPU.

--- STAGE 3: Inference & Dashboard ---
Run the function `create_dashboard()`.
- This verifies if a GPU is available.
- It loads an interactive UI widget.
- You will need to input:
  1. OpenAI API Key
  2. Hugging Face Token
  3. Path to your LoRA (default is the output path from Stage 2).

========================================================================
6. THE EVALUATION PIPELINE
========================================================================

When generating images via the Dashboard, the system performs the following:

1. **Semantic Expansion**: Uses GPT-4 to convert a simple prompt (e.g., "A bowl on a table") into a rich, spatially aware description.
2. **Generation**: Uses Stable Diffusion + Your Custom LoRA to generate the image.
3. **CLIP Score**: Calculates the quantitative similarity between the text and the generated image.
4. **Reasoning Critique**: Uses GPT-3.5 as a "Judge" to evaluate if the spatial details were correctly realized.

========================================================================
7. TROUBLESHOOTING
========================================================================

- **"Training data not found"**: This confirms you skipped the manual download step. Verify the COCO images are actually present in your Google Drive folder.
- **"CUDA/GPU not found"**: Ensure your runtime type in Colab is set to GPU.
- **"API Connection Error"**: Check your OpenAI credit balance and Hugging Face token permissions.