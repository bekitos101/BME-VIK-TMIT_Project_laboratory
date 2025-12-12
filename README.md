# BME-VIK-TMIT_Project_laboratory

## Project Description
This project reproduces and simplifies the core ideas of the Thought2Text (2024) paper, which generates natural-language text directly from EEG activity.
The full system is implemented in three stages:

### Stage 1 – EEG Encoder Training:
A CNN/ChannelNet-based EEG encoder learns to map EEG spectrograms to the CLIP image-embedding space.

### Stage 2 – Projector Training:
A projector maps CLIP embeddings into the hidden-state dimension of a language model (DeepSeek-Coder-1.3B-Instruct).
The LLM is trained to predict captions conditioned on this inserted EEG/CLIP embedding.

### Stage 3 – End-to-End EEG→LLM Caption Generation:
The encoder and projector are combined, and the LLM is lightly fine-tuned via LoRA by introducing an <eeg> token whose embedding slot is filled with the projected EEG vector.

The goal is to test how far a smaller LLM (DeepSeek-1.3B) can reproduce the performance trends of the original Mistral-7B model used in the Thought2Text paper.

## Milestone 1: Data Structure Exploration & Indexing
--------------------------------------------------
This phase successfully prepared the raw data for the subsequent training stages.

### 1\. Data Source and Download

*   **Source:** The project utilizes a publicly available **EEG-to-Text dataset** collected for six subjects, similar to the one used in the Thought2Text paper (CVPR2017/ZuCo-related data linking image stimuli with EEG and text captions).
    
*   **Download:** The dataset is available on this link  [Google Drive – EEG–Vision–Caption Dataset](https://drive.google.com/drive/folders/1XqV6MMl28iYXkQBMEFHfEXllGmCbqpOu)
        

### 2\. Data Exploration Summary

The provided Python notebook performed exploratory data analysis to build and verify a master index:

| Metric                     | Value   | Notes                                                                 |
|-----------------------------|---------|-----------------------------------------------------------------------|
| Total EEG Spectrogram Windows | 11,965  | Final number of data points.                                          |
| ImageNet Classes            | 40      | Number of object categories in the visual stimuli.                    |
| Unique Trials (base_id)     | 1,996   | Total distinct (Image + Caption) trials.                              |
| Alignment Quality           | 100%    | All 11,965 spectrograms are perfectly aligned with a sketch image and a ground-truth caption. |


### 3\. Final Output (Data Preparation)

The final output of Milestone 1 is the **master index**, which will serve as the map for all subsequent data loading.


## Instructions on How to Run the Solution

The solution is provided as a **Google Colab Notebook**: `data processing and preparation.ipynb`.

### Prerequisites

### 1. Google Drive
Ensure the **EEG-to-Text dataset** is uploaded to your Google Drive.

### 2. Dataset Structure
The dataset should be organized within a root folder (for example, `capstone`) with the following structure:
```
capstone/
├── block/
├── images/
```
---

### Execution Steps

#### 1. Open Notebook
Open **`thought2text_colab_M1.ipynb`** in **Google Colab**.

#### 2. Run Setup
Execute the initial setup cells to prepare the environment:

#### a. Mount your Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```
#### b. Define the Base Directory Path

Make sure the path matches the location of your project folder:
```python
from pathlib import Path
BASE_DIR = Path('/content/drive/MyDrive/capstone')
```
#### c.Run Exploration
Execute the subsequent cells to generate and save the final index used for training and evaluation.


## Milestone 2: Baseline implementation and and Model Variants: 
Milestone 2 focused on implementing the full EEG→Text training pipeline and experimenting with different model variants at each stage. The goal was to evaluate how architectural changes and training strategies affect alignment quality and caption generation.

Several encoder architectures were explored to extract meaningful embeddings from EEG spectrograms:

### V1: Simple CNN baseline

Two convolutional layers + linear projection

Served as an initial reference point

### V2: ChannelNet-inspired encoder

Temporal dilation blocks

Spatial convolution blocks

Multiple residual layers

### CLIP → LLM Projector

A projector module was implemented to map CLIP embedding (512-dim) to  LLM hidden state (2048-dim)

### LLM Prompting and Embedding Integration
DeepSeek uses a custom chat template, so prompting had to be rebuilt manually. the implemented components are:
1. Chat-template generator for prefix prompts
2. Manual token-embedding concatenator:

### End-to-End Training Loop
The full pipeline was trained in several configurations:
1.Encoder trained first to align EEG embeddings with CLIP vectors
2. Projector trained to map CLIP→LLM embedding space
3.to-end EEG→Caption training with either:Frozen LLM, or LoRA lightweight finetuning.

## How to Run the Full System

This project is fully ipynb-friendlt, but GPU acceleration is required.
For training all three stages, you should use:
A100 or L4 GPUs (Colab Pro / Pro+ recommended)
Make sure your dataset paths in the notebooks match your Google Drive folder.All notebooks automatically install their own dependencies.

**Stage 1 — EEG Encoder Training** 
Notebook:
```
EEG_to_text_stage1_encoder.ipynb
```

**Stage 2 — Projector Training (CLIP → LLM)g** 
Notebook:
```
EEG_to_text_stage2_projector.ipynb

```

**Stage 3 — EEG → Caption Generation** 
Notebook:
```
EEG_to_text_stage3_Encoder_Projector_LLM.ipynb

```
## Expected results:
Semantically meaningful captions
Moderate BLEU/ROUGE scores
Strong BERTScore / SBERT similarity scores

## Important Notes
-Use the same BASE directory path across all notebooks so saved weights load correctly.
-The three notebooks must be executed in order:
Stage 1 → Stage 2 → Stage 3
-Training without a strong GPU may fail due to memory limits.

## Acknowledgment
This project builds on the excellent work of Mishra et al. in the paper:
_A. Mishra, S. M. Shurid, et al.
Thought2Text: Text Generation from EEG Signal using Large Language Models (LLMs).
arXiv preprint arXiv:2410.07507, 2024_

Their framework provided the scientific foundation for the three-stage training pipeline used in this project.Our work adapts and extends their approach using the DeepSeek family of language models.
