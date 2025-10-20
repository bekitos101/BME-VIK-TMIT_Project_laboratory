# BME-VIK-TMIT_Project_laboratory

## Project Description
This project adapts the cutting-edge Thought2Text framework, which translates EEG brain signals into natural language text using a three-stage fine-tuning process. Our primary goal is to investigate the impact of replacing the original LLM (Mistral-7B-Instruct) with the DeepSeek-7B base model.

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
