# CIS 680 FInal Project LoRA-SAM

## 1. Code Usage
### 1-1. Data preprocessing
Create a folder `data` and place `sa1b` under it, then run `python3 lora_sam/dataset.py`

### 1-2. Training 
After preprocessing, run `python3 traning.py`

### 1-3. Inference 
Run `python3 inference.py`, change line 21-27 to test out different prompts

## 2. Results
### 2-1. Loss Curve
![](result_plots/loss_curve.png?raw=true)

### 2-2. Bounding Box Prompt
![](result_plots/building_bbox.png?raw=true)

![](result_plots/stairs_bbox.png?raw=true)

### 2-3. Point Prompt
![](result_plots/building_dots.png?raw=true)

![](result_plots/stairs_dots.png?raw=true)