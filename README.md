# CIS 680 FInal Project LoRA-SAM

## Code Usage
### Data preprocessing
Create a folder `data` and place `sa1b` under it, then run `python3 lora_sam/dataset.py`. 

### Training 
After preprocessing, run `python3 traning.py`

### Inference 
Run `python3 inference.py`, change line 21-27 to test out different prompts. 

## Results
### Loss Curve
![](result_plots/loss_curve.png?raw=true)

### Bounding Box Prompt
![](result_plots/building_bbox.png?raw=true)

![](result_plots/stairs_bbox.png?raw=true)

### Point Prompt
![](result_plots/building_dots.png?raw=true)

![](result_plots/stairs_dots.png?raw=true)