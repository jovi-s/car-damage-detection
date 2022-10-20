# Car Damage Recognition

The aim of this demo is to create an AI based on deep learning where the algorithm automatically detect a vehicle's body and analyzes the extent of the damage.

## Description

Car damage recognition ML algorithms can be retrained based on the customer’s data set and delivered on-premises or as SaaS. This way, the API adapts to the specific business needs and serves as a well-integrated solution to be used in the claims automation process.

The solution speeds up data processing, saving the company’s spendings on human resources, defending form fraud (in 80% and more), and boosting the process of image data analysis in times. The system is used on sight and guides a user on actions to meet photo requirements. Deploying Car Damage Recognition, businesses replace a human-operated time-consuming process of claims proceeding and approval with machine learning algorithms and analytical systems.


### Evaluation

Intersection over Union (IOU) and mean Average Precision (mAP) are used during the validation phase as evaluation metrics. The scores for the last training epoch on segmentations are:

<center>

| Average Precision, Recall | IOU                   | Area   | Max Detections | Score |
|---------------------------|-----------------------|--------|----------------|-------|
| Average Precision         | (AP) @ IoU=0.50:0.95  | all    | 100            | 0.116 |
| Average Precision         | (AP) @ IoU=0.50       | all    | 100            | 0.272 |
| Average Precision         | (AP) @ IoU=0.75       | all    | 100            | 0.121 |
| Average Precision         | (AP) @ IoU=0.50:0.95  | small  | 100            | 0.000 |
| Average Precision         | (AP) @ IoU=0.50:0.95  | medium | 100            | 0.092 |
| Average Precision         | (AP) @ IoU=0.50:0.95  | large  | 100            | 0.184 |
| Average Recall            | (AR) @ IoU=0.50:0.95  | all    | 1              | 0.083 |
| Average Recall            | (AR) @ IoU=0.50:0.95  | all    | 10             | 0.250 |
| Average Recall            | (AR) @ IoU=0.50:0.95  | all    | 100            | 0.308 |
| Average Recall            | (AR) @ IoU=0.50:0.95  | small  | 100            | 0.000 |
| Average Recall            | (AR) @ IoU=0.50:0.95  | medium | 100            | 0.215 |
| Average Recall            | (AR) @ IoU=0.50:0.95  | large  | 100            | 0.460 |

</center>

## Getting Started

This project was built on Python version 3.8.10.

### Installation

1. Install `pycocotools` locally to execute exploration and training notebooks
2. git clone/ download zip
3. cd into dir
4.
```
pip install -r requirements.txt
```

### Executing program

- Dashboard
```
python main.py
```
- Notebooks
    - EDA: `notebooks/0_dataset_exploration.ipynb`
    - MASK RCNN Training: `notebooks/1_mask_rcnn_pytorch_transfer_learn.ipynb`
    - Inference: `notebooks/2_inference.ipynb`
    - Model Explainability: `notebooks/3_model_explainability.ipynb`

- AWS Deployment using SageMaker Studio
    1. Use SageMaker Studio Image Build CLI
    2. Push container to ECR
    3. Use ECS/ EC2 to serve container

## Authors

[Jovinder Singh](https://github.com/jovi-s/)

## Acknowledgments

- [Car Damage Detection Dataset](https://www.kaggle.com/datasets/lplenka/coco-car-damage-detection-dataset)
