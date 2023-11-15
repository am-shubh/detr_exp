# Detr with SAHI

### Experiments Information
I did two experiments and code was developed in the same way.
- Exp 1 - without SAHI: Developed the scripts and dockerised it. Tested end-to-end flow for Detr training, evaluation, prediction.
- Exp 2 - with SAHI: Modified the above code to include SAHI for training and prediction.

**NOTE** : Both experiment's mount direcotry are shared via [google drive](https://drive.google.com/drive/folders/1bfVZLKjhbJypeUB90jSapHpm8rfvs9wc?usp=sharing)

### Mount Volume structure
The mount directory contains following files/folders for entire pipeline to work.
- Config: It stores dynamic configurations for training and prediction pipeline.
- Logs: It stores logs for scripts as well as pytorch lightning logs are also dumped in same folder after trainig.
- Dataset: It contains assignment dataset in coco format. For SAHI training, the dataset is prepared and stored in same directory.
- Predictions: It stores all prediction visualisation on test dataset.
- Weights: It contains two sub-folders:
  - Pretrained: To store previous training weights or open source weights.
  - Trained: To store current training model files.

### Command to run experiments
1. You can create a docker image and then mount above directory while running container. Assuming you have cloned this repo. Run following commands:
   ```
   cd detr_exp/
   docker build -t detr_exp:dev
   docker run -v ....your mount_volume....:/exp --gpus all -it detr_exp:dev /bin/bash
   ```
2. If you want to run this in local environment then use *requirements.txt* file to update the necessary packages. I have used python version 3.10.9

### Issues
1. The dataset, which was shared, is already processed meaning it contains a lot of augmented image. Augmentation techniques like resizing without maintaining aspect ratio and salt & pepper noise should not have been used.
2. There is skewness in data meaning not all classes are in similar proportion, this can lead to some biasness.


### Exp 1 WITHOUT SAHI - Summary
- With the params provided in the config.json, here are the evaluation results:
>INFO - IoU metric: bbox
INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.524
INFO -  Average Precision  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.671
INFO -  Average Precision  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.606
INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.364
INFO -  Average Precision  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.610
INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.607
INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.711
INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.711
INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = -1.000
INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.584
INFO -  Average Recall     (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.769

### Exp 2 WITH SAHI - Summary
- There is no direct implementation of SAHI functions for training. So as of now, with little explorations, I believe there are two ways to achieve the same:
  - Pre-process the Train and Valid dataset to slice the data and train the model. Then use full images from the test set to check model performance. This is the easiest and fastes to do and hence I have implemnted this.
  - Use Slicing function as a part of data loader, so that it can be coupled with other augmentation techniques.

- With the params provided in the config.json, here are the evaluation results:
> *to_be_update*


### TODO
1. Train model with SAHI pre-processing and update the evaluation results and predictions.


### References
1. https://github.com/obss/sahi
2. https://colab.research.google.com/drive/17az1rlOZPZJzuKEbkR9v7cDY0mth1gsb?usp=sharing