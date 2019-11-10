# Implementation and Evaluation of Deep Learning Approaches for Real-time 3D Robot Localization in an Augmented Reality Use Case

#### External links
* The manual 6D object pose annotation tool - [http://annotate.photo/](http://annotate.photo?ref=github1)
* [Single Shot Pose Estimation (SSPE)](https://github.com/microsoft/singleshotpose/)
* [Betapose](https://github.com/sjtuytc/betapose)
* The 5 datasets used for training and evaluation - [Google Drive](https://drive.google.com/open?id=1xiS53HLcr5vGQRmiOt7Yotwso9g-ddht)
* The weights for SSPE and Betapose - [Google Drive](https://drive.google.com/open?id=1XQGH31AxFJWjLGV19yYY9HBrvVzqMojH)

#### Speed demo

[![Speed demo](https://img.youtube.com/vi/4mMKnfgYzVU/0.jpg)](https://www.youtube.com/watch?v=4mMKnfgYzVU)


#### Accuracy demo

[![Accuracy demo](https://img.youtube.com/vi/d-I8oVhZjPM/0.jpg)](https://www.youtube.com/watch?v=d-I8oVhZjPM)

# Training

If you are going to use the already existing weights you can skip this step.

The `betapose` and `sspe` folders are used for training. To train each of the methods reformat the datasets in the needed format. You can do that by using the tools provided in this repository under `tools/converters/`. After your dataset is ready follow the README.md files of each method which explain how to do the training in more detail.

Notice: For training SSPE you would need a GPU with at least 8GB of memory

# Evaluation

To evaluate SSPE just run the SSPE [evaluation script](https://github.com/danieldimit/6d-pose-estimation-with-ml-in-ar/blob/master/sspe/valid.py).

To evaluate Betapose with the same metrics as the ones used by SSPE do the following:
1) Run the [betapose_evaluate.py](https://github.com/danieldimit/6d-pose-estimation-with-ml-in-ar/blob/master/betapose/3_6Dpose_estimator/betapose_evaluate.py) with `save` option enabled
2) Then get the outputed `Betapose-results.json` file and move it in the directory of [the evaluation script](https://github.com/danieldimit/6d-pose-estimation-with-ml-in-ar/blob/master/tools/evaluators/betapose/validateBetapose.py). Then also move the SSPE label files for the same images in the same directory. Then run [the evaluation script](https://github.com/danieldimit/6d-pose-estimation-with-ml-in-ar/blob/master/tools/evaluators/betapose/validateBetapose.py)

# Running the AR-service

