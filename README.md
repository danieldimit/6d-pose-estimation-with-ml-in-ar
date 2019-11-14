# Implementation and Evaluation of Deep Learning Approaches for Real-time 3D Robot Localization in an Augmented Reality Use Case

#### External links
* The manual 6D object pose annotation tool - [http://annotate.photo/](http://annotate.photo?ref=github1)
* [Single Shot Pose Estimation (SSPE)](https://github.com/microsoft/singleshotpose/)
* [Betapose](https://github.com/sjtuytc/betapose)
* [Gabriel](https://github.com/cmusatyalab/gabriel)
* The 5 datasets used for training and evaluation - [Google Drive](https://drive.google.com/open?id=1xiS53HLcr5vGQRmiOt7Yotwso9g-ddht)
* The weights for SSPE and Betapose - [Google Drive](https://drive.google.com/open?id=1XQGH31AxFJWjLGV19yYY9HBrvVzqMojH)
* [Presentation slides](https://drive.google.com/open?id=1bbbv07PCKYgoZUM4EUmJncLuz8X0vKwG) and [documentation](https://drive.google.com/open?id=1tO_l7d1eu-N_9qK5HzfsC3qdeGQWaC-b)

#### General information
This project is a fusion of [SSPE](https://github.com/microsoft/singleshotpose/), [Betapose](https://github.com/sjtuytc/betapose) and [Gabriel](https://github.com/cmusatyalab/gabriel). The `betapose` and `sspe` directories are used for training and evaluation. The `tools` directory contrains useful tools like projectors, dataset converters and augmetors as well as PLY-manipulators which are mainly useful for data preparation. The `gabriel` directory contains the [Gabriel](https://github.com/cmusatyalab/gabriel) code base with 2 new proxies (cognitive engines) for Betapose and SSPE.

In many subdirectories there is a `commands.txt` file which contains the actual commands that were used during the project. They can be used for reference.

The programs were run with Nvidia driver 418.87 and CUDA version 10.0. Betapose and SSPE (including their gabriel proxies) were run on Python 3.6.8, while the 2 other servers of Gabriel were run on Python 2.7.16. 

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

You have to run the Gabriel servers locally. In case you're stuck with those instructions you can look at the [Gabriel README](https://github.com/cmusatyalab/gabriel) for more info. 

It is advised to have 3 Anaconda environments - for Betapose (Python 2.6), for SSPE (Python 2.6) and for Gabriel (Python 2.6). After having installed all needed libraries in the coresponding environments you can purceed with running the servers:

1) Connect the computer and the AR-enabled device to the same WiFi network. It would be easiest if you make one of the devices a hotspot.
2) (Gabriel Environment) Run the control server from the `gabriel/server/bin` directory
```
python gabriel-control -n wlp3s0 -l
```
3) (Gabriel Environment) Run the ucomm server from the `gabriel/server/bin` directory
```
python gabriel-ucomm -n wlp3s0
```

Now we have the core of Gabriel running with no cognitive engines running. To start the __SSPE congnitive engine__ do the following:
1) Go to the `gabriel/server/bin/example-proxies/gabriel-proxy-sspe` directory
2) Put the PLY 3D model in the `./sspd/3d_models` directory
3) Put the configuration file in the `./sspd/cfg` directory
4) Put the weights in the `./sspd/backup/kuka` directory
5) Run the following command (it might change according to the names you've used):
```
python proxy.py ./sspd/cfg/kuka.data ./sspd/cfg/yolo-pose-noanchor.cfg ./sspd/backup/kuka/model.weights 0.0.0.0:8021
```

To start the __Betapose cognitive engine__ do the following:
1) Go to the `gabriel/server/bin/example-proxies/gabriel-proxy-betapose` directory
2) Put the PLY 3D object model in the `./betapose/models/models` directory
3) Put the PLY 3D key points model in the `./betapose/models/kpmodels` directory
4) Put the KPD weights named `kpd.pkl` in the `./betapose/weights` directory
5) Put the YOLO weights named `yolo.weights` in the `./betapose/weights` directory
6) Run the following command:
```
python proxy.py --control_server 0.0.0.0:8021
```
