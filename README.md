# Crnn_Ctpn_Unified #
Unified text detection and recognition

First of all, thanks for the author of crnn and ctpn algorithm with opening their codes.
Here is [crnn](https://github.com/meijieru/crnn.pytorch) original codes, here is [ctpn](https://github.com/tianzhi0549/CTPN) original codes and a tensorflow version can be found [here](https://github.com/eragonruan/text-detection-ctpn)

# In my repository, ctpn was built by tensorflow, meanwile crnn was built by pytorch
## Environment 
 Ubuntu16.04(it's more appropriate to build warp-ctc than other)

 python3
 
 tensorflow-gpu(version>=1.6)
 
 pytorch(version>=0.3)
 
 warp-ctc [click here to download](https://github.com/SeanNaren/Warp-ctc)
 
 Just using pip to establish your dev environment, eg. pip install tensorflow-gpu==1.6, pip install torchvesion(it will install pytorch meanwhile)
 
## Run it
 I share my models of crnn and ctpn, you can run it just for take a look of its effect.
 
 Images and results are stored in detect_data file. 
 >cd end2endDec_Rec
 
 >python3 ./ctpn_crnn_combined/end2endDet_Rec.py
 
 ## Results
 following is detect images example 
 ![](https://github.com/Sierkinhane/Crnn_Ctpn_Unified/raw/master/end2endDec_Rec/detect_data/detect_images/1.png)
 ### results were save in detect_data/txt_results
 
 ## Train model
 Here I just share the method that was used to train a crnn model.  In my training, ultimately it reached an accuracy of **97.8%.**
 Before training, you should generate a lmdb dataset. Click [here](https://pan.baidu.com/s/1ufYbnZAZ1q0AlK7yZ08cvQ) for downloading a 3.6 million chinese characters training sets.
 >
 
 
 
