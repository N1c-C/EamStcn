# Object Segmentation in Videos
## EamStcn: Efficient Adaptive Memory Space-Time Correspondence Network 


This repository is my master’s project exploring accurate real-time video object segmentation (VOS). 

The research concentrates on two areas:
 1) The underuse of EfficentNets in video object segmentation (VOS) applications - Residual Networks (ResNets) are habitually used in all VOS research
 2) A dynamic save algorithm based on the rate of pixel change between frames to improve the performance of the popular memory model approach for VOS applications.

<p></p><br/>

# Baisc Operation
* EamStcn is a Space-Time Correspondence Network (Cheng et al., 2021) that saves predictions(values/segmentations) through time to form a matching set (or memory) to compare with the current frame<br/>
* For each frame in a video sequence, a query key is generated and used to reference the memory<br/> 
* The memory consists of previously saved query keys and their corresponding value (segmentation/image mask)<br/>
* Similarity or affinity is used to return the top 20 matching features which are merged and decoded to produce the segmentation mask. For this model, L2 similarity is used as per the original paper but other functions (e.g. cosine) produce the same results<br/>
* Memory models typically save every fifth frame to memory. EamStcn uses a novel, motion-aware save function[1] that measures the rate of pixel change between frames and saves the current segmentation when a threshold is met<br/>
<br/>
<p align="center" width="100%">
    <img width="70%" src="https://github.com/N1c-C/EamStcn/assets/103114303/141419f8-f480-456e-b44b-426db0c3b643">
  <br/><strong>EamStcn Overview: EffiecientNetV2 CNN networks are used for the Query and Value Encoders. An accuracy-speed trade-off to suit an application can be achieved by careful selection of the two networks</strong>
</p>
<br/>

# Training

EfficientNets pre-trained on the [**ImageNet**](https://www.image-net.org/) dataset were trained in two phases
* Phase 1: Still images and random affine transforms are used to create faux video sequences of three frames with a single object
* Phase 2: The _YouTube2018_ training dataset was used. Three temporally ordered frames with augmentations formed the training sample.  The gap between frames gradually increased from five to twenty-five before reducing back to five. A maximum of two objects was randomly selected from any given sequence in the dataset

<br/>

# J&F Accuracy and Segmentation Results
For this project, B1 EfficientNets were used for both encoders due to time constraints and the difficulty in training larger models without sudden divergence. A feature expansion block after stage 4 of the query/key encoder was necessary to maximise accuracy. This final block widens the final set of features to 512 from 112.

### J&F score on the DAVIS-17 dataset was 84%


<p align="center" width="100%">
    <img width="70%" src="https://github.com/N1c-C/EamStcn/assets/103114303/9b467219-e73e-4068-bf79-69174eacca09">
</p>


<br/>

# Main Conclusions

1) An EfficientNet backbone for the query and value encoders improved the overall J&F score by 2% compared with ResNets, with a small loss in inference speed. Inference speeds were significantly faster than the ResNet control when smaller EfficientNet models were used, with a loss in J&F accuracy of 1.5%  However, EfficientNets do not generalise to unseen data as well as ResNets, nor are they so simple to train or implement.

2) Although incredibly simple, the dynamic memory save algorithm generally improves a network's accuracy between 0.4 and 1%. Typically, adaptive saving reduces the number of frames stored across a dataset (Good for longer videos), in turn increasing the inference speed by a few per cent. On datasets where the adaptive save becomes a penalty compared to a fixed saving rate - the function still achieves the optimal balance of the highest accuracy/fastest inference speed.
  
3) The EamStcn model struggles with multi-similar-object segmentation where the objects move and occupy space where the others have previously been such as the carousel video from the DAVIS test-dev data set. As such the model would benefit from the inclusion of positional encodings to help with complex scenes.

<p align="center" width="100%">
    <img width="60%" src="https://github.com//N1c-C/EamStcn/assets/103114303/3a47a5cd-5c05-417f-b5b1-2b10b2b9621f">
    <br/>Example of poor segmentation on a rotating similar object sequence 
</p>

Check out the [**Wiki**](https://github.com/N1c-C/EamStcn-Video-Object-Segmentation/wiki) for an explanation of the save function and full results.

<br/>

# References

[1] The adaptive save function was inspired by Wang et al., 2021.



Cheng, H. K., Tai, Y.-W. & Tang, C.-K. 2021. Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation. In: Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P. S. & Vaughan, J. W. (eds.).

Wang, H., Jiang, X., Ren, H., Hu, Y. & Bai, S. SwiftNet: Real-time Video Object Segmentation.  2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20-25 June 2021 2021. 1296-1305, 10.1109/CVPR46437.2021.00135.

