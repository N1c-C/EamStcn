# Object Segmentation in Videos
## EamStcn: Efficient Adaptive Memory Space-Time Correspondence Network 


This repository is my master’s project exploring accurate real-time video object segmentation (VOS). 

The research concentrates on two areas:
 1) The underuse of EfficentNets in video object segmentation (VOS) applications. (The default in the sector is to use a Residual Network (ResNet))
 2) A dynamic save algorithm based on the rate-of-change between frames to improve the performance of the popular memory model approach for VOS applications.

Check out the [**wiki**](https://github.com/N1c-C/EamStcn-Video-Object-Segmentation/wiki) for an explanation of the model and extra results.


EamStcn is a Space-Time Correspondence Network (Cheng et al., 2021) that saves predictions(values/segmentations) through time to form a matching set (or memory) to compare the current frame with. For each frame in a video sequence, a query key is generated and used to reference the memory. The memory consists of previously saved query keys and their corresponding value (segmentation/image mask).  L2 Similarity (opposed to the typically used cosine similarity function) returns the top 20 matching features which are merged and decoded to produce the segmentation mask.

All previous memory model research uses ResNets for the query and value encoders. EamStcn uses EfficientNets for the two encoders. The code provided allows for any combination of EfficientNet to be chosen.

Memory models typically save every fifth frame to memory. This process is inefficient as it often stores redundant information (when there is little change in an object across a video) or misses storing vitally important information (when there is a sudden rapid change in appearance or position). EamStcn uses a novel, motion-aware save function[1] that measures the rate of change between frames to determine if the current one should be kept in the memory.
<br/><br/><br/>

<p align="center" width="100%">
    <img width="70%" src="https://github.com/N1c-C/EamStcn/assets/103114303/141419f8-f480-456e-b44b-426db0c3b643">
  <br/>EamStcn Overview
</p>

<br/><br/><br/>
# Segmentation Results
The best model achieved an 84% J&F score on the DAVIS-17 dataset.


<p align="center" width="100%">
    <img width="70%" src="https://github.com/N1c-C/EamStcn/assets/103114303/9b467219-e73e-4068-bf79-69174eacca09">
</p>


<br/><br/><br/>

# Main Conclusions

1) An EfficientNet backbone for the query and value encoders improved the overall J&F score by 2% compared with ResNets, with a small loss in inference speed. Inference speeds were significantly faster than the ResNet control when smaller EfficientNet models were used, with a loss in J&F accuracy of 1.5%  However, EfficientNets do not generalise to unseen data as well as ResNets, nor are they so simple to train or implement.

2) Although incredibly simple, the dynamic memory save algorithm generally improves a network's accuracy between 0.4 and 1%. Typically, adaptive saving reduces the number of frames stored across a dataset (Good for longer videos), in turn increasing the inference speed by a few per cent. On datasets where the adaptive save becomes a penalty compared to a fixed saving rate - the function still achieves the optimal balance of the highest accuracy/fastest inference speed.
  
3) The EamStcn model struggles with multi-similar-object segmentation where the objects move and occupy space where the others have previously been such as the carousel video from the DAVIS test-dev data set. As such the model would benefit from the inclusion of positional encodings to help with complex scenes.


<p align="center" width="100%">
    <img width="60%" src="https://github.com//N1c-C/EamStcn/assets/103114303/3a47a5cd-5c05-417f-b5b1-2b10b2b9621f">
    <br/>Example of poor segmentation on a rotating similar object sequence 
</p>



<br/><br/><br/>

# References

[1] The adaptive save function was inspired by Wang et al., 2021.



Cheng, H. K., Tai, Y.-W. & Tang, C.-K. 2021. Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation. In: Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P. S. & Vaughan, J. W. (eds.).

Wang, H., Jiang, X., Ren, H., Hu, Y. & Bai, S. SwiftNet: Real-time Video Object Segmentation.  2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20-25 June 2021 2021. 1296-1305, 10.1109/CVPR46437.2021.00135.

