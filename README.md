# EamStcn - Efficient Adaptive Memory: Space-Time Correspondence Network(Cheng et al., 2021) 

The result of my Masters' project exploring using EfficentNets in VOS applications. I demonstrate that an EfficientNet backbone improves the overall J&F score by a few per cent, compared with ResNets, with little or no loss in inference speed. However, EfficientNets do not generalise to unseen data as well as ResNets, nor are they so simple to train or implement.

EamStcn is a memory model that saves predictions through time to form a matching set. Each frame produces a query key used to reference the memory to lookup the similar features from the previous frames. Traditionally memory models save every fifth frame. I introduce a novel, motion-aware save function[1] that measures the rate of change between frames to determine if it should be kept in the memory. Although incredibly simple, the algorithm generally improves a network's accuracy between 0.4 and 1%. Typically adaptive saving reduces the number of frames stored across a dataset, in turn increasing the inference speed by a few per cent. On datasets where the adaptive save becomes a penalty compared to a fixed saving rate - the function still achieves the optimal balance of the highest accuracy/fastest inference speed. 





[1] The adaptive save function was inspired by Wang et al., 2021.



Cheng, H. K., Tai, Y.-W. & Tang, C.-K. 2021. Rethinking Space-Time Networks with Improved Memory Coverage for Efficient Video Object Segmentation. In: Ranzato, M., Beygelzimer, A., Dauphin, Y., Liang, P. S. & Vaughan, J. W. (eds.).

Wang, H., Jiang, X., Ren, H., Hu, Y. & Bai, S. SwiftNet: Real-time Video Object Segmentation.  2021 IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 20-25 June 2021 2021. 1296-1305, 10.1109/CVPR46437.2021.00135.

![image](https://user-images.githubusercontent.com/103114303/224546499-75a84d16-c6e4-4bb9-8afd-6db1520b4e6b.png)
