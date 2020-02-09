
Image Denoising
https://github.com/wenbihan/reproducible-image-denoising-state-of-the-art/


# reproducible-image-denoising-state-of-the-art
Collection of popular and reproducible image denoising works.

Criteria: works must have codes available, and the reproducible results demonstrate state-of-the-art performances.

This collection is inspired by the [summary by flyywh](https://github.com/flyywh/Image-Denoising-State-of-the-art)

Note: This repo focuses on single image denoising in general, and will exclude multi-frame and video denoising works.

## Denoising Algorithms
#### Filter
 * NLM [[Web]](https://sites.google.com/site/shreyamsha/publications/image-denoising-based-on-nlfmt) [[Code]](https://www.mathworks.com/matlabcentral/fileexchange/44090-image-denoising-based-on-non-local-means-filter-and-its-method-noise-thresholding?focused=3806802&tab=function) [[PDF]](https://link.springer.com/article/10.1007/s11760-012-0389-y)
   * A non-local algorithm for image denoising (CVPR 05), Buades et al.
   * Image denoising based on non-local means filter and its method noise thresholding (SIVP2013), B. Kumar
 * BM3D [[Web]](http://www.cs.tut.fi/~foi/GCF-BM3D/) [[Code]](http://www.cs.tut.fi/~foi/GCF-BM3D/BM3D.zip) [[PDF]](http://www.cs.tut.fi/~foi/GCF-BM3D/SPIE08_deblurring.pdf)
   * Image restoration by sparse 3D transform-domain collaborative filtering (SPIE Electronic Imaging 2008), Dabov et al.   
 * PID [[Web]](http://www.cgg.unibe.ch/publications/progressive-image-denoising) [[Code]](http://www.cgg.unibe.ch/publications/progressive-image-denoising/pid.zip) [[PDF]](http://www.cgg.unibe.ch/publications/2014/progressive-image-denoising/at_download/file)
   * Progressive Image Denoising (TIP 2014), C. Knaus et al.

#### Sparse Coding
 * KSVD [[Web]](http://www.cs.technion.ac.il/~ronrubin/software.html) [[Code]](https://github.com/jbhuang0604/SelfSimSR/tree/master/Lib/KSVD) [[PDF]](http://www.egr.msu.edu/~aviyente/elad06.pdf)
   * Image Denoising Via Sparse and Redundant Representations Over Learned Dictionaries (TIP 2006), Elad et al.
 * LSSC [[Web]](https://lear.inrialpes.fr/people/mairal/) [[Code]](https://lear.inrialpes.fr/people/mairal/resources/denoise_ICCV09.tar.gz) [[PDF]](http://www.di.ens.fr/~fbach/iccv09_mairal.pdf)
   * Non-local Sparse Models for Image Restoration (ICCV 2009), Mairal et al.
 * NCSR [[Web]](http://www4.comp.polyu.edu.hk/~cslzhang/NCSR.htm) [[Code]](http://www4.comp.polyu.edu.hk/~cslzhang/code/NCSR.rar) [[PDF]](http://www4.comp.polyu.edu.hk/~cslzhang/paper/NCSR_TIP_final.pdf)
   * Nonlocally Centralized Sparse Representation for Image Restoration (TIP 2012), Dong et al.  
 * OCTOBOS [[Web]](http://transformlearning.csl.illinois.edu/projects/) [[Code]](https://github.com/wenbihan/octobos_IJCV2016) [[PDF]](http://transformlearning.csl.illinois.edu/assets/Sai/JournalPapers/SaiBihanIJCV2014OCTOBOS.pdf)
   * Structured Overcomplete Sparsifying Transform Learning with Convergence Guarantees and Applications (IJCV 2015), Wen et al. 
 * GSR [[Web]](https://jianzhang.tech/projects/GSR/) [[Code]](http://csjianzhang.github.io/codes/GSR_Code_Package_3.0.zip) [[PDF]](http://csjianzhang.github.io/papers/TIP2014_single.pdf)
   * Group-based Sparse Representation for Image Restoration (TIP 2014), Zhang et al.
 * TWSC [[Web]](https://github.com/csjunxu/TWSC-ECCV2018) [[Code]](https://github.com/csjunxu/TWSC-ECCV2018) [[PDF]](http://openaccess.thecvf.com/content_ECCV_2018/papers/XU_JUN_A_Trilateral_Weighted_ECCV_2018_paper.pdf)
   * A Trilateral Weighted Sparse Coding Scheme for Real-World Image Denoising (ECCV 2018), Xu et al.
  
#### Classical External Priors
 * EPLL [[Web]](https://people.csail.mit.edu/danielzoran/) [[Code]](https://people.csail.mit.edu/danielzoran/epllcode.zip) [[PDF]](http://people.ee.duke.edu/~lcarin/EPLICCVCameraReady.pdf)
   * From Learning Models of Natural Image Patches to Whole Image Restoration (ICCV2011), Zoran et al.
 * GHP [[Web]][[Code]](https://github.com/tingfengainiaini/GHPBasedImageRestoration) [[PDF]](https://www.cv-foundation.org/openaccess/content_cvpr_2013/papers/Zuo_Texture_Enhanced_Image_2013_CVPR_paper.pdf)
   * Texture Enhanced Image Denoising via Gradient Histogram Preservation (CVPR2013), Zuo et al.
 * PGPD [[Web]][[Code]](https://github.com/csjunxu/PGPD_Offline_BID) [[PDF]](http://www4.comp.polyu.edu.hk/~cslzhang/paper/PGPD.pdf)
   * Patch Group Based Nonlocal Self-Similarity Prior Learning for Image Denoising (ICCV 2015), Xu et al.
 * PCLR [[Web]][[Code]](http://www4.comp.polyu.edu.hk/~cslzhang/code/PCLR.zip) [[PDF]](http://www4.comp.polyu.edu.hk/~cslzhang/paper/PCLR.pdf)
   * External Patch Prior Guided Internal Clustering for Image Denoising (ICCV 2015), Chen et al.
   
#### Low Rank
 * SAIST [[Web]](http://see.xidian.edu.cn/faculty/wsdong/wsdong_Publication.htm) [Code by request] [[PDF]](http://see.xidian.edu.cn/faculty/wsdong/Papers/Journal/TIP_LASSC.pdf)
   * Nonlocal image restoration with bilateral variance estimation: a low-rank approach (TIP2013), Dong et al.
 * WNNM [[Web]](https://sites.google.com/site/shuhanggu/home) [[Code]](http://www4.comp.polyu.edu.hk/~cslzhang/code/WNNM_code.zip) [[PDF]](https://pdfs.semanticscholar.org/6d55/6272625b672ba54b5ab3d9e6474088a4b78f.pdf)
   * Weighted Nuclear Norm Minimization with Application to Image Denoising (CVPR2014), Gu et al.
 * Multi-channel WNNM [[Web]](http://www4.comp.polyu.edu.hk/~csjunxu/Publications.html) [[Code]](http://www4.comp.polyu.edu.hk/~csjunxu/code/MCWNNM.zip) [[PDF]](http://www4.comp.polyu.edu.hk/~csjunxu/paper/MCWNNM.pdf)
   * Multi-channel Weighted Nuclear Norm Minimization for Real Color Image Denoising (ICCV 2017), Xu et al.
   
#### Deep Denoising
 * SF [[Web]](http://www.visinf.tu-darmstadt.de/vi_research/code/index.en.jsp#shrinkage_fields) [[Code]](https://github.com/uschmidt83/shrinkage-fields) [[PDF]](http://research.uweschmidt.org/pubs/cvpr14schmidt.pdf)
   * Shrinkage Fields for Effective Image Restoration (CVPR 2014), Schmidt et al.
 * TNRD [[Web]](http://www.icg.tugraz.at/Members/Chenyunjin/about-yunjin-chen) [[Code]](https://www.dropbox.com/s/8j6b880m6ddxtee/TNRD-Codes.zip?dl=0) [[PDF]](https://arxiv.org/pdf/1508.02848.pdf)
   * Trainable nonlinear reaction diffusion: A flexible framework for fast and effective image restoration (TPAMI 2016), Chen et al.
 * RED [[Web]](https://bitbucket.org/chhshen/image-denoising/) [[Code]](https://bitbucket.org/chhshen/image-denoising/) [[PDF]](https://arxiv.org/pdf/1603.09056.pdf)
   * Image Restoration Using Very Deep Convolutional Encoder-Decoder Networks with Symmetric Skip Connections (NIPS2016), Mao et al.
 * DnCNN [[Web]](https://github.com/cszn/DnCNN) [[Code]](https://github.com/cszn/DnCNN) [[PDF]](https://arxiv.org/pdf/1608.03981v1.pdf)
   * Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising (TIP2017), Zhang et al.
 * MemNet [[Web]](https://github.com/tyshiwo/MemNet) [[Code]](https://github.com/tyshiwo/MemNet) [[PDF]](http://cvlab.cse.msu.edu/pdfs/Image_Restoration%20using_Persistent_Memory_Network.pdf)
   * MemNet: A Persistent Memory Network for Image Restoration (ICCV2017), Tai et al.  
 * WIN [[Web]](https://github.com/cswin/WIN) [[Code]](https://github.com/cswin/WIN) [[PDF]](https://arxiv.org/pdf/1707.09135.pdf)
   * Learning Pixel-Distribution Prior with Wider Convolution for Image Denoising (Arxiv), Liu et al.    
 * F-W Net [[Web]](https://github.com/sunke123/FW-Net) [[Code]](https://github.com/sunke123/FW-Net) [[PDF]](https://arxiv.org/abs/1802.10252)
   * L_p-Norm Constrained Coding With Frank-Wolfe Network (Arxiv), Sun et al.
 * NLCNN [[Web]](https://cig.skoltech.ru/publications) [[Code]](https://github.com/cig-skoltech/NLNet) [[PDF]](http://www.skoltech.ru/app/data/uploads/sites/19/2017/06/1320.pdf)
   * Non-Local Color Image Denoising with Convolutional Neural Networks (CVPR 2017), Lefkimmiatis.
 * xUnit [[Web]](https://github.com/kligvasser/xUnit) [[Code]](https://github.com/kligvasser/xUnit) [[PDF]](https://arxiv.org/pdf/1711.06445.pdf)
   * xUnit: Learning a Spatial Activation Function for Efficient Image Restoration (Arxiv), Kligvasser et al.  
 * UDNet [[Web]](https://github.com/cig-skoltech/UDNet) [[Code]](https://github.com/cig-skoltech/UDNet) [[PDF]](https://arxiv.org/pdf/1711.07807.pdf)
   * Universal Denoising Networks : A Novel CNN Architecture for Image Denoising (CVPR 2018), Stamatios  Lefkimmiatis.   
 * Wavelet-CNN [[Web]](https://github.com/lpj0/MWCNN) [[Code]](https://github.com/lpj0/MWCNN) [[PDF]](https://arxiv.org/abs/1805.07071)
   * Multi-level Wavelet-CNN for Image Restoration (Arxiv), Liu et al.  
 * FFDNet [[Web]](https://github.com/cszn/FFDNet/) [[Code]](https://github.com/cszn/FFDNet/) [[PDF]](https://arxiv.org/abs/1710.04026)
   * FFDNet: Toward a Fast and Flexible Solution for CNN-Based Image Denoising (TIP), Zhang et al.
 * FC-AIDE [[Web]](https://github.com/csm9493/FC-AIDE) [[Code]](https://github.com/csm9493/FC-AIDE) [[PDF]](https://arxiv.org/pdf/1807.07569.pdf)
   * Fully Convolutional Pixel Adaptive Image Denoiser (Arxiv), Cha et al.  
 * CBDNet [[Web]](https://github.com/GuoShi28/CBDNet) [[Code]](https://github.com/GuoShi28/CBDNet) [[PDF]](https://arxiv.org/pdf/1807.04686.pdf)
   * Toward Convolutional Blind Denoising of Real Photographs (Arxiv), Guo et al.  
 * UDN [[Web]](https://cig.skoltech.ru/publications) [[Code]](https://github.com/cig-skoltech/UDNet) [[PDF]](http://www.skoltech.ru/app/data/uploads/sites/19/2018/03/UDNet_CVPR2018.pdf)
   * Universal Denoising Networks- A Novel CNN Architecture for Image Denoising (CVPR 2018), Lefkimmiatis.     
 * N3 [[Web]](https://github.com/visinf/n3net) [[Code]](https://github.com/visinf/n3net) [[PDF]](https://arxiv.org/abs/1810.12575)
   * Neural Nearest Neighbors Networks (NIPS 2018), Plotz et al.  
 * NLRN [[Web]](https://github.com/Ding-Liu/NLRN) [[Code]](https://github.com/Ding-Liu/NLRN) [[PDF]](https://arxiv.org/pdf/1806.02919.pdf)
   * Non-Local Recurrent Network for Image Restoration (NIPS 2018), Liu et al.
 * RDN+ [[Web]](https://github.com/yulunzhang/RDN) [[Code]](https://github.com/yulunzhang/RDN) [[PDF]](https://arxiv.org/abs/1812.10477)
   * Residual Dense Network for Image Restoration (CVPR 2018), Zhang et al.
 * FOCNet [[Web]](https://github.com/hsijiaxidian/FOCNet) [[Code]](https://github.com/hsijiaxidian/FOCNet) [[PDF]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Jia_FOCNet_A_Fractional_Optimal_Control_Network_for_Image_Denoising_CVPR_2019_paper.pdf)
   * FOCNet: A Fractional Optimal Control Network for Image Denoising (CVPR 2019), Jia et al.

#### Unsupervised / Weakly-Supervised Deep Denoising
 * Noise2Noise [[Web]](https://github.com/yu4u/noise2noise) [[TF Code]](https://github.com/NVlabs/noise2noise) [[Keras Unofficial Code]](https://github.com/yu4u/noise2noise) [[PDF]](https://arxiv.org/pdf/1803.04189.pdf)
   * Noise2Noise: Learning Image Restoration without Clean Data (ICML 2018), Lehtinen et al.      
 * DIP [[Web]](https://dmitryulyanov.github.io/deep_image_prior) [[Code]](https://github.com/DmitryUlyanov/deep-image-prior) [[PDF]](https://sites.skoltech.ru/app/data/uploads/sites/25/2018/04/deep_image_prior.pdf)
   * Deep Image Prior (CVPR 2018), Ulyanov et al.
 * Noise2Void [[Web]](https://github.com/juglab/n2v) [[Code]](https://github.com/juglab/n2v) [[PDF]](https://arxiv.org/abs/1811.10980)
   * Learning Denoising from Single Noisy Images (CVPR 2019), Krull et al.
 * Noise2Self [[Web]](https://github.com/czbiohub/noise2self) [[Code]](https://github.com/czbiohub/noise2self) [[PDF]](https://arxiv.org/abs/1811.10980)
   * LNoise2Self: Blind Denoising by Self-Supervision (ICML 2019), Batson and Royer
 * Self-Supervised Denoising [[Web]](https://github.com/NVlabs/selfsupervised-denoising) [[Code]](https://github.com/NVlabs/selfsupervised-denoising) [[PDF]](https://arxiv.org/abs/1901.10277)
   * High-Quality Self-Supervised Deep Image Denoising (NIPS 2019), Laine et al.   

#### Real Noise Removal 
 * RIDNet [[Web]](https://github.com/saeed-anwar/RIDNet) [[Code]](https://github.com/saeed-anwar/RIDNet) [[PDF]](https://arxiv.org/abs/1904.07396)
   * Real Image Denoising with Feature Attention (ICCV 2019), Anwar and Barnes.      
 * CBDNet [[Web]](https://github.com/GuoShi28/CBDNet) [[Code]](https://github.com/GuoShi28/CBDNet) [[PDF]](http://openaccess.thecvf.com/content_CVPR_2019/papers/Guo_Toward_Convolutional_Blind_Denoising_of_Real_Photographs_CVPR_2019_paper.pdf)
   * Real Image Denoising with Feature Attention (CVPR 2019), Guo et al. 
 * VDNNet [[Web]](https://github.com/zsyOAOA/VDNet) [[Code]](https://github.com/zsyOAOA/VDNet) [[PDF]](https://arxiv.org/pdf/1908.11314v2.pdf)
   * Variational Denoising Network: Toward Blind Noise Modeling and Removal (NIPS 2019), Yue et al. 
 
#### Hybrid Model for Denoising
 * STROLLR [[PDF]](http://transformlearning.csl.illinois.edu/assets/Bihan/ConferencePapers/BihanICASSP2017strollr.pdf) [[Code]](https://github.com/wenbihan/strollr2d_icassp2017) 
   * When Sparsity Meets Low-Rankness: Transform Learning With Non-Local Low-Rank Constraint for Image Restoration (ICASSP 2017), Wen et al.
 * Meets High-level Tasks [[PDF]](https://arxiv.org/pdf/1706.04284.pdf) [[Code]](https://github.com/wenbihan/DeepDenoising) 
   * When Image Denoising Meets High-Level Vision Tasks: A Deep Learning Approach (IJCAI 2018), Liu et al.
 * USA [[PDF]](https://arxiv.org/pdf/1905.08965.pdf) [[Code]](https://github.com/sharonwang1/seg_denoising) 
   * Segmentation-aware Image Denoising Without Knowing True Segmentation (Arxiv), Wang et al.

#### Image Noise Level Estimation
 * SINLE [[PDF]](http://www.ok.sc.e.titech.ac.jp/res/NLE/TIP2013-noise-level-estimation06607209.pdf) [[Code]](https://www.mathworks.com/matlabcentral/fileexchange/36921-noise-level-estimation-from-a-single-image) [[Slides]](https://wwwpub.zih.tu-dresden.de/~hh3/Hauptsem/SS16/noise.pdf)
   * Single-image Noise Level Estimation for Blind Denoising (TIP 2014), Liu et al.


#### Novel Benchmark
 * ReNOIR [[Web]](http://ani.stat.fsu.edu/~abarbu/Renoir.html) [[Data]](http://ani.stat.fsu.edu/~abarbu/Renoir.html) [[PDF]](https://arxiv.org/pdf/1409.8230.pdf)
   * RENOIR - A Dataset for Real Low-Light Image Noise Reduction (Arxiv 2014), Anaya, Barbu.   
 * Darmstadt [[Web]](https://noise.visinf.tu-darmstadt.de/) [[Data]](https://noise.visinf.tu-darmstadt.de/downloads/) [[PDF]](https://download.visinf.tu-darmstadt.de/papers/2017-cvpr-ploetz-benchmarking_denoising_algorithms-preprint.pdf)
   * Benchmarking Denoising Algorithms with Real Photographs (CVPR 2017), Tobias Plotz, Stefan Roth.
 * PolyU [[Web]](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset) [[Data]](https://github.com/csjunxu/PolyU-Real-World-Noisy-Images-Dataset) [[PDF]](https://arxiv.org/pdf/1804.02603.pdf)
   * Real-world Noisy Image Denoising: A New Benchmark (Arxiv), Xu et al.
   
#### Commonly Used Denoising Dataset
 * Kodak [[Web]](http://r0k.us/graphics/kodak/)
 * USC SIPI-Misc [[Web]](http://sipi.usc.edu/database/database.php?volume=misc) 
 * BSD [[Web]](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/)  

#### Commonly Used Image Quality Metrics
 * PSNR (Peak Signal-to-Noise Ratio) [[Wiki]](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) [[Matlab Code]](https://www.mathworks.com/help/images/ref/psnr.html) [[Python Code]](https://github.com/aizvorski/video-quality)
 * SSIM (Structural similarity) [[Wiki]](https://en.wikipedia.org/wiki/Structural_similarity) [[Matlab Code]](http://www.cns.nyu.edu/~lcv/ssim/ssim_index.m) [[Python Code]](https://github.com/aizvorski/video-quality/blob/master/ssim.py)
 * NIQE (Naturalness Image Quality Evaluator) [[Web]](http://live.ece.utexas.edu/research/Quality/nrqa.htm) [[Matlab Code]](http://live.ece.utexas.edu/research/Quality/nrqa.htm) [[Python Code]](https://github.com/aizvorski/video-quality/blob/master/niqe.py)



   
