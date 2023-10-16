# Awesome Time Series Forecasting/Prediction Papers
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/ddz16/TSFpaper)

This repository contains a reading list of papers on **Time Series Forecasting/Prediction (TSF)** and **Spatio-Temporal Forecasting/Prediction (STF)**. These papers are mainly categorized according to the type of model. **This repository is still being continuously improved. If you have found any relevant papers that need to be included in this repository, please feel free to submit a pull request (PR) or open an issue.**

Each paper may apply to one or several types of forecasting, including univariate time series forecasting, multivariate time series forecasting, and spatio-temporal forecasting, which are also marked in the Type column. **If covariates and exogenous variables are not considered**, univariate time series forecasting involves predicting the future of one variable with the history of this variable, while multivariate time series forecasting involves predicting the future of C variables with the history of C variables. **Note that repeating univariate forecasting multiple times can also achieve the goal of multivariate forecasting. However, univariate forecasting methods cannot extract relationships between variables, so the basis for distinguishing between univariate and multivariate forecasting methods is whether the method involves interaction between variables. Besides, in the era of deep learning, many univariate models can be easily modified to directly process multiple variables for multivariate forecasting. And multivariate models generally can be directly used for univariate forecasting. Here we classify solely based on the model's description in the original paper.** Spatio-temporal forecasting is often used in traffic and weather forecasting, and it adds a spatial dimension compared to univariate and multivariate forecasting. **In spatio-temporal forecasting, if each measurement point has only one variable, it is equivalent to multivariate forecasting. Therefore, the distinction between spatio-temporal forecasting and multivariate forecasting is not clear. Spatio-temporal models can usually be directly applied to multivariate forecasting, and multivariate models can also be used for spatio-temporal forecasting with minor modifications. Here we also classify solely based on the model's description in the original paper.**

* ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) univariate time series forecasting: ![](https://latex.codecogs.com/svg.image?\inline&space;L\times&space;1&space;\to&space;H\times&space;1), where L is the history length, H is the prediction horizon length.
* ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) multivariate time series forecasting: ![](https://latex.codecogs.com/svg.image?\inline&space;L\times&space;C&space;\to&space;H\times&space;C), where C is the number of variables (channels).
* ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) spatio-temporal forecasting: ![](https://latex.codecogs.com/svg.image?\inline&space;N&space;\times&space;L\times&space;C&space;\to&space;N&space;\times&space;H\times&space;C), where N is the spatial dimension (number of measurement points).

🚩 **I have marked some recommended papers with 🌟 (Just my personal preference 😉).**

## Survey.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
15-11-23|[Multi-step](https://ieeexplore.ieee.org/abstract/document/7422387)|ACOMP 2015|Comparison of Strategies for Multi-step-Ahead Prediction of Time Series Using Neural Network|None
19-06-20|[DL](https://ieeexplore.ieee.org/abstract/document/8742529)| SENSJ 2019|A Review of Deep Learning Models for Time Series Prediction|None
20-09-27|[DL](https://arxiv.org/abs/2004.13408)|Arxiv 2020|Time Series Forecasting With Deep Learning: A Survey|None
22-02-15|[Transformer](https://arxiv.org/abs/2202.07125)|IJCAI 2023|Transformers in Time Series: A Survey|[PaperList](https://github.com/qingsongedu/time-series-transformers-review)
23-03-25|[STGNN](https://arxiv.org/abs/2303.14483)|Arxiv 2023|Spatio-Temporal Graph Neural Networks for Predictive Learning in Urban Computing: A Survey|None
23-05-01|[Diffusion](https://arxiv.org/abs/2305.00624)|Arxiv 2023|Diffusion Models for Time Series Applications: A Survey|None
23-06-16|[SSL](https://arxiv.org/abs/2306.10125)|Arxiv 2023|Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects|None
23-07-07|[GNN](https://arxiv.org/abs/2307.03759)|Arxiv 2023|A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection|[PaperList](https://github.com/KimMeen/Awesome-GNN4TS)
23-10-09|[BasicTS](https://arxiv.org/abs/2310.06119)|Arxiv 2023|Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis|Benchmark
23-10-11|[ProbTS](https://arxiv.org/abs/2310.07446)|Arxiv 2023|ProbTS: A Unified Toolkit to Probe Deep Time-series Forecasting|Toolkit


## Transformer.
Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|----|-----|-----|-----
19-06-29|[LogTrans](https://arxiv.org/abs/1907.00235)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPS 2019|Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting|[flowforecast](https://github.com/AIStream-Peelout/flow-forecast/blob/master/flood_forecast/transformer_xl/transformer_bottleneck.py) |
19-12-19|[TFT](https://arxiv.org/abs/1912.09363)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |IJoF 2021|[Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://www.zhihu.com/question/451816360/answer/2319401126)|[tft](https://github.com/google-research/google-research/tree/master/tft) |
20-01-23|[InfluTrans](https://arxiv.org/abs/2001.08317)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |Arxiv 2020|[Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case](https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e)|[influenza transformer](https://github.com/KasperGroesLudvigsen/influenza_transformer) |
20-06-05|[AST](https://proceedings.neurips.cc/paper/2020/file/c6b8c8d762da15fa8dbbdfb6baf9e260-Paper.pdf)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPS 2020|Adversarial Sparse Transformer for Time Series Forecasting|[AST](https://github.com/hihihihiwsf/AST)
20-12-14|[Informer](https://arxiv.org/abs/2012.07436)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |AAAI 2021|[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://zhuanlan.zhihu.com/p/467523291)|[Informer](https://github.com/zhouhaoyi/Informer2020)
21-05-22|[ProTran](https://proceedings.neurips.cc/paper_files/paper/2021/file/c68bd9055776bf38d8fc43c0ed283678-Paper.pdf)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |NIPS 2021|Probabilistic Transformer for Time Series Analysis|None
21-06-24|[Autoformer](https://arxiv.org/abs/2106.13008)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |NIPS 2021|[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://zhuanlan.zhihu.com/p/385066440)|[Autoformer](https://github.com/thuml/Autoformer)
21-09-17|[Aliformer](https://arxiv.org/abs/2109.08381)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2021 | From Known to Unknown: Knowledge-guided Transformer for Time-Series Sales Forecasting in Alibaba | None
21-10-05|[Pyraformer](https://openreview.net/pdf?id=0EXmFzUn5I)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICLR 2022|[Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting](https://zhuanlan.zhihu.com/p/467765457)|[Pyraformer](https://github.com/alipay/Pyraformer)
22-01-14|[Preformer](https://arxiv.org/abs/2202.11356)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICASSP 2023|[Preformer: Predictive Transformer with Multi-Scale Segment-wise Correlations for Long-Term Time Series Forecasting](https://zhuanlan.zhihu.com/p/536398013)|[Preformer](https://github.com/ddz16/Preformer)
22-01-30|[FEDformer](https://arxiv.org/abs/2201.12740)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICML 2022|[FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://zhuanlan.zhihu.com/p/528131016)|[FEDformer](https://github.com/MAZiqing/FEDformer)
22-02-03|[ETSformer](https://arxiv.org/abs/2202.01381)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2022|[ETSformer: Exponential Smoothing Transformers for Time-series Forecasting](https://blog.salesforceairesearch.com/etsformer-time-series-forecasting/)|[etsformer](https://github.com/salesforce/etsformer)
22-02-07|[TACTiS](https://arxiv.org/abs/2202.03528)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICML 2022|TACTiS: Transformer-Attentional Copulas for Time Series|[TACTiS](https://github.com/ServiceNow/tactis)
22-04-28|[Triformer](https://arxiv.org/abs/2204.13767)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |IJCAI 2022|[Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting](https://blog.csdn.net/zj_18706809267/article/details/125048492)| [Triformer](https://github.com/razvanc92/triformer)
22-05-27|[TDformer](https://arxiv.org/abs/2212.08151)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |NIPSW 2022|[First De-Trend then Attend: Rethinking Attention for Time-Series Forecasting](https://zhuanlan.zhihu.com/p/596022160)|[TDformer](https://github.com/BeBeYourLove/TDformer)
22-05-28|[Non-stationary Transformer](https://arxiv.org/abs/2205.14415)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |NIPS 2022|[Non-stationary Transformers: Rethinking the Stationarity in Time Series Forecasting](https://zhuanlan.zhihu.com/p/535931701)|[Non-stationary Transformers](https://github.com/thuml/Nonstationary_Transformers)
22-06-08|[Scaleformer](https://arxiv.org/abs/2206.04038)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICLR 2023|[Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting](https://zhuanlan.zhihu.com/p/535556231)|[Scaleformer](https://github.com/BorealisAI/scaleformer)
22-08-14|[Quatformer](https://dl.acm.org/doi/abs/10.1145/3534678.3539234)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |KDD 2022|Learning to Rotate: Quaternion Transformer for Complicated Periodical Time Series Forecasting|[Quatformer](https://github.com/DAMO-DI-ML/KDD2022-Quatformer)
22-08-30|[Persistence Initialization](https://arxiv.org/abs/2208.14236)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |Arxiv 2022|[Persistence Initialization: A novel adaptation of the Transformer architecture for Time Series Forecasting](https://zhuanlan.zhihu.com/p/582419707)|None
22-09-08|[W-Transformers](https://arxiv.org/abs/2209.03945)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |Arxiv 2022|[W-Transformers: A Wavelet-based Transformer Framework for Univariate Time Series Forecasting](https://zhuanlan.zhihu.com/p/582419707)|[w-transformer](https://github.com/capwidow/w-transformer)
22-09-22|[Crossformer](https://openreview.net/forum?id=vSVLM2j9eie)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICLR 2023|Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting|[Crossformer](https://github.com/Thinklab-SJTU/Crossformer)
22-09-22|[PatchTST](https://arxiv.org/abs/2211.14730)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |ICLR 2023|[A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://zhuanlan.zhihu.com/p/602332939)|[PatchTST](https://github.com/yuqinie98/patchtst)
22-11-29|[AirFormer](https://arxiv.org/abs/2211.15979)| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2023 |AirFormer: Predicting Nationwide Air Quality in China with Transformers | [AirFormer](https://github.com/yoshall/airformer)
23-01-19|[PDFormer](https://arxiv.org/abs/2301.07945)| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2023 | PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction | [PDFormer](https://github.com/BUAABIGSCity/PDFormer)
23-05-20|[CARD](https://arxiv.org/abs/2305.12095)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2023|Make Transformer Great Again for Time Series Forecasting: Channel Aligned Robust Dual Transformer|None
23-05-24|[JTFT](https://arxiv.org/abs/2305.14649) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | A Joint Time-frequency Domain Transformer for Multivariate Time Series Forecasting | None
23-05-30|[HSTTN](https://arxiv.org/abs/2305.18724) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | IJCAI 2023 | Long-term Wind Power Forecasting with Hierarchical Spatial-Temporal Transformer | None
23-05-30|[Client](https://arxiv.org/abs/2305.18838) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | Client: Cross-variable Linear Integrated Enhanced Transformer for Multivariate Long-Term Time Series Forecasting | [Client](https://github.com/daxin007/client)
23-05-30|[Taylorformer](https://arxiv.org/abs/2305.19141) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | Taylorformer: Probabilistic Predictions for Time Series and other Processes | [Taylorformer](https://www.dropbox.com/s/vnxuwq7zm7m9bj8/taylorformer.zip?dl=0)
23-06-05|[Corrformer](https://www.nature.com/articles/s42256-023-00667-9)🌟 | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | NMI 2023 | [Interpretable weather forecasting for worldwide stations with a unified deep model](https://zhuanlan.zhihu.com/p/635902919) | [Corrformer](https://github.com/thuml/Corrformer)
23-06-14|[GCformer](https://arxiv.org/abs/2306.08325) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | GCformer: An Efficient Framework for Accurate and Scalable Long-Term Multivariate Time Series Forecasting | [GCformer](https://github.com/zyj-111/gcformer)
23-07-04 | [SageFormer](https://arxiv.org/abs/2307.01616) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | SageFormer: Series-Aware Graph-Enhanced Transformers for Multivariate Time Series Forecasting | None 
23-07-10 | [DifFormer](https://ieeexplore.ieee.org/abstract/document/10177239) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | TPAMI 2023 | DifFormer: Multi-Resolutional Differencing Transformer With Dynamic Ranging for Time Series Analysis | None
23-07-27 | [HUTFormer](https://arxiv.org/abs/2307.14596) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2023 | HUTFormer: Hierarchical U-Net Transformer for Long-Term Traffic Forecasting | None
23-08-07 | [DSformer](https://arxiv.org/abs/2308.03274) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | CIKM 2023 | DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction | None
23-08-09 | [SBT](https://arxiv.org/abs/2308.04637) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2023 | Sparse Binary Transformers for Multivariate Time Series Modeling | None
23-08-09 | [PETformer](https://arxiv.org/abs/2308.04791) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | PETformer: Long-term Time Series Forecasting via Placeholder-enhanced Transformer | None
23-10-02 | [TACTiS-2](https://browse.arxiv.org/abs/2310.01327)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series | None
23-10-03 | [PrACTiS](https://browse.arxiv.org/abs/2310.01720)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | PrACTiS: Perceiver-Attentional Copulas for Time Series | None
23-10-10 | [iTransformer](https://arxiv.org/abs/2310.06625)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | iTransformer: Inverted Transformers Are Effective for Time Series Forecasting | None


## RNN.
Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----|-----
17-03-21|[LSTNet](https://arxiv.org/abs/1703.07015)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |SIGIR 2018|[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://zhuanlan.zhihu.com/p/467944750)|[LSTNet](https://github.com/laiguokun/LSTNet) |
17-04-07|[DA-RNN](https://arxiv.org/abs/1704.02971)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |IJCAI 2017| A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction | [DARNN](https://github.com/sunfanyunn/DARNN) |
17-04-13|[DeepAR](https://arxiv.org/abs/1704.04110)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |IJoF 2019|[DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://zhuanlan.zhihu.com/p/542066911)|[DeepAR](https://github.com/brunoklein99/deepar) |
17-11-29|[MQRNN](https://arxiv.org/abs/1711.11053)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPSW 2017|A Multi-Horizon Quantile Recurrent Forecaster|[MQRNN](https://github.com/tianchen101/MQRNN) |
18-06-23|[mWDN](https://arxiv.org/abs/1806.08946)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |KDD 2018|Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis|[mWDN](https://github.com/yakouyang/Multilevel_Wavelet_Decomposition_Network_Pytorch) |
18-09-06|[MTNet](https://arxiv.org/abs/1809.02105)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |AAAI 2019| A Memory-Network Based Solution for Multivariate Time-Series Forecasting |[MTNet](https://github.com/Maple728/MTNet) |
19-05-28|[DF-Model](https://arxiv.org/abs/1905.12417)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICML 2019| Deep Factors for Forecasting | None |
19-07-18|[ESLSTM](https://www.sciencedirect.com/science/article/pii/S0169207019301153)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |IJoF 2020|A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting| None |
19-07-25|[MH-TAL](https://dl.acm.org/doi/abs/10.1145/3292500.3330662)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |KDD 2019|Multi-Horizon Time Series Forecasting with Temporal Attention Learning| None |
22-05-16|[C2FAR](https://openreview.net/forum?id=lHuPdoHBxbg)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPS 2022|[C2FAR: Coarse-to-Fine Autoregressive Networks for Precise Probabilistic Forecasting](https://zhuanlan.zhihu.com/p/600602517)|[C2FAR](https://github.com/huaweicloud/c2far_forecasting) |
23-06-02|[RNN-ODE-Adap](https://arxiv.org/abs/2306.01674)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2023|Neural Differential Recurrent Neural Network with Adaptive Time Steps| [RNN_ODE_Adap](https://github.com/Yixuan-Tan/RNN_ODE_Adap) |
23-08-22|[SegRNN](https://arxiv.org/abs/2308.11200)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |Arxiv 2023| SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting| None |


## MLP.
Date     | Method                                        |Type| Conference | Paper Title and Paper Interpretation (In Chinese)            | Code                                           |
| -------- | --------------------------------------------- |-----| ---------- | ------------------------------------------------------------ | ---------------------------------------------- |
| 19-05-24 | [NBeats](https://arxiv.org/abs/1905.10437)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | ICLR 2020 | [N-BEATS: Neural Basis Expansion Analysis For Interpretable Time Series Forecasting](https://zhuanlan.zhihu.com/p/572850227)      | [NBeats](https://github.com/philipperemy/n-beats) |
| 21-04-12 | [NBeatsX](https://arxiv.org/abs/2104.05522)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen)| IJoF 2022 | [Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx](https://zhuanlan.zhihu.com/p/572955881)      | [NBeatsX](https://github.com/cchallu/nbeatsx) |
| 22-01-30 | [N-HiTS](https://arxiv.org/abs/2201.12886)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | AAAI 2023 | [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://zhuanlan.zhihu.com/p/573203887)      | [N-HiTS](https://github.com/cchallu/n-hits) |
| 22-05-15 | [DEPTS](https://arxiv.org/abs/2203.07681)  | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | ICLR 2022 | [DEPTS: Deep Expansion Learning for Periodic Time Series Forecasting](https://zhuanlan.zhihu.com/p/572984932)      | [DEPTS](https://github.com/weifantt/depts) |
| 22-05-24 | [FreDo](https://arxiv.org/abs/2205.12301)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |Arxiv 2022|FreDo: Frequency Domain-based Long-Term Time Series Forecasting| None |
| 22-05-26 | [DLinear](https://arxiv.org/abs/2205.13504)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | AAAI 2023 | [Are Transformers Effective for Time Series Forecasting?](https://zhuanlan.zhihu.com/p/569194246)      | [DLinear](https://github.com/cure-lab/DLinear) |
| 22-06-24 | [TreeDRNet](https://arxiv.org/abs/2206.12106)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2022 | TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting | None                                           |
| 22-07-04 | [LightTS](https://arxiv.org/abs/2207.01186) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2022 | Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures | [LightTS](https://tinyurl.com/5993cmus)            |
| 22-08-10 | [STID](https://arxiv.org/abs/2208.05233) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | CIKM 2022 | Spatial-Temporal Identity: A Simple yet Effective Baseline for Multivariate Time Series Forecasting |  [STID](https://github.com/zezhishao/stid) |
| 23-01-30 | [SimST](https://arxiv.org/abs/2301.12603) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2023 | Do We Really Need Graph Neural Networks for Traffic Forecasting? | None |
| 23-02-09 | [MTS-Mixers](https://arxiv.org/abs/2302.04501)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [MTS-Mixers](https://github.com/plumprc/MTS-Mixers)    |
| 23-03-10 | [TSMixer](https://arxiv.org/abs/2303.06053)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | TSMixer: An all-MLP Architecture for Time Series Forecasting | None  |
| 23-04-17 | [TiDE](https://arxiv.org/abs/2304.08424)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://zhuanlan.zhihu.com/p/624828590) | [TiDE](https://zhuanlan.zhihu.com/p/624828590) |
| 23-05-18 | [RTSF](https://arxiv.org/abs/2305.10721) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping | [RTSF](https://github.com/plumprc/rtsf) |
| 23-05-30 | [Koopa](https://arxiv.org/abs/2305.18803)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2023 | [Koopa: Learning Non-stationary Time Series Dynamics with Koopman Predictors](https://zhuanlan.zhihu.com/p/635356173) | [Koopa](https://github.com/thuml/Koopa) |
| 23-06-14 | [CI-TSMixer](https://arxiv.org/abs/2306.09364) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2023  | TSMixer: Lightweight MLP-Mixer Model for Multivariate Time Series Forecasting | None |
| 23-07-06 | [FITS](https://arxiv.org/abs/2307.03756) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023  | FITS: Modeling Time Series with 10k Parameters | [FITS](https://anonymous.4open.science/r/FITS) |
| 23-08-14 | [ST-MLP](https://arxiv.org/abs/2308.07496) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2023  | ST-MLP: A Cascaded Spatio-Temporal Linear Framework with Channel-Independence Strategy for Traffic Forecasting | None |
| 23-08-25 | [TFDNet](https://arxiv.org/abs/2308.13386) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | TFDNet: Time-Frequency Enhanced Decomposed Network for Long-term Time Series Forecasting | None |


## TCN/CNN.
Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|----|-----|-----|-----
| 19-05-09 | [DeepGLO](https://arxiv.org/abs/1905.03806)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2019 | Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting| [deepglo](https://github.com/rajatsen91/deepglo)         |    
| 19-05-22 | [DSANet](https://dl.acm.org/doi/abs/10.1145/3357384.3358132) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | CIKM 2019 | DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting | [DSANet](https://github.com/bighuang624/DSANet)         |    
| 19-12-11 | [MLCNN](https://arxiv.org/abs/1912.05122) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2020 | Towards Better Forecasting by Fusing Near and Distant Future Visions | [MLCNN](https://github.com/smallGum/MLCNN-Multivariate-Time-Series)         |   
| 21-06-17 | [SCINet](https://arxiv.org/abs/2106.09305) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2022 | [SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction](https://mp.weixin.qq.com/s/mHleT4EunD82hmEfHnhkig) | [SCINet](https://github.com/cure-lab/SCINet)         |    
| 22-09-22 | [MICN](https://openreview.net/forum?id=zt53IDUR1U) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2023 | [MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting](https://zhuanlan.zhihu.com/p/603468264) | [MICN](https://github.com/whq13018258357/MICN)            |
| 22-09-22 | [TimesNet](https://arxiv.org/abs/2210.02186)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2023 | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://zhuanlan.zhihu.com/p/604100426) | [TimesNet](https://github.com/thuml/TimesNet)          |
| 23-02-23 | [LightCTS](https://arxiv.org/abs/2302.11974) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | SIGMOD 2023 | LightCTS: A Lightweight Framework for Correlated Time Series Forecasting | [LightCTS](https://github.com/ai4cts/lightcts)          |
| 23-05-25 | [TLNets](https://arxiv.org/abs/2305.15770) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | TLNets: Transformation Learning Networks for long-range time-series prediction | [TLNets](https://github.com/anonymity111222/tlnets)      |
| 23-06-04 | [Cross-LKTCN](https://arxiv.org/abs/2306.02326) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | Cross-LKTCN: Modern Convolution Utilizing Cross-Variable Dependency for Multivariate Time Series Forecasting Dependency for Multivariate Time Series Forecasting | None |
| 23-06-12 | [MPPN](https://arxiv.org/abs/2306.06895) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | MPPN: Multi-Resolution Periodic Pattern Network For Long-Term Time Series Forecasting | None |
| 23-06-19 | [FDNet](https://arxiv.org/abs/2306.10703) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KBS 2023 | FDNet: Focal Decomposed Network for Efficient, Robust and Practical Time Series Forecasting | [FDNet](https://github.com/OrigamiSL/FDNet-KBS-2023) |
| 23-10-01 | [PatchMixer](https://browse.arxiv.org/abs/2310.00655) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | PatchMixer: A Patch-Mixing Architecture for Long-Term Time Series Forecasting | None |


## GNN.
Date | Method | Type | Conference | Paper Title and Paper Interpretation (In Chinese) | Code |
| ---- | ------ | ------ | ---------- | ------------------------------------------------- | ---- |
| 17-09-14 | [STGCN](https://arxiv.org/abs/1709.04875)🌟 | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | IJCAI 2018 | Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting | [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18) |
| 19-05-31 | [Graph WaveNet](https://arxiv.org/abs/1906.00121) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | IJCAI 2019 | Graph WaveNet for Deep Spatial-Temporal Graph Modeling | [Graph-WaveNet](https://github.com/nnzhan/Graph-WaveNet) |
| 19-07-17 | [ASTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2019 | Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting | [ASTGCN](https://github.com/guoshnBJTU/ASTGCN-r-pytorch) |
| 20-04-03 | [SLCNN](https://ojs.aaai.org/index.php/AAAI/article/view/5470) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2020 | Spatio-Temporal Graph Structure Learning for Traffic Forecasting | None |
| 20-04-03 | [GMAN](https://ojs.aaai.org/index.php/AAAI/article/view/5477) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2020 | GMAN: A Graph Multi-Attention Network for Traffic Prediction | [GMAN](https://github.com/zhengchuanpan/GMAN) |
| 20-05-03 | [MTGNN](https://arxiv.org/abs/2005.01165)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2020 | Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks | [MTGNN](https://github.com/nnzhan/MTGNN)  |
| 21-03-13 | [StemGNN](https://arxiv.org/abs/2103.07719)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2020 | Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting | [StemGNN](https://github.com/microsoft/StemGNN) |
| 22-05-16 | [TPGNN](https://openreview.net/forum?id=pMumil2EJh) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2022 | Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks | [TPGNN](https://github.com/zyplanet/TPGNN) |
| 22-06-18 | [D2STGNN](https://arxiv.org/abs/2206.09112) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | VLDB 2022 | Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting | [D2STGNN](https://github.com/zezhishao/d2stgnn) |  
| 23-07-10 | [NexuSQN](https://arxiv.org/abs/2307.01482) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2023 | Nexus sine qua non: Essentially connected neural networks for spatial-temporal forecasting of multivariate time series | None |


## SSM (State Space Model).
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 18-05-18 | [DSSM](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting) | NIPS 2018 | Deep State Space Models for Time Series Forecasting | None   |
| 19-08-10 | [DSSMF](https://arxiv.org/abs/2102.00397) | IJCAI 2019 | Learning Interpretable Deep State Space Model for Probabilistic Time Series Forecasting | None   |
| 22-08-19 | [SSSD](https://arxiv.org/abs/2208.09399) | TMLR 2022 | Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models | [SSSD](https://github.com/AI4HealthUOL/SSSD) |
| 22-09-22 | [SpaceTime](https://arxiv.org/abs/2303.09489) | ICLR 2023 | Effectively Modeling Time Series with Simple Discrete State Spaces | [SpaceTime](https://github.com/hazyresearch/spacetime)   |
| 22-12-24 | [LS4](https://arxiv.org/abs/2212.12749) | ICML 2023 | Deep Latent State Space Models for Time-Series Generation | None   |


## Generation Model.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 20-02-14 | [MAF](https://arxiv.org/abs/2002.06103)🌟 | ICLR 2021 | [Multivariate Probabilitic Time Series Forecasting via Conditioned Normalizing Flows](https://zhuanlan.zhihu.com/p/615795048) | [MAF](https://github.com/zalandoresearch/pytorch-ts)   |
| 21-01-18 | [TimeGrad](https://arxiv.org/abs/2101.12072)🌟 | ICML 2021 | [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](https://zhuanlan.zhihu.com/p/615858953) | [TimeGrad](https://github.com/zalandoresearch/pytorch-ts)   |
| 21-07-07 | [CSDI](https://arxiv.org/abs/2107.03502) | NIPS 2021 | [CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://zhuanlan.zhihu.com/p/615998087) | [CSDI](https://github.com/ermongroup/csdi)   |
| 22-05-16 | [MANF](https://arxiv.org/abs/2205.07493)|Arxiv 2022|Multi-scale Attention Flow for Probabilistic Time Series Forecasting| None |
| 22-05-16 | [D3VAE](https://arxiv.org/abs/2301.03028) | NIPS 2022 | Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement | [D3VAE](https://github.com/paddlepaddle/paddlespatial)   |
| 22-05-16 | [LaST](https://openreview.net/pdf?id=C9yUwd72yy) | NIPS 2022 | LaST: Learning Latent Seasonal-Trend Representations for Time Series Forecasting | [LaST](https://github.com/zhycs/LaST)   |
| 22-12-28 | [Hier-Transformer-CNF](https://arxiv.org/abs/2212.13706) | Arxiv 2022 | End-to-End Modeling Hierarchical Time Series Using Autoregressive Transformer and Conditional Normalizing Flow based Reconciliation | None   |
| 23-03-13 | [HyVAE](https://arxiv.org/abs/2303.07048) | Arxiv 2023 | Hybrid Variational Autoencoder for Time Series Forecasting | None   |
| 23-06-05 | [WIAE](https://arxiv.org/abs/2306.03782) | Arxiv 2023 | Non-parametric Probabilistic Time Series Forecasting via Innovations Representation | None   |
| 23-06-08 | [TimeDiff](https://arxiv.org/abs/2306.05043)🌟 | ICML 2023 | Non-autoregressive Conditional Diffusion Models for Time Series Prediction | None |
| 23-07-21 | [TSDiff](https://arxiv.org/abs/2307.11494) | Arxiv 2023 | Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting | None |


## Time-index.
Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|----|-----|-----|-----
| 17-05-25 | [ND](https://arxiv.org/abs/1705.09137) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | TNNLS 2017 | [Neural Decomposition of Time-Series Data for Effective Generalization](https://zhuanlan.zhihu.com/p/574742701)  | None |
| 17-08-25 | [Prophet](https://peerj.com/preprints/3190/)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen)  | TAS 2018 | [Forecasting at Scale](https://facebook.github.io/prophet/) | [Prophet](https://github.com/facebook/prophet)   |
| 22-07-13 | [DeepTime](https://arxiv.org/abs/2207.06046) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICML 2023 | [Learning Deep Time-index Models for Time Series Forecasting](https://blog.salesforceairesearch.com/deeptime-meta-learning-time-series-forecasting/) | [DeepTime](https://github.com/salesforce/DeepTime) |
| 23-06-09 | [TimeFlow](https://arxiv.org/abs/2306.05880) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen)  | Arxiv 2023 | Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations | None |


## Plug and Play (Model-Agnostic).
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 19-02-21 | [DAIN](https://arxiv.org/abs/1902.07892)🌟 | TNNLS 2020 | Deep Adaptive Input Normalization for Time Series Forecasting | [DAIN](https://github.com/passalis/dain)   |
| 19-09-19 | [DILATE](https://arxiv.org/abs/1909.09020) | NIPS 2019 | Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models | [DILATE](https://github.com/vincent-leguen/DILATE) |
| 21-07-19 | [TAN](https://arxiv.org/abs/2107.09031) | NIPS 2021 | Topological Attention for Time Series Forecasting | [TAN](https://github.com/plus-rkwitt/TAN)   |
| 21-09-29 | [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p)🌟 | ICLR 2022 | [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://zhuanlan.zhihu.com/p/473553126) | [RevIN](https://github.com/ts-kim/RevIN)   |
| 22-02-23 | [MQF2](https://arxiv.org/abs/2202.11316) | AISTATS 2022 | Multivariate Quantile Function Forecaster | None   |
| 22-05-18 | [FiLM](https://arxiv.org/abs/2205.08897) |NIPS 2022|FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting | [FiLM](https://github.com/tianzhou2011/FiLM) |
| 23-02-18 | [FrAug](https://arxiv.org/abs/2302.09292) | Arxiv 2023 | FrAug: Frequency Domain Augmentation for Time Series Forecasting | [FrAug](https://anonymous.4open.science/r/Fraug-more-results-1785)   |
| 23-02-22 | [Dish-TS](https://arxiv.org/abs/2302.14829) | AAAI 2023 | [Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting](https://zhuanlan.zhihu.com/p/613566978) | [Dish-TS](https://github.com/weifantt/Dish-TS)   |
| 23-02-23 | [Adaptive Sampling](https://arxiv.org/abs/2302.11870) | NIPSW 2022 | Adaptive Sampling for Probabilistic Forecasting under Distribution Shift | None   |
| 23-04-19 | [RoR](https://arxiv.org/abs/2304.09836) | ICML 2023 | Regions of Reliability in the Evaluation of Multivariate Probabilistic Forecasts | [RoR](https://github.com/ServiceNow/regions-of-reliability) |
| 23-05-26 | [BetterBatch](https://arxiv.org/abs/2305.17028) | Arxiv 2023 | Better Batch for Deep Probabilistic Time Series Forecasting | None |
| 23-05-28 | [PALS](https://arxiv.org/abs/2305.18382) | Arxiv 2023 | Adaptive Sparsity Level during Training for Efficient Time Series Forecasting with Transformers | None |
| 23-06-09 | [FeatureProgramming](https://arxiv.org/abs/2306.06252) | ICML 2023 | Feature Programming for Multivariate Time Series Prediction | [FeatureProgramming](https://github.com/SirAlex900/FeatureProgramming) |
| 23-07-18 | [Look_Ahead](https://dl.acm.org/doi/abs/10.1145/3539618.3592013) | SIGIR 2023 | Look Ahead: Improving the Accuracy of Time-Series Forecasting by Previewing Future Time Features | [Look_Ahead](https://github.com/sunsunmin/Look_Ahead) |
| 23-09-14 | [QFCV](https://arxiv.org/abs/2309.07435) | Arxiv 2023 | Uncertainty Intervals for Prediction Errors in Time Series Forecasting | [QFCV](https://github.com/huixu18/qfcv) |
| 23-10-09 | [PeTS](https://arxiv.org/abs/2310.06077) | Arxiv 2023 | Performative Time-Series Forecasting | [PeTS](https://github.com/adityalab/pets) |

## Pretrain & Representation.
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 20-10-06 | [TST](https://arxiv.org/abs/2010.02803) | KDD 2021 | A Transformer-based Framework for Multivariate Time Series Representation Learning | [mvts_transformer](https://github.com/gzerveas/mvts_transformer)   |
| 22-06-18 | [STEP](https://arxiv.org/abs/2206.09113) | KDD 2022 | Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting | [STEP](https://github.com/zezhishao/step)   |
| 23-02-23 | [FPT](https://arxiv.org/abs/2302.11939) | NIPS 2023 | One Fits All:Power General Time Series Analysis by Pretrained LM | [One-Fits-All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)   |
| 23-05-17 | (LLMTime)[https://arxiv.org/abs/2310.07820] | NIPS 2023 | Large Language Models Are Zero-Shot Time Series Forecasters | [LLMTime](https://github.com/ngruver/llmtime) |
| 23-08-02 | [Floss](https://arxiv.org/abs/2308.01011) | Arxiv 2023 | Enhancing Representation Learning for Periodic Time Series with Floss: A Frequency Domain Regularization Approach | [floss](https://github.com/agustdd/floss) |
| 23-08-16 | [TEST](https://arxiv.org/abs/2308.08241) | Arxiv 2023 | TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series | None |
| 23-08-16 | [LLM4TS](https://arxiv.org/abs/2308.08469) | Arxiv 2023 | LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs | None |
| 23-10-03 | [Time-LLM](https://arxiv.org/abs/2310.01728) | Arxiv 2023 | Time-LLM: Time Series Forecasting by Reprogramming Large Language Models | None |
| 23-10-08 | [TEMPO](https://arxiv.org/abs/2310.04948) | Arxiv 2023 | TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting | None |
| 23-10-12 | [Lag-Llama](https://arxiv.org/abs/2310.08278) | Arxiv 2023 | Lag-Llama: Towards Foundation Models for Time Series Forecasting | [Lag-Llama](https://github.com/kashif/pytorch-transformer-ts) |


## Online.
Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----|-----
| 22-02-23 | [FSNet](https://openreview.net/pdf?id=q-PbpHD3EOk) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2023 | Learning Fast and Slow for Online Time Series Forecasting | [FSNet](https://github.com/salesforce/fsnet)   |
| 23-09-22 | [OneNet](https://arxiv.org/abs/2309.12659) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2023 | OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling | [OneNet](https://github.com/yfzhang114/OneNet)   |
| 23-09-25 | [MemDA](https://arxiv.org/abs/2309.14216) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | CIKM 2023 | MemDA: Forecasting Urban Time Series with Memory-based Drift Adaptation |  None  |

## Theory.
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 22-10-25 | [WaveBound](https://arxiv.org/abs/2210.14303) | NIPS 2022 | WaveBound: Dynamic Error Bounds for Stable Time Series Forecasting | [WaveBound](https://github.com/choyi0521/WaveBound) |
| 23-05-25 | [Ensembling](https://arxiv.org/abs/2305.15786) | ICML 2023 | Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting | None |


## Other.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 16-12-05 | [TRMF](https://proceedings.neurips.cc/paper_files/paper/2016/file/85422afb467e9456013a2a51d4dff702-Paper.pdf) | NIPS 2016 | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction | [TRMF](https://github.com/rofuyu/exp-trmf-nips16)   |
