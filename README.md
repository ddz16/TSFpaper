# TSF Paper
This repository contains a reading list of papers on time series forecasting/prediction (TSF). These papers are mainly categorized according to the type of model.


## Survey.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
15-11-23|[Multi-step](https://ieeexplore.ieee.org/abstract/document/7422387)|ACOMP 2015|Comparison of Strategies for Multi-step-Ahead Prediction of Time Series Using Neural Network|None
19-06-20|[DL](https://ieeexplore.ieee.org/abstract/document/8742529)| SENSJ 2019|A Review of Deep Learning Models for Time Series Prediction|None
20-09-27|[DL](https://arxiv.org/abs/2004.13408)|Arxiv 2020|Time Series Forecasting With Deep Learning: A Survey|None
22-02-15|[Transformer](https://arxiv.org/abs/2202.07125)|Arxiv 2022|Transformers in Time Series: A Survey|None
23-05-01|[Diffusion](https://arxiv.org/abs/2305.00624)|Arxiv 2023|Diffusion Models for Time Series Applications: A Survey|None


## Transformer.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
19-06-29|[LogTrans](https://arxiv.org/abs/1907.00235)|NIPS 2019|Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting|[flowforecast](https://github.com/AIStream-Peelout/flow-forecast/blob/master/flood_forecast/transformer_xl/transformer_bottleneck.py) |
19-12-19|[TFT](https://arxiv.org/abs/1912.09363)|IJoF 2021|[Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://www.zhihu.com/question/451816360/answer/2319401126)|[tft](https://github.com/google-research/google-research/tree/master/tft) |
20-01-23|[InfluTrans](https://arxiv.org/abs/2001.08317)|Arxiv 2020|[Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case](https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e)|[influenza_transformer](https://github.com/KasperGroesLudvigsen/influenza_transformer) |
20-06-05|[AST](https://proceedings.neurips.cc/paper/2020/file/c6b8c8d762da15fa8dbbdfb6baf9e260-Paper.pdf)|NIPS 2020|Adversarial Sparse Transformer for Time Series Forecasting|[AST](https://github.com/hihihihiwsf/AST)
20-12-14|[Informer](https://arxiv.org/abs/2012.07436)|AAAI 2021|[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://zhuanlan.zhihu.com/p/467523291)|[Informer](https://github.com/zhouhaoyi/Informer2020)
21-05-22|[ProTran](https://proceedings.neurips.cc/paper_files/paper/2021/file/c68bd9055776bf38d8fc43c0ed283678-Paper.pdf)|NIPS 2021|Probabilistic Transformer for Time Series Analysis|None
21-06-24|[Autoformer](https://arxiv.org/abs/2106.13008)|NIPS 2021|[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://zhuanlan.zhihu.com/p/385066440)|[Autoformer](https://github.com/thuml/Autoformer)
21-10-05|[Pyraformer](https://openreview.net/pdf?id=0EXmFzUn5I)|ICLR 2022|[Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting](https://zhuanlan.zhihu.com/p/467765457)|[Pyraformer](https://github.com/alipay/Pyraformer)
22-01-14|[Preformer](https://arxiv.org/abs/2202.11356)|ICASSP 2023|[Preformer: Predictive Transformer with Multi-Scale Segment-wise Correlations for Long-Term Time Series Forecasting](https://zhuanlan.zhihu.com/p/536398013)|[Preformer](https://github.com/ddz16/Preformer)
22-01-30|[FEDformer](https://arxiv.org/abs/2201.12740)|ICML 2022|[FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://zhuanlan.zhihu.com/p/528131016)|[FEDformer](https://github.com/MAZiqing/FEDformer)
22-02-03|[ETSformer](https://arxiv.org/abs/2202.01381)|Arxiv 2022|[ETSformer: Exponential Smoothing Transformers for Time-series Forecasting](https://blog.salesforceairesearch.com/etsformer-time-series-forecasting/)|[etsformer](https://github.com/salesforce/etsformer)
22-02-07|[TACTiS](https://arxiv.org/abs/2202.03528)|ICML 2022|TACTiS: Transformer-Attentional Copulas for Time Series|[TACTiS](https://github.com/ServiceNow/tactis)
22-04-28|[Triformer](https://arxiv.org/abs/2204.13767)|IJCAI 2022|[Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting](https://blog.csdn.net/zj_18706809267/article/details/125048492)| [Triformer](https://github.com/razvanc92/triformer)
22-05-27|[TDformer](https://arxiv.org/abs/2212.08151)|NIPSW 2022|[First De-Trend then Attend: Rethinking Attention for Time-Series Forecasting](https://zhuanlan.zhihu.com/p/596022160)|[TDformer](https://github.com/BeBeYourLove/TDformer)
22-05-28|[Non-stationary Transformer](https://arxiv.org/abs/2205.14415)|NIPS 2022|[Non-stationary Transformers: Rethinking the Stationarity in Time Series Forecasting](https://zhuanlan.zhihu.com/p/535931701)|[Non-stationary Transformers](https://github.com/thuml/Nonstationary_Transformers)
22-06-08|[Scaleformer](https://arxiv.org/abs/2206.04038)|ICLR 2023|[Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting](https://zhuanlan.zhihu.com/p/535556231)|[Scaleformer](https://github.com/BorealisAI/scaleformer)
22-08-14|[Quatformer](https://dl.acm.org/doi/abs/10.1145/3534678.3539234)|KDD 2022|Learning to Rotate: Quaternion Transformer for Complicated Periodical Time Series Forecasting|[Quatformer](https://github.com/DAMO-DI-ML/KDD2022-Quatformer)
22-08-30|[Persistence Initialization](https://arxiv.org/abs/2208.14236)|Arxiv 2022|[Persistence Initialization: A novel adaptation of the Transformer architecture for Time Series Forecasting](https://zhuanlan.zhihu.com/p/582419707)|None
22-09-08|[W-Transformers](https://arxiv.org/abs/2209.03945)|Arxiv 2022|[W-Transformers: A Wavelet-based Transformer Framework for Univariate Time Series Forecasting](https://zhuanlan.zhihu.com/p/582419707)|[w-transformer](https://github.com/capwidow/w-transformer)
22-09-22|[Crossformer](https://openreview.net/forum?id=vSVLM2j9eie)|ICLR 2023|Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting|[Crossformer](https://github.com/Thinklab-SJTU/Crossformer)
22-09-22|[PatchTST](https://arxiv.org/abs/2211.14730)|ICLR 2023|[A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://zhuanlan.zhihu.com/p/602332939)|[PatchTST](https://github.com/yuqinie98/patchtst)
23-05-20|[CARD](https://arxiv.org/abs/2305.12095)|Arxiv 2023|Make Transformer Great Again for Time Series Forecasting: Channel Aligned Robust Dual Transformer|None
23-05-24|[JTFT](https://arxiv.org/abs/2305.14649) | Arxiv 2023 | A Joint Time-frequency Domain Transformer for Multivariate Time Series Forecasting | None
23-05-30|[HSTTN](https://arxiv.org/abs/2305.18724) | IJCAI 2023 | Long-term Wind Power Forecasting with Hierarchical Spatial-Temporal Transformer | None
23-05-30|[Client](https://arxiv.org/abs/2305.18838) | Arxiv 2023 | Client: Cross-variable Linear Integrated Enhanced Transformer for Multivariate Long-Term Time Series Forecasting | [Client](https://github.com/daxin007/client)


## RNN.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
17-03-21|[LSTNet](https://arxiv.org/abs/1703.07015)|SIGIR 2018|[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://zhuanlan.zhihu.com/p/467944750)|[LSTNet](https://github.com/laiguokun/LSTNet) |
17-04-07|[DA-RNN](https://arxiv.org/abs/1704.02971)|IJCAI 2017| A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction | [DARNN](https://github.com/sunfanyunn/DARNN) |
17-04-13|[DeepAR](https://arxiv.org/abs/1704.04110)|IJoF 2019|[DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://zhuanlan.zhihu.com/p/542066911)|[DeepAR](https://github.com/brunoklein99/deepar) |
17-11-29|[MQRNN](https://arxiv.org/abs/1711.11053)|NIPSW 2017|A Multi-Horizon Quantile Recurrent Forecaster|[MQRNN](https://github.com/tianchen101/MQRNN) |
18-06-23|[mWDN](https://arxiv.org/abs/1806.08946)|KDD 2018|Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis|[mWDN](https://github.com/yakouyang/Multilevel_Wavelet_Decomposition_Network_Pytorch) |
18-09-06|[MTNet](https://arxiv.org/abs/1809.02105)|AAAI 2019| A Memory-Network Based Solution for Multivariate Time-Series Forecasting |[MTNet](https://github.com/Maple728/MTNet) |
19-05-28|[DF-Model](https://arxiv.org/abs/1905.12417)|ICML 2019| Deep Factors for Forecasting | None |
19-07-01|[MH-RNN](https://dl.acm.org/doi/abs/10.1145/3292500.3330662)|KDD 2019| Multi-Horizon Time Series Forecasting with Temporal Attention Learning | None |
19-07-18|[ESLSTM](https://www.sciencedirect.com/science/article/pii/S0169207019301153)|IJoF 2020|A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting| None |
19-07-25|[MH-TAL](https://dl.acm.org/doi/abs/10.1145/3292500.3330662)|KDD 2019|Multi-Horizon Time Series Forecasting with Temporal Attention Learning| None |
22-05-16|[C2FAR](https://openreview.net/forum?id=lHuPdoHBxbg)|NIPS 2022|[C2FAR: Coarse-to-Fine Autoregressive Networks for Precise Probabilistic Forecasting](https://zhuanlan.zhihu.com/p/600602517)|[C2FAR](https://github.com/huaweicloud/c2far_forecasting) |


## MLP.

| Date     | Method                                        | Conference | Paper Title and Paper Interpretation (In Chinese)            | Code                                           |
| -------- | --------------------------------------------- | ---------- | ------------------------------------------------------------ | ---------------------------------------------- |
| 17-05-25 | [ND](https://arxiv.org/abs/1705.09137)   | TNNLS 2017 | [Neural Decomposition of Time-Series Data for Effective Generalization](https://zhuanlan.zhihu.com/p/574742701)  | None |
| 19-05-24 | [NBeats](https://arxiv.org/abs/1905.10437)   | ICLR 2020 | [N-BEATS: Neural Basis Expansion Analysis For Interpretable Time Series Forecasting](https://zhuanlan.zhihu.com/p/572850227)      | [NBeats](https://github.com/philipperemy/n-beats) |
| 21-04-12 | [NBeatsX](https://arxiv.org/abs/2104.05522)   | IJoF 2022 | [Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx](https://zhuanlan.zhihu.com/p/572955881)      | [NBeatsX](https://github.com/cchallu/nbeatsx) |
| 22-01-30 | [N-HiTS](https://arxiv.org/abs/2201.12886)   | AAAI 2023 | [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://zhuanlan.zhihu.com/p/573203887)      | [N-HiTS](https://github.com/cchallu/n-hits) |
| 22-05-15 | [DEPTS](https://arxiv.org/abs/2203.07681)   | ICLR 2022 | [DEPTS: Deep Expansion Learning for Periodic Time Series Forecasting](https://zhuanlan.zhihu.com/p/572984932)      | [DEPTS](https://github.com/weifantt/depts) |
| 22-05-24 | [FreDo](https://arxiv.org/abs/2205.12301)|Arxiv 2022|FreDo: Frequency Domain-based Long-Term Time Series Forecasting| None |
| 22-05-26 | [DLinear](https://arxiv.org/abs/2205.13504)   | AAAI 2023 | [Are Transformers Effective for Time Series Forecasting?](https://zhuanlan.zhihu.com/p/569194246)      | [DLinear](https://github.com/cure-lab/DLinear) |
| 22-06-24 | [TreeDRNet](https://arxiv.org/abs/2206.12106) | Arxiv 2022 | TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting | None                                           |
| 22-07-04 | [LightTS](https://arxiv.org/abs/2207.01186) | Arxiv 2022 | Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures | [LightTS](https://tinyurl.com/5993cmus)            |
| 23-02-09 | [MTS-Mixers](https://arxiv.org/abs/2302.04501) | Arxiv 2023 | MTS-Mixers: Multivariate Time Series Forecasting via Factorized Temporal and Channel Mixing | [MTS-Mixers](https://github.com/plumprc/MTS-Mixers)    |
| 23-03-10 | [TSMixer](https://arxiv.org/abs/2303.06053) | Arxiv 2023 | TSMixer: An all-MLP Architecture for Time Series Forecasting | None  |
| 23-04-17 | [TiDE](https://arxiv.org/abs/2304.08424) | Arxiv 2023 | [Long-term Forecasting with TiDE: Time-series Dense Encoder](https://zhuanlan.zhihu.com/p/624828590) | [TiDE](https://zhuanlan.zhihu.com/p/624828590) |
| 23-05-18 | [RTSF](https://arxiv.org/abs/2305.10721) | Arxiv 2023 | Revisiting Long-term Time Series Forecasting: An Investigation on Linear Mapping | [RTSF](https://github.com/plumprc/rtsf) |


## TCN.

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 19-05-09 | [DeepGLO](https://arxiv.org/abs/1905.03806) | NIPS 2019 | Think Globally, Act Locally: A Deep Neural Network Approach to High-Dimensional Time Series Forecasting| [deepglo](https://github.com/rajatsen91/deepglo)         |    
| 19-05-22 | [DSANet](https://dl.acm.org/doi/abs/10.1145/3357384.3358132) | CIKM 2019 | DSANet: Dual Self-Attention Network for Multivariate Time Series Forecasting | [DSANet](https://github.com/bighuang624/DSANet)         |    
| 19-12-11 | [MLCNN](https://arxiv.org/abs/1912.05122) | AAAI 2020 | Towards Better Forecasting by Fusing Near and Distant Future Visions | [MLCNN](https://github.com/smallGum/MLCNN-Multivariate-Time-Series)         |   
| 21-06-17 | [SCINet](https://arxiv.org/abs/2106.09305) | NIPS 2022 | [SCINet: Time Series Modeling and Forecasting with Sample Convolution and Interaction](https://mp.weixin.qq.com/s/mHleT4EunD82hmEfHnhkig) | [SCINet](https://github.com/cure-lab/SCINet)         |    
| 22-09-22 | [MICN](https://openreview.net/forum?id=zt53IDUR1U) | ICLR 2023 | [MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting](https://zhuanlan.zhihu.com/p/603468264) | [MICN](https://github.com/whq13018258357/MICN)            |
| 22-09-22 | [TimesNet](https://arxiv.org/abs/2210.02186) | ICLR 2023 | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://zhuanlan.zhihu.com/p/604100426) | [TimesNet](https://github.com/thuml/TimesNet)          |
| 23-02-23 | [LightCTS](https://arxiv.org/abs/2302.11974) | SIGMOD 2023 | LightCTS: A Lightweight Framework for Correlated Time Series Forecasting | [LightCTS](https://github.com/ai4cts/lightcts)          |
| 23-05-25 | [TLNets](https://arxiv.org/abs/2305.15770) | Arxiv 2023 | TLNets: Transformation Learning Networks for long-range time-series prediction | [TLNets](https://github.com/anonymity111222/tlnets)      |
| 23-06-04 | [Cross-LKTCN](https://arxiv.org/abs/2306.02326) | Arxiv 2023 | Cross-LKTCN: Modern Convolution Utilizing Cross-Variable Dependency for Multivariate Time Series Forecasting Dependency for Multivariate Time Series Forecasting | None |


## SSM (State Space Model)
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 18-05-18 | [DSSM](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting) | NIPS 2018 | Deep State Space Models for Time Series Forecasting | None   |
| 22-08-19 | [SSSD](https://arxiv.org/abs/2208.09399) | TMLR 2022 | Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models | [SSSD](https://github.com/AI4HealthUOL/SSSD) |
| 22-09-22 | [SpaceTime](https://arxiv.org/abs/2303.09489) | ICLR 2023 | Effectively Modeling Time Series with Simple Discrete State Spaces | [SpaceTime](https://github.com/hazyresearch/spacetime)   |
| 22-12-24 | [LS4](https://arxiv.org/abs/2212.12749) | Arxiv 2022 | Deep Latent State Space Models for Time-Series Generation | None   |

## GNN (Spatio-Temporal Modeling).

| Date | Method | Conference | Paper Title and Paper Interpretation (In Chinese) | Code |
| ---- | ------ | ---------- | ------------------------------------------------- | ---- |
|      |        |            |                                                   |      |

## Generation Model.

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 20-02-14 | [MAF](https://arxiv.org/abs/2002.06103) | ICLR 2021 | [Multivariate Probabilitic Time Series Forecasting via Conditioned Normalizing Flows](https://zhuanlan.zhihu.com/p/615795048) | [MAF](https://github.com/zalandoresearch/pytorch-ts)   |
| 21-01-18 | [TimeGrad](https://arxiv.org/abs/2101.12072) | ICML 2021 | [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](https://zhuanlan.zhihu.com/p/615858953) | [TimeGrad](https://github.com/zalandoresearch/pytorch-ts)   |
| 21-07-07 | [CSDI](https://arxiv.org/abs/2107.03502) | NIPS 2021 | [CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://zhuanlan.zhihu.com/p/615998087) | [CSDI](https://github.com/ermongroup/csdi)   |
| 22-05-16|[MANF](https://arxiv.org/abs/2205.07493)|Arxiv 2022|Multi-scale Attention Flow for Probabilistic Time Series Forecasting| None |
| 22-05-16 | [D3VAE](https://arxiv.org/abs/2301.03028) | NIPS 2022 | Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement | [D3VAE](https://github.com/paddlepaddle/paddlespatial)   |
| 22-05-16 | [LaST](https://openreview.net/pdf?id=C9yUwd72yy) | NIPS 2022 | LaST: Learning Latent Seasonal-Trend Representations for Time Series Forecasting | [LaST](https://github.com/zhycs/LaST)   |
| 22-12-28 | [Hier-Transformer-CNF](https://arxiv.org/abs/2212.13706) | Arxiv 2022 | End-to-End Modeling Hierarchical Time Series Using Autoregressive Transformer and Conditional Normalizing Flow based Reconciliation | None   |
| 23-03-13 | [HyVAE](https://arxiv.org/abs/2303.07048) | Arxiv 2023 | Hybrid Variational Autoencoder for Time Series Forecasting | None   |



## Plug and Play (Model-Agnostic).
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 19-02-21 | [DAIN](https://arxiv.org/abs/1902.07892) | TNNLS 2020 | Deep Adaptive Input Normalization for Time Series Forecasting | [DAIN](https://github.com/passalis/dain)   |
| 19-09-19 | [DILATE](https://arxiv.org/abs/1909.09020) | NIPS 2019 | Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models | [DILATE](https://github.com/vincent-leguen/DILATE) |
| 21-07-19 | [TAN](https://arxiv.org/abs/2107.09031) | NIPS 2021 | Topological Attention for Time Series Forecasting | [TAN](https://github.com/plus-rkwitt/TAN)   |
| 21-09-29 | [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p) | ICLR 2022 | [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://zhuanlan.zhihu.com/p/473553126) | [RevIN](https://github.com/ts-kim/RevIN)   |
| 22-02-23 | [MQF2](https://arxiv.org/abs/2202.11316) | AISTATS 2022 | Multivariate Quantile Function Forecaster | None   |
| 22-05-18 |[FiLM](https://arxiv.org/abs/2205.08897)|NIPS 2022|FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting | [FiLM](https://github.com/tianzhou2011/FiLM) |
| 23-02-18 | [FrAug](https://arxiv.org/abs/2302.09292) | Arxiv 2023 | FrAug: Frequency Domain Augmentation for Time Series Forecasting | [FrAug](https://anonymous.4open.science/r/Fraug-more-results-1785)   |
| 23-02-22 | [Dish-TS](https://arxiv.org/abs/2302.14829) | AAAI 2023 | [Dish-TS: A General Paradigm for Alleviating Distribution Shift in Time Series Forecasting](https://zhuanlan.zhihu.com/p/613566978) | [Dish-TS](https://github.com/weifantt/Dish-TS)   |
| 23-02-23 | [Adaptive Sampling](https://arxiv.org/abs/2302.11870) | NIPSW 2022 | Adaptive Sampling for Probabilistic Forecasting under Distribution Shift | None   |


## Pretrain & Representation.
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 23-02-23 | [FPT](https://arxiv.org/abs/2302.11939) | Arxiv 2023 | Power Time Series Forecasting by Pretrained LM | [FPT](https://anonymous.4open.science/r/Pretrained-LM-for-TSForcasting-C561)   |


## Theory.
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 22-10-25 | [WaveBound](https://arxiv.org/abs/2210.14303) | NIPS 2022 | WaveBound: Dynamic Error Bounds for Stable Time Series Forecasting | None |
| 23-05-25 | [Ensembling](https://arxiv.org/abs/2305.15786) | ICML 2023 | Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting | None |


## Other.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 16-12-05 | [TRMF](https://proceedings.neurips.cc/paper_files/paper/2016/file/85422afb467e9456013a2a51d4dff702-Paper.pdf) | NIPS 2016 | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction | [TRMF](https://github.com/rofuyu/exp-trmf-nips16)   |
| 17-08-25 | [Prophet](https://peerj.com/preprints/3190/) | TAS 2018 | [Forecasting at Scale](https://facebook.github.io/prophet/) | [Prophet](https://github.com/facebook/prophet)   |
