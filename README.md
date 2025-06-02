# Awesome Time Series Forecasting/Prediction Papers
[![Awesome](https://awesome.re/badge.svg)](https://awesome.re) 
![PRs Welcome](https://img.shields.io/badge/PRs-Welcome-green) 
![Stars](https://img.shields.io/github/stars/ddz16/TSFpaper)

This repository contains a reading list of papers (**400+ papers !!!**) on **Time Series Forecasting/Prediction (TSF)** and **Spatio-Temporal Forecasting/Prediction (STF)**. These papers are mainly categorized according to the type of model. **This repository is still being continuously improved. In addition to papers that have been accepted by top conferences or journals, the repository also includes the latest papers from [arXiv](https://arxiv.org/). If you have found any relevant papers that need to be included in this repository, please feel free to submit a pull request (PR) or open an issue.** If you find this repository useful, please give it a 🌟. 

Each paper may apply to one or several types of forecasting, including univariate time series forecasting, multivariate time series forecasting, and spatio-temporal forecasting, which are also marked in the Type column. **If covariates and exogenous variables are not considered**, univariate time series forecasting involves predicting the future of one variable with the history of this variable, while multivariate time series forecasting involves predicting the future of C variables with the history of C variables. **Note that repeating univariate forecasting multiple times can also achieve the goal of multivariate forecasting, which is called _channel-independent_. However, univariate forecasting methods cannot extract relationships between variables, so the basis for distinguishing between univariate and multivariate forecasting methods is whether the method involves interaction between variables. Besides, in the era of deep learning, many univariate models can be easily modified to directly process multiple variables for multivariate forecasting. And multivariate models generally can be directly used for univariate forecasting. Here we classify solely based on the model's description in the original paper.** Spatio-temporal forecasting is often used in traffic and weather forecasting, and it adds a spatial dimension compared to univariate and multivariate forecasting. **In spatio-temporal forecasting, if each measurement point has only one variable, it is equivalent to multivariate forecasting. Therefore, the distinction between spatio-temporal forecasting and multivariate forecasting is not clear. Spatio-temporal models can usually be directly applied to multivariate forecasting, and multivariate models can also be used for spatio-temporal forecasting with minor modifications. Here we also classify solely based on the model's description in the original paper.**

* ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) univariate time series forecasting: ![](https://latex.codecogs.com/svg.image?\inline&space;L_1\times&space;1&space;\to&space;L_2\times&space;1), where ![](https://latex.codecogs.com/svg.image?\inline&space;L_1) is the history length, ![](https://latex.codecogs.com/svg.image?\inline&space;L_2) is the prediction horizon length.
* ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) multivariate time series forecasting: ![](https://latex.codecogs.com/svg.image?\inline&space;L_1\times&space;C&space;\to&space;L_2\times&space;C), where ![](https://latex.codecogs.com/svg.image?\inline&space;C) is the number of variables (channels).
* ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) spatio-temporal forecasting: ![](https://latex.codecogs.com/svg.image?\inline&space;N&space;\times&space;L_1\times&space;C&space;\to&space;N&space;\times&space;L_2\times&space;C), where ![](https://latex.codecogs.com/svg.image?\inline&space;N) is the spatial dimension (number of measurement points). However, some spatio-temporal models set the output channel to 1, and even the input channel to 1, which is actually equivalent to multivariate time series forecasting.
* ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) irregular time series: observation/sampling times are irregular.


## News.
🚩 2023/11/1: **I have marked some recommended papers with 🌟 (Just my personal preference 😉).**

🚩 2023/11/1: **I have added a new category ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange): models specifically designed for irregular time series.**

🚩 2023/11/1: **I also recommend you to check out some other GitHub repositories about awesome time series papers: [time-series-transformers-review](https://github.com/qingsongedu/time-series-transformers-review), [awesome-AI-for-time-series-papers](https://github.com/qingsongedu/awesome-AI-for-time-series-papers), [time-series-papers](https://github.com/xiyuanzh/time-series-papers), [deep-learning-time-series](https://github.com/Alro10/deep-learning-time-series).**

🚩 2023/11/3: **There are some popular toolkits or code libraries that integrate many time series models: [PyPOTS](https://github.com/WenjieDu/PyPOTS), [Time-Series-Library](https://github.com/thuml/Time-Series-Library), [Prophet](https://github.com/facebook/prophet), [Darts](https://github.com/unit8co/darts), [Kats](https://github.com/facebookresearch/Kats), [tsai](https://github.com/timeseriesAI/tsai), [GluonTS](https://github.com/awslabs/gluonts), [PyTorchForecasting](https://github.com/jdb78/pytorch-forecasting), [tslearn](https://github.com/tslearn-team/tslearn), [AutoGluon](https://github.com/autogluon/autogluon), [flow-forecast](https://github.com/AIStream-Peelout/flow-forecast), [PyFlux](https://github.com/RJT1990/pyflux).**

🚩 2023/12/28: **Since the topic of LLM(Large Language Model)+TS(Time Series) has been popular recently, I have introduced a category (LLM) to include related papers. This is distinguished from the Pretrain category. Pretrain mainly contains papers which design agent tasks (contrastive or generative) suitable for time series, and only use large-scale time series data for pre-training.**

🚩 2024/4/1: **Some researchers have introduced the recently popular [Mamba](https://arxiv.org/abs/2312.00752) model into the field of time series forecasting, which can be found in the SSM (State Space Model) table.**

🚩 2024/6/7: **I will mark some hot papers with 🔥 (Papers with over 100 citations).**

🚩 2024/9/10: **I am preparing to open [a new GitHub repository](https://github.com/ddz16/VSTFpaper) to collect papers related to Video Spatio-Temporal Forecasting (VSTF). The mapping function for VSTF is ![](https://latex.codecogs.com/svg.image?\inline&space;H&space;\times&space;W&space;\times&space;L_1\times&space;C&space;\to&space;H&space;\times&space;W&space;\times&space;L_2\times&space;C), where ![](https://latex.codecogs.com/svg.image?\inline&space;H) and ![](https://latex.codecogs.com/svg.image?\inline&space;W) are the height and width of each frame. Compared to spatio-temporal forecasting mentioned before, it replaces ![](https://latex.codecogs.com/svg.image?\inline&space;N) with ![](https://latex.codecogs.com/svg.image?\inline&space;H&space;\times&space;W). This setup is commonly used in video prediction and weather forecasting. Stay tuned!**

🚩 2024/10/23: **I have introduced a new table (Multimodal) to include papers that utilize multimodal data (such as relevant text) to assist in forecasting and a new table (KAN) to include papers that utilize Kolmogorov–Arnold Networks.**

🚩 2024/12/30: **Christoph Bergmeir raised insightful questions about the benchmarks in the field of time series forecasting during [his talk at NIPS 2024](https://cbergmeir.com/talks/neurips2024/). This critique is highly valuable and well worth watching. I strongly recommend watching this talk before embarking on time series research.**

🚩 2025/06/02: **I have categorized the papers in the Pretrain & Representation Table into two groups: Representation Learning and Foundation Models. The former focuses on designing pretrain tasks (such as contrastive learning and masked modeling), while the latter typically provides time series foundation models pre-trained on large-scale time series datasets.**

<details><summary><h2 style="display: inline;">Survey & Benchmark.</h2></summary>

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
15-11-23|[Multi-step](https://ieeexplore.ieee.org/abstract/document/7422387)|ACOMP 2015|Comparison of Strategies for Multi-step-Ahead Prediction of Time Series Using Neural Network|None
19-06-20|[DL](https://ieeexplore.ieee.org/abstract/document/8742529)🔥 | SENSJ 2019|A Review of Deep Learning Models for Time Series Prediction|None
20-09-27|[DL](https://arxiv.org/abs/2004.13408)🔥 |Arxiv 2020|Time Series Forecasting With Deep Learning: A Survey|None
22-02-15|[Transformer](https://arxiv.org/abs/2202.07125)🔥 |IJCAI 2023|Transformers in Time Series: A Survey|[PaperList](https://github.com/qingsongedu/time-series-transformers-review)
23-03-25|[STGNN](https://arxiv.org/abs/2303.14483)🔥 |TKDE 2023|Spatio-Temporal Graph Neural Networks for Predictive Learning in Urban Computing: A Survey|None
23-05-01|[Diffusion](https://arxiv.org/abs/2305.00624)|Arxiv 2023|Diffusion Models for Time Series Applications: A Survey|None
23-06-14|[LargeST](https://arxiv.org/abs/2306.08259)|NIPS 2023|LargeST: A Benchmark Dataset for Large-Scale Traffic Forecasting|[largest](https://github.com/liuxu77/largest)
23-06-16|[SSL](https://arxiv.org/abs/2306.10125)🔥|TPAMI 2024|Self-Supervised Learning for Time Series Analysis: Taxonomy, Progress, and Prospects|None
23-06-20|[OpenSTL](https://arxiv.org/abs/2306.11249)|NIPS 2023|OpenSTL: A Comprehensive Benchmark of Spatio-Temporal Predictive Learning|[Benchmark](https://github.com/chengtan9907/OpenSTL)
23-07-07|[GNN](https://arxiv.org/abs/2307.03759)🔥|Arxiv 2023|A Survey on Graph Neural Networks for Time Series: Forecasting, Classification, Imputation, and Anomaly Detection|[PaperList](https://github.com/KimMeen/Awesome-GNN4TS)
23-10-09|[BasicTS](https://arxiv.org/abs/2310.06119)|TKDE 2024|Exploring Progress in Multivariate Time Series Forecasting: Comprehensive Benchmarking and Heterogeneity Analysis|[BasicTS](https://github.com/GestaltCogTeam/BasicTS)
23-10-11|[ProbTS](https://arxiv.org/abs/2310.07446)|Arxiv 2023|ProbTS: Benchmarking Point and Distributional Forecasting across Diverse Prediction Horizons|[ProbTS](https://github.com/microsoft/probts)
23-12-28|[TSPP](https://arxiv.org/abs/2312.17100)|Arxiv 2023|TSPP: A Unified Benchmarking Tool for Time-series Forecasting|[TSPP](https://github.com/NVIDIA/DeepLearningExamples)
24-01-05|[Diffusion](https://arxiv.org/abs/2401.03006)|Arxiv 2024|The Rise of Diffusion Models in Time-Series Forecasting|None
24-02-15|[LLM](https://arxiv.org/abs/2402.10350)🔥|Arxiv 2024|Large Language Models for Forecasting and Anomaly Detection: A Systematic Literature Review|None
24-03-21|[FM](https://arxiv.org/abs/2403.14735)|KDD 2024|Foundation Models for Time Series Analysis: A Tutorial and Survey| None
24-03-29|[TFB](https://arxiv.org/abs/2403.20150)🌟 |VLDB 2024|TFB: Towards Comprehensive and Fair Benchmarking of Time Series Forecasting Methods|[TFB](https://github.com/decisionintelligence/TFB)
24-04-24|[Mamba-360](https://arxiv.org/abs/2404.16112)|Arxiv 2024|Mamba-360: Survey of State Space Models as Transformer Alternative for Long Sequence Modelling: Methods, Applications, and Challenges|[Mamba-360](https://github.com/badripatro/mamba360)
24-04-29|[Diffusion](https://arxiv.org/abs/2404.18886)|Arxiv 2024|A Survey on Diffusion Models for Time Series and Spatio-Temporal Data|[PaperList](https://github.com/yyysjz1997/Awesome-TimeSeries-SpatioTemporal-Diffusion-Model)
24-05-03|[FoundationModels](https://arxiv.org/abs/2405.02358)|Arxiv 2024|A Survey of Time Series Foundation Models: Generalizing Time Series Representation with Large Language Model|[PaperList](https://github.com/start2020/awesome-timeseries-llm-fm)
24-07-18|[TSLib](https://arxiv.org/abs/2407.13278)🌟 |Arxiv 2024|Deep Time Series Models: A Comprehensive Survey and Benchmark|[TSLib](https://github.com/thuml/Time-Series-Library)
24-07-29|[Transformer](https://arxiv.org/abs/2407.19784) |Arxiv 2024| Survey and Taxonomy: The Role of Data-Centric AI in Transformer-Based Time Series Forecasting| None
24-10-14|[GIFT-Eval](https://arxiv.org/abs/2410.10393) |Arxiv 2024| GIFT-Eval: A Benchmark For General Time Series Forecasting Model Evaluation | None
24-10-15|[FoundTS](https://arxiv.org/abs/2410.11802) |Arxiv 2024| FoundTS: Comprehensive and Unified Benchmarking of Foundation Models for Time Series Forecasting | [FoundTS](https://anonymous.4open.science/r/FoundTS-C2B0)
24-10-24|[Architecture](https://arxiv.org/abs/2411.05793) |Arxiv 2024| A Comprehensive Survey of Time Series Forecasting: Architectural Diversity and Open Challenges | None
24-10-29|[STGNN](https://arxiv.org/abs/2410.22377) |Arxiv 2024| A Systematic Literature Review of Spatio-Temporal Graph Neural Network Models for Time Series Forecasting and Classification | None
24-12-19|[Benchmark](https://arxiv.org/abs/2412.14435)🌟 |AAAI 2025| Cherry-Picking in Time Series Forecasting: How to Select Datasets to Make Your Model Shine | [bench](https://github.com/luisroque/bench)
25-02-11|[Physiome-ODE](https://arxiv.org/abs/2502.07489) |ICLR 2025| Physiome-ODE: A Benchmark for Irregularly Sampled Multivariate Time Series Forecasting Based on Biological ODEs | [Physiome-ODE](https://anonymous.4open.science/r/Phyisiome-ODE-E53D)
25-02-15|[Channel-Strategy](https://arxiv.org/abs/2502.10721) |Arxiv 2025| A Comprehensive Survey of Deep Learning for Multivariate Time Series Forecasting: A Channel Strategy Perspective | [CS4TS](https://github.com/decisionintelligence/CS4TS)
25-02-17|[Positional-Encoding](https://arxiv.org/abs/2502.12370) |Arxiv 2025| Positional Encoding in Transformer-Based Time Series Models: A Survey | [pe-benchmark](https://github.com/imics-lab/positional-encoding-benchmark)
25-02-19|[LTSF](https://arxiv.org/abs/2502.14045) |Arxiv 2025| Position: There are no Champions in Long-Term Time Series Forecasting | None
25-02-26|[FinTSB](https://arxiv.org/abs/2502.18834) |Arxiv 2025| FinTSB: A Comprehensive and Practical Benchmark for Financial Time Series Forecasting | [FinTSB](https://github.com/TongjiFinLab/FinTSB)
25-03-13|[DL](https://arxiv.org/abs/2503.10198) |Arxiv 2025| Deep Learning for Time Series Forecasting: A Survey | None
25-04-05|[FoundationModels](https://arxiv.org/abs/2504.04011) |Arxiv 2025| Foundation Models for Time Series: A Survey | None


</details>


<details><summary><h2 style="display: inline;">Transformer.</h2></summary>

Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|----|-----|-----|-----
19-06-29|[LogTrans](https://arxiv.org/abs/1907.00235)🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPS 2019|Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting|[flowforecast](https://github.com/AIStream-Peelout/flow-forecast/blob/master/flood_forecast/transformer_xl/transformer_bottleneck.py) |
19-12-19|[TFT](https://arxiv.org/abs/1912.09363)🌟🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |IJoF 2021|[Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting](https://www.zhihu.com/question/451816360/answer/2319401126)|[tft](https://github.com/google-research/google-research/tree/master/tft) |
20-01-23|[InfluTrans](https://arxiv.org/abs/2001.08317)🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |Arxiv 2020|[Deep Transformer Models for Time Series Forecasting: The Influenza Prevalence Case](https://towardsdatascience.com/how-to-make-a-pytorch-transformer-for-time-series-forecasting-69e073d4061e)|[influenza transformer](https://github.com/KasperGroesLudvigsen/influenza_transformer) |
20-06-05|[AST](https://proceedings.neurips.cc/paper/2020/file/c6b8c8d762da15fa8dbbdfb6baf9e260-Paper.pdf) 🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPS 2020|Adversarial Sparse Transformer for Time Series Forecasting|[AST](https://github.com/hihihihiwsf/AST)
20-12-14|[Informer](https://arxiv.org/abs/2012.07436)🌟🔥| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |AAAI 2021|[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://zhuanlan.zhihu.com/p/467523291)|[Informer](https://github.com/zhouhaoyi/Informer2020)
21-05-22|[ProTran](https://proceedings.neurips.cc/paper_files/paper/2021/file/c68bd9055776bf38d8fc43c0ed283678-Paper.pdf)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |NIPS 2021|Probabilistic Transformer for Time Series Analysis|None
21-06-24|[Autoformer](https://arxiv.org/abs/2106.13008)🌟🔥| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |NIPS 2021|[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://zhuanlan.zhihu.com/p/385066440)|[Autoformer](https://github.com/thuml/Autoformer)
21-09-17|[Aliformer](https://arxiv.org/abs/2109.08381)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2021 | From Known to Unknown: Knowledge-guided Transformer for Time-Series Sales Forecasting in Alibaba | None
21-10-05|[Pyraformer](https://openreview.net/pdf?id=0EXmFzUn5I)🔥| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICLR 2022|[Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting](https://zhuanlan.zhihu.com/p/467765457)|[Pyraformer](https://github.com/alipay/Pyraformer)
22-01-14|[Preformer](https://arxiv.org/abs/2202.11356)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICASSP 2023|[Preformer: Predictive Transformer with Multi-Scale Segment-wise Correlations for Long-Term Time Series Forecasting](https://zhuanlan.zhihu.com/p/536398013)|[Preformer](https://github.com/ddz16/Preformer)
22-01-30|[FEDformer](https://arxiv.org/abs/2201.12740)🌟🔥 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICML 2022|[FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://zhuanlan.zhihu.com/p/528131016)|[FEDformer](https://github.com/MAZiqing/FEDformer)
22-02-03|[ETSformer](https://arxiv.org/abs/2202.01381) 🔥| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2022|[ETSformer: Exponential Smoothing Transformers for Time-series Forecasting](https://blog.salesforceairesearch.com/etsformer-time-series-forecasting/)|[etsformer](https://github.com/salesforce/etsformer)
22-02-07|[TACTiS](https://arxiv.org/abs/2202.03528)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICML 2022|TACTiS: Transformer-Attentional Copulas for Time Series|[TACTiS](https://github.com/ServiceNow/tactis)
22-04-28|[Triformer](https://arxiv.org/abs/2204.13767)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |IJCAI 2022|[Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting](https://blog.csdn.net/zj_18706809267/article/details/125048492)| [Triformer](https://github.com/razvanc92/triformer)
22-05-27|[TDformer](https://arxiv.org/abs/2212.08151)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |NIPSW 2022|[First De-Trend then Attend: Rethinking Attention for Time-Series Forecasting](https://zhuanlan.zhihu.com/p/596022160)|[TDformer](https://github.com/BeBeYourLove/TDformer)
22-05-28|[Non-stationary Transformer](https://arxiv.org/abs/2205.14415) 🔥 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |NIPS 2022|[Non-stationary Transformers: Rethinking the Stationarity in Time Series Forecasting](https://zhuanlan.zhihu.com/p/535931701)|[Non-stationary Transformers](https://github.com/thuml/Nonstationary_Transformers)
22-06-08|[Scaleformer](https://arxiv.org/abs/2206.04038)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICLR 2023|[Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting](https://zhuanlan.zhihu.com/p/535556231)|[Scaleformer](https://github.com/BorealisAI/scaleformer)
22-08-14|[Quatformer](https://dl.acm.org/doi/abs/10.1145/3534678.3539234)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |KDD 2022|Learning to Rotate: Quaternion Transformer for Complicated Periodical Time Series Forecasting|[Quatformer](https://github.com/DAMO-DI-ML/KDD2022-Quatformer)
22-08-30|[Persistence Initialization](https://arxiv.org/abs/2208.14236)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |Arxiv 2022|[Persistence Initialization: A novel adaptation of the Transformer architecture for Time Series Forecasting](https://zhuanlan.zhihu.com/p/582419707)|None
22-09-08|[W-Transformers](https://arxiv.org/abs/2209.03945)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |ICMLA 2022|[W-Transformers: A Wavelet-based Transformer Framework for Univariate Time Series Forecasting](https://zhuanlan.zhihu.com/p/582419707)|[w-transformer](https://github.com/capwidow/w-transformer)
22-09-22|[Crossformer](https://openreview.net/forum?id=vSVLM2j9eie) 🔥 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICLR 2023|Crossformer: Transformer Utilizing Cross-Dimension Dependency for Multivariate Time Series Forecasting|[Crossformer](https://github.com/Thinklab-SJTU/Crossformer)
22-09-22|[PatchTST](https://arxiv.org/abs/2211.14730)🌟🔥| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |ICLR 2023|[A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://zhuanlan.zhihu.com/p/602332939)|[PatchTST](https://github.com/yuqinie98/patchtst)
22-11-29|[AirFormer](https://arxiv.org/abs/2211.15979)| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2023 |AirFormer: Predicting Nationwide Air Quality in China with Transformers | [AirFormer](https://github.com/yoshall/airformer)
22-12-06|[TVT](https://arxiv.org/abs/2212.02789) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2022 | A K-variate Time Series Is Worth K Words: Evolution of the Vanilla Transformer Architecture for Long-term Multivariate Time Series Forecasting | None
23-01-05|[Conformer](https://arxiv.org/abs/2301.02068)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICDE 2023|Towards Long-Term Time-Series Forecasting: Feature, Pattern, and Distribution|[Conformer](https://github.com/PaddlePaddle/PaddleSpatial/tree/main/research/Conformer)
23-01-19|[PDFormer](https://arxiv.org/abs/2301.07945)🔥| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2023 | PDFormer: Propagation Delay-Aware Dynamic Long-Range Transformer for Traffic Flow Prediction | [PDFormer](https://github.com/BUAABIGSCity/PDFormer)
23-03-01|[ViTST](https://arxiv.org/abs/2303.12799)| ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | NIPS 2023 | Time Series as Images: Vision Transformer for Irregularly Sampled Time Series |[ViTST](https://github.com/Leezekun/ViTST)
23-05-20|[CARD](https://arxiv.org/abs/2305.12095)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICLR 2024| CARD: Channel Aligned Robust Blend Transformer for Time Series Forecasting | [CARD](https://github.com/wxie9/card)
23-05-24|[JTFT](https://arxiv.org/abs/2305.14649) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NN 2024 | A Joint Time-frequency Domain Transformer for Multivariate Time Series Forecasting | None
23-05-30|[HSTTN](https://arxiv.org/abs/2305.18724) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | IJCAI 2023 | Long-term Wind Power Forecasting with Hierarchical Spatial-Temporal Transformer | None
23-05-30|[Client](https://arxiv.org/abs/2305.18838) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | Client: Cross-variable Linear Integrated Enhanced Transformer for Multivariate Long-Term Time Series Forecasting | [Client](https://github.com/daxin007/client)
23-05-30|[Taylorformer](https://arxiv.org/abs/2305.19141) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | Taylorformer: Probabilistic Predictions for Time Series and other Processes | [Taylorformer](https://www.dropbox.com/s/vnxuwq7zm7m9bj8/taylorformer.zip?dl=0)
23-06-05|[Corrformer](https://www.nature.com/articles/s42256-023-00667-9)🌟 | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | NMI 2023 | [Interpretable weather forecasting for worldwide stations with a unified deep model](https://zhuanlan.zhihu.com/p/635902919) | [Corrformer](https://github.com/thuml/Corrformer)
23-06-14|[GCformer](https://arxiv.org/abs/2306.08325) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | CIKM 2023 | GCformer: An Efficient Solution for Accurate and Scalable Long-Term Multivariate Time Series Forecasting | [GCformer](https://github.com/zyj-111/gcformer)
23-07-04 | [SageFormer](https://arxiv.org/abs/2307.01616) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | IoT 2024 | SageFormer: Series-Aware Graph-Enhanced Transformers for Multivariate Time Series Forecasting | [SageFormer](https://github.com/zhangzw16/SageFormer) 
23-07-10 | [DifFormer](https://ieeexplore.ieee.org/abstract/document/10177239) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | TPAMI 2023 | DifFormer: Multi-Resolutional Differencing Transformer With Dynamic Ranging for Time Series Analysis | None
23-07-27 | [HUTFormer](https://arxiv.org/abs/2307.14596) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2023 | HUTFormer: Hierarchical U-Net Transformer for Long-Term Traffic Forecasting | None
23-08-07 | [DSformer](https://arxiv.org/abs/2308.03274) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | CIKM 2023 | DSformer: A Double Sampling Transformer for Multivariate Time Series Long-term Prediction | None
23-08-09 | [SBT](https://arxiv.org/abs/2308.04637) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2023 | Sparse Binary Transformers for Multivariate Time Series Modeling | None
23-08-09 | [PETformer](https://arxiv.org/abs/2308.04791) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | PETformer: Long-term Time Series Forecasting via Placeholder-enhanced Transformer | None
23-10-02 | [TACTiS-2](https://browse.arxiv.org/abs/2310.01327)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2024 | TACTiS-2: Better, Faster, Simpler Attentional Copulas for Multivariate Time Series | None
23-10-03 | [PrACTiS](https://browse.arxiv.org/abs/2310.01720)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | PrACTiS: Perceiver-Attentional Copulas for Time Series | None
23-10-10 | [iTransformer](https://arxiv.org/abs/2310.06625)🔥 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2024 | [iTransformer: Inverted Transformers Are Effective for Time Series Forecasting](https://zhuanlan.zhihu.com/p/662250788) | [iTransformer](https://github.com/thuml/iTransformer)
23-10-26 | [ContiFormer](https://seqml.github.io/contiformer/)| ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | NIPS 2023 | ContiFormer: Continuous-Time Transformer for Irregular Time Series Modeling | [ContiFormer](https://github.com/microsoft/SeqML/tree/main/ContiFormer)
23-10-31 | [BasisFormer](https://arxiv.org/abs/2310.20496)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2023 | BasisFormer: Attention-based Time Series Forecasting with Learnable and Interpretable Basis | [basisformer](https://github.com/nzl5116190/basisformer)
23-11-07 | [MTST](https://arxiv.org/abs/2311.04147)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | AISTATS 2023 | Multi-resolution Time-Series Transformer for Long-term Forecasting | [MTST](https://github.com/networkslab/MTST)
23-11-30 | [MultiResFormer](https://arxiv.org/abs/2311.18780)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | MultiResFormer: Transformer with Adaptive Multi-Resolution Modeling for General Time Series Forecasting | None
23-12-10 | [FPPformer](https://arxiv.org/abs/2312.05792)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | IOT 2023 | Take an Irregular Route: Enhance the Decoder of Time-Series Forecasting Transformer | [FPPformer](https://github.com/OrigamiSL/FPPformer)
23-12-11 | [Dozerformer](https://arxiv.org/abs/2312.06874)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | Dozerformer: Sequence Adaptive Sparse Transformer for Multivariate Time Series Forecasting | None
23-12-11 | [CSformer](https://arxiv.org/abs/2312.06220)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | Dance of Channel and Sequence: An Efficient Attention-Based Approach for Multivariate Time Series Forecasting | None
23-12-23 | [MASTER](https://arxiv.org/abs/2312.15235)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2024 | MASTER: Market-Guided Stock Transformer for Stock Price Forecasting | [MASTER](https://github.com/SJTU-Quant/MASTER)
23-12-30 | [PCA+former](https://arxiv.org/abs/2401.00230)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | Transformer Multivariate Forecasting: Less is More? | None
24-01-16 | [PDF](https://openreview.net/forum?id=dp27P5HBBt)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | ICLR 2024 | [Periodicity Decoupling Framework for Long-term Series Forecasting](https://zhuanlan.zhihu.com/p/699708089) | [PDF](https://github.com/Hank0626/PDF)
24-01-16 | [Pathformer](https://openreview.net/forum?id=lJkOCMP2aW)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2024 | Multi-scale Transformers with Adaptive Pathways for Time Series Forecasting | [pathformer](https://github.com/decisionintelligence/pathformer)
24-01-16 | [VQ-TR](https://openreview.net/forum?id=IxpTsFS7mh)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2024 | VQ-TR: Vector Quantized Attention for Time Series Forecasting | None
24-01-22 | [HDformer](https://arxiv.org/abs/2401.11929)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | The Bigger the Better? Rethinking the Effective Model Scale in Long-term Time Series Forecasting | None
24-02-04 | [Minusformer](https://arxiv.org/abs/2402.02332)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Minusformer: Improving Time Series Forecasting by Progressively Learning Residuals | [Minusformer](https://github.com/anoise/minusformer)
24-02-08 | [AttnEmbed](https://arxiv.org/abs/2402.05370)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2024 | Attention as Robust Representation for Time Series Forecasting | [AttnEmbed](https://anonymous.4open.science/r/AttnEmbed-7430)
24-02-15 | [SAMformer](https://arxiv.org/abs/2402.10198)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICML 2024 | [Unlocking the Potential of Transformers in Time Series Forecasting with Sharpness-Aware Minimization and Channel-Wise Attention](https://romilbert.github.io/samformer_slides.pdf) | [SAMformer](https://github.com/romilbert/samformer)
24-02-25 | [PDETime](https://arxiv.org/abs/2402.16913)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | PDETime: Rethinking Long-Term Multivariate Time Series Forecasting from the perspective of partial differential equations | None
24-02-29 | [TimeXer](https://arxiv.org/abs/2402.19072)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2024 | TimeXer: Empowering Transformers for Time Series Forecasting with Exogenous Variables | [TimeXer](https://github.com/thuml/TimeXer)
24-03-05 | [InjectTST](https://arxiv.org/abs/2403.02814)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | InjectTST: A Transformer Method of Injecting Global Information into Independent Channels for Long Time Series Forecasting | None
24-03-13 | [Caformer](https://arxiv.org/abs/2403.08572)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Caformer: Rethinking Time Series Analysis from Causal Perspective | None
24-03-14 | [MCformer](https://arxiv.org/abs/2403.09223)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | MCformer: Multivariate Time Series Forecasting with Mixed-Channels Transformer | None
24-04-08 | [ATFNet](https://arxiv.org/abs/2404.05192)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2024 | ATFNet: Adaptive Time-Frequency Ensembled Network for Long-term Time Series Forecasting | [ATFNet](https://github.com/YHYHYHYHYHY/ATFNet)
24-04-12 | [TSLANet](https://arxiv.org/abs/2404.08472)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | ICML 2024 | TSLANet: Rethinking Transformers for Time Series Representation Learning | [TSLANet](https://github.com/emadeldeen24/TSLANet)
24-04-16 | [T2B-PE](https://arxiv.org/abs/2404.10337)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Intriguing Properties of Positional Encoding in Time Series Forecasting | [T2B-PE](https://github.com/jlu-phyComputer/T2B-PE)
24-05-14 | [DGCformer](https://arxiv.org/abs/2405.08440)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | DGCformer: Deep Graph Clustering Transformer for Multivariate Time Series Forecasting | None
24-05-19 | [VCformer](https://arxiv.org/abs/2405.11470)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | VCformer: Variable Correlation Transformer with Inherent Lagged Correlation for Multivariate Time Series Forecasting | [VCformer](https://github.com/CSyyn/VCformer)
24-05-22 | [GridTST](https://arxiv.org/abs/2405.13810)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Leveraging 2D Information for Long-term Time Series Forecasting with Vanilla Transformers | [GridTST](https://github.com/Hannibal046/GridTST)
24-05-23 | [ICTSP](https://arxiv.org/abs/2405.14982)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2025 | In-context Time Series Predictor | None
24-05-27 | [CATS](https://arxiv.org/abs/2405.16877)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | NIPS 2024 | Are Self-Attentions Effective for Time Series Forecasting? | [CATS](https://github.com/dongbeank/CATS)
24-06-06 | [TwinS](https://arxiv.org/abs/2406.03710)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | TwinS: Revisiting Non-Stationarity in Multivariate Time Series Forecasting | None
24-06-07 | [UniTST](https://arxiv.org/abs/2406.04975)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | UniTST: Effectively Modeling Inter-Series and Intra-Series Dependencies for Multivariate Time Series Forecasting | None
24-06-11 | [DeformTime](https://arxiv.org/abs/2406.07438)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | DeformTime: Capturing Variable Dependencies with Deformable Attention for Time Series Forecasting | [DeformTime](https://github.com/ClaudiaShu/DeformTime)
24-06-13 | [Fredformer](https://arxiv.org/abs/2406.09009)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2024 | Fredformer: Frequency Debiased Transformer for Time Series Forecasting | [Fredformer](https://github.com/chenzrg/fredformer)
24-07-18 | [FSatten-SOatten](https://arxiv.org/abs/2407.13806)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Revisiting Attention for Multivariate Time Series Forecasting | [FSatten-SOatten](https://github.com/Joeland4/FSatten-SOatten)
24-07-18 | [MTE](https://arxiv.org/abs/2407.15869)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2024 | Long Input Sequence Network for Long Time Series Forecasting | [MTE](https://github.com/Houyikai/MTE)
24-07-31 | [FreqTSF](https://arxiv.org/abs/2407.21275)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | FreqTSF: Time Series Forecasting Via Simulating Frequency Kramer-Kronig Relations | None
24-08-05 | [DRFormer](https://arxiv.org/abs/2408.02279)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | CIKM 2024 | DRFormer: Multi-Scale Transformer Utilizing Diverse Receptive Fields for Long Time-Series Forecasting | [DRFormer](https://github.com/ruixindingECNU/DRFormer)
24-08-08 | [STHD](https://arxiv.org/abs/2408.04245)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | CIKM 2024 | Scalable Transformer for High Dimensional Multivariate Time Series Forecasting | [STHD](https://github.com/xinzzzhou/ScalableTransformer4HighDimensionMTSF)
24-08-16 | [S3Attention](https://arxiv.org/abs/2408.08567)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | S3Attention: Improving Long Sequence Attention with Smoothed Skeleton Sketching | [S3Attention](https://github.com/wxie9/S3Attention)
24-08-19 | [PMformer](https://arxiv.org/abs/2408.09703)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Partial-Multivariate Model for Forecasting | None
24-08-19 | [sTransformer](https://arxiv.org/abs/2408.09723)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | sTransformer: A Modular Approach for Extracting Inter-Sequential and Temporal Information for Time-Series Forecasting | None
24-08-20 | [PRformer](https://arxiv.org/abs/2408.10483)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | PRformer: Pyramidal Recurrent Transformer for Multivariate Time Series Forecasting | [PRformer](https://github.com/usualheart/PRformer)
24-09-25 | [DeformableTST](https://openreview.net/forum?id=B1Iq1EOiVU)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | NIPS 2024 | DeformableTST: Transformer for Time Series Forecasting without Over-reliance on Patching | [DeformableTST](https://github.com/luodhhh/DeformableTST)
24-09-30 | [CTLPE](https://arxiv.org/abs/2409.20092)| ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | Arxiv 2024 | Continuous-Time Linear Positional Embedding for Irregular Time Series Forecasting | None
24-10-02 | [TiVaT](https://arxiv.org/abs/2410.01531)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | TiVaT: Joint-Axis Attention for Time Series Forecasting with Lead-Lag Dynamics | None
24-10-04 | [ARMA](https://arxiv.org/abs/2410.03159)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2024 | Autoregressive Moving-average Attention Mechanism for Time Series Forecasting | [ARMA-Attention](https://github.com/LJC-FVNR/ARMA-Attention)
24-10-06 | [TimeBridge](https://arxiv.org/abs/2410.04442)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | TimeBridge: Non-Stationarity Matters for Long-term Time Series Forecasting | [TimeBridge](https://github.com/Hank0626/TimeBridge)
24-10-30 | [WaveRoRA](https://arxiv.org/abs/2410.22649)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | WaveRoRA: Wavelet Rotary Route Attention for Multivariate Time Series Forecasting | None
24-10-31 | [Ada-MSHyper](https://arxiv.org/abs/2410.23992)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2024 | Ada-MSHyper: Adaptive Multi-Scale Hypergraph Transformer for Time Series Forecasting | [Ada-MSHyper](https://github.com/shangzongjiang/Ada-MSHyper)
24-11-03 | [PSformer](https://arxiv.org/abs/2411.01419)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | PSformer: Parameter-efficient Transformer with Segment Attention for Time Series Forecasting | None
24-11-04 | [ElasTST](https://arxiv.org/abs/2411.01842)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | NIPS 2024 | ElasTST: Towards Robust Varied-Horizon Forecasting with Elastic Time-Series Transformer | [elastst](https://github.com/microsoft/ProbTS/tree/elastst)
24-11-07 | [Peri-midFormer](https://arxiv.org/abs/2411.04554)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | NIPS 2024 | Peri-midFormer: Periodic Pyramid Transformer for Time Series Analysis | [Peri-midFormer](https://github.com/WuQiangXDU/Peri-midFormer)
24-12-02 | [MuSiCNet](https://arxiv.org/abs/2412.01063)| ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | IJCAIW 2024 | MuSiCNet: A Gradual Coarse-to-Fine Framework for Irregularly Sampled Multivariate Time Series Analysis | None
24-12-04 | [HOT](https://arxiv.org/abs/2412.02919)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Higher Order Transformers: Efficient Attention Mechanism for Tensor Structured Data | None
24-12-16 | [EDformer](https://arxiv.org/abs/2412.12227)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | EDformer: Embedded Decomposition Transformer for Interpretable Multivariate Time Series Predictions | [EDformer](https://github.com/sanjaylopa22/EDformer-Main)
24-12-17 | [TimeCHEAT](https://arxiv.org/abs/2412.12886)| ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | AAAI 2025 | TimeCHEAT: A Channel Harmony Strategy for Irregularly Sampled Multivariate Time Series Analysis | None
24-12-25 | [Ister](https://arxiv.org/abs/2412.18798)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Ister: Inverted Seasonal-Trend Decomposition Transformer for Explainable Multivariate Time Series Forecasting | None
25-01-06 | [Sensorformer](https://arxiv.org/abs/2501.03284)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Sensorformer: Cross-patch attention with global-patch compression is effective for high-dimensional multivariate time series forecasting | [Sensorformer](https://github.com/BigYellowTiger/Sensorformer)
25-01-14 | [LiPFormer](https://arxiv.org/abs/2501.10448)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | ICDE 2025 | Towards Lightweight Time Series Forecasting: a Patch-wise Transformer with Weak Data Enriching | None
25-01-22 | [T-Graphormer](https://arxiv.org/abs/2501.13274)| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2025 | T-Graphormer: Using Transformers for Spatiotemporal Forecasting | None
25-01-23 | [FreEformer](https://arxiv.org/abs/2501.13989)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | FreEformer: Frequency Enhanced Transformer for Multivariate Time Series Forecasting | None
25-01-23 | [SimMTSF](https://openreview.net/forum?id=oANkBaVci5)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2025 | A Simple Baseline for Multivariate Time Series Forecasting | None
25-01-24 | [VarDrop](https://arxiv.org/abs/2501.14183)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2025 | VarDrop: Enhancing Training Efficiency by Reducing Variate Redundancy in Periodic Time Series Forecasting | [VarDrop](https://github.com/kaist-dmlab/VarDrop)
25-01-28 | [Spikformer](https://arxiv.org/abs/2501.16745)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Toward Relative Positional Encoding in Spiking Transformers | None
25-02-10 | [Powerformer](https://arxiv.org/abs/2502.06151)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | Powerformer: A Transformer with Weighted Causal Attention for Time-series Forecasting | None
25-02-11 | [SAMoVAR](https://arxiv.org/abs/2502.07244)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Linear Transformers as VAR Models: Aligning Autoregressive Attention Mechanisms with Autoregressive Forecasting | [SAMoVAR](https://github.com/LJC-FVNR/Structural-Aligned-Mixture-of-VAR)
25-02-12 | [HDT](https://arxiv.org/abs/2502.08302)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2025 | HDT: Hierarchical Discrete Transformer for Multivariate Time Series Forecasting | [HDT](https://github.com/hdtkk/HDT)
25-02-13 | [FaCT](https://arxiv.org/abs/2502.09683)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Channel Dependence, Limited Lookback Windows, and the Simplicity of Datasets: How Biased is Time Series Forecasting? | None
25-02-17 | [S2TX](https://arxiv.org/abs/2502.11340)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | S2TX: Cross-Attention Multi-Scale State-Space Transformer for Time Series Forecasting | None
25-02-19 | [AutoFormer-TS](https://arxiv.org/abs/2502.13721)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Learning Novel Transformer Architecture for Time-series Forecasting | None
25-02-27 | [PFformer](https://arxiv.org/abs/2502.20571)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | PAKDD 2025 | PFformer: A Position-Free Transformer Variant for Extreme-Adaptive Multivariate Time Series Forecasting | None
25-03-03 | [ContexTST](https://arxiv.org/abs/2503.01157)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | Unify and Anchor: A Context-Aware Transformer for Cross-Domain Time Series Forecasting | None
25-03-07 | [PPDformer](https://ieeexplore.ieee.org/document/10890581)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICASSP 2025 | PPDformer: Channel-Specific Periodic Patch Division for Time Series Forecasting | [PPDformer](https://github.com/damonwan1/PPDformer)
25-03-10 | [Attn-L-Reg](https://arxiv.org/abs/2503.06867)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Enhancing Time Series Forecasting via Logic-Inspired Regularization | None
25-03-11 | [MFRS](https://arxiv.org/abs/2503.08328)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | MFRS: A Multi-Frequency Reference Series Approach to Scalable and Accurate Time-Series Forecasting | [MFRS](https://github.com/yuliang555/MFRS)
25-03-13 | [EiFormer](https://arxiv.org/abs/2503.10858)| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2025 | Towards Efficient Large Scale Spatial-Temporal Time Series Forecasting via Improved Inverted Transformers | None
25-03-22 | [Sentinel](https://arxiv.org/abs/2503.17658)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Sentinel: Multi-Patch Transformer with Temporal and Channel Attention for Time Series Forecasting | None
25-03-31 | [CITRAS](https://arxiv.org/abs/2503.24007)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | CITRAS: Covariate-Informed Transformer for Time Series Forecasting | None
25-04-02 | [Times2D](https://arxiv.org/abs/2504.00118)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2025 | Times2D: Multi-Period Decomposition and Derivative Mapping for General Time Series Forecasting | [Times2D](https://github.com/Tims2D/Times2D)
25-04-17 | [TimeCapsule](https://arxiv.org/abs/2504.12721)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2025 | TimeCapsule: Solving the Jigsaw Puzzle of Long-Term Time Series Forecasting with Compressed Predictive Representations | None
25-04-26 | [TSRM](https://arxiv.org/abs/2504.18878)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | TSRM: A Lightweight Temporal Feature Encoding Architecture for Time Series Forecasting and Imputation | [TSRM](https://github.com/RobertLeppich/TSRM)
25-05-01 | [Gateformer](https://arxiv.org/abs/2505.00307)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Gateformer: Advancing Multivariate Time Series Forecasting through Temporal and Variate-Wise Attention with Gated Representations | [Gateformer](https://github.com/nyuolab/gateformer)
25-05-04 | [CASA](https://arxiv.org/abs/2505.02011)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | IJCAI 2025 | CASA: CNN Autoencoder-based Score Attention for Efficient Multivariate Long-term Time-series Forecasting | [casa](https://github.com/lmh9507/casa)
25-05-05 | [SCFormer](https://arxiv.org/abs/2505.02655)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | SCFormer: Structured Channel-wise Transformer with Cumulative Historical State for Multivariate Time Series Forecasting | [SCFormer](https://github.com/ShiweiGuo1995/SCFormer)
25-05-19 | [TQNet](https://arxiv.org/abs/2505.12917)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICML 2025 | Temporal Query Network for Efficient Multivariate Time Series Forecasting | [TQNet](https://github.com/ACAT-SCUT/TQNet)
25-05-20 | [LMHR](https://arxiv.org/abs/2505.14737)| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2025 | Leveraging Multivariate Long-Term History Representation for Time Series Forecasting | None
25-05-21 | [Sonnet](https://arxiv.org/abs/2505.15312)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Sonnet: Spectral Operator Neural Network for Multivariable Time Series Forecasting | [Sonnet](https://github.com/ClaudiaShu/Sonnet)


</details>

<details><summary><h2 style="display: inline;">RNN.</h2></summary>

Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----|-----
17-03-21|[LSTNet](https://arxiv.org/abs/1703.07015)🌟🔥 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |SIGIR 2018|[Modeling Long- and Short-Term Temporal Patterns with Deep Neural Networks](https://zhuanlan.zhihu.com/p/467944750)|[LSTNet](https://github.com/laiguokun/LSTNet) |
17-04-07|[DA-RNN](https://arxiv.org/abs/1704.02971)🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |IJCAI 2017| A Dual-Stage Attention-Based Recurrent Neural Network for Time Series Prediction | [DARNN](https://github.com/sunfanyunn/DARNN) |
17-04-13|[DeepAR](https://arxiv.org/abs/1704.04110)🌟🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |IJoF 2019|[DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://zhuanlan.zhihu.com/p/542066911)|[DeepAR](https://github.com/brunoklein99/deepar) |
17-11-29|[MQRNN](https://arxiv.org/abs/1711.11053)🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPSW 2017|A Multi-Horizon Quantile Recurrent Forecaster|[MQRNN](https://github.com/tianchen101/MQRNN) |
18-06-23|[mWDN](https://arxiv.org/abs/1806.08946)🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |KDD 2018|Multilevel Wavelet Decomposition Network for Interpretable Time Series Analysis|[mWDN](https://github.com/yakouyang/Multilevel_Wavelet_Decomposition_Network_Pytorch) |
18-09-06|[MTNet](https://arxiv.org/abs/1809.02105)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |AAAI 2019| A Memory-Network Based Solution for Multivariate Time-Series Forecasting |[MTNet](https://github.com/Maple728/MTNet) |
19-05-28|[DF-Model](https://arxiv.org/abs/1905.12417)🔥 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |ICML 2019| Deep Factors for Forecasting | None |
19-07-18|[ESLSTM](https://www.sciencedirect.com/science/article/pii/S0169207019301153)🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |IJoF 2020|A hybrid method of exponential smoothing and recurrent neural networks for time series forecasting| None |
19-07-25|[MH-TAL](https://dl.acm.org/doi/abs/10.1145/3292500.3330662)🔥 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |KDD 2019|Multi-Horizon Time Series Forecasting with Temporal Attention Learning| None |
21-11-22|[CRU](https://arxiv.org/abs/2111.11344) | ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | ICML 2022 | Modeling Irregular Time Series with Continuous Recurrent Units | [CRU](https://github.com/boschresearch/continuous-recurrent-units)
22-05-16|[C2FAR](https://openreview.net/forum?id=lHuPdoHBxbg)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPS 2022|[C2FAR: Coarse-to-Fine Autoregressive Networks for Precise Probabilistic Forecasting](https://zhuanlan.zhihu.com/p/600602517)|[C2FAR](https://github.com/huaweicloud/c2far_forecasting) |
23-06-02|[RNN-ODE-Adap](https://arxiv.org/abs/2306.01674)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2023|Neural Differential Recurrent Neural Network with Adaptive Time Steps| [RNN_ODE_Adap](https://github.com/Yixuan-Tan/RNN_ODE_Adap) |
23-08-22|[SegRNN](https://arxiv.org/abs/2308.11200)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |Arxiv 2023| SegRNN: Segment Recurrent Neural Network for Long-Term Time Series Forecasting| [SegRNN](https://github.com/lss-1138/SegRNN) |
23-10-05|[PA-RNN](https://arxiv.org/abs/2310.03243)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPS 2023| Sparse Deep Learning for Time Series Data: Theory and Applications | None |
23-11-03|[WITRAN](https://openreview.net/forum?id=y08bkEtNBK)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPS 2023| WITRAN: Water-wave Information Transmission and Recurrent Acceleration Network for Long-range Time Series Forecasting | [WITRAN](https://github.com/Water2sea/WITRAN) |
23-12-14|[DAN](https://arxiv.org/abs/2312.08763)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |AAAI 2024| Learning from Polar Representation: An Extreme-Adaptive Model for Long-Term Time Series Forecasting | [DAN](https://github.com/davidanastasiu/dan) |
23-12-22|[SutraNets](https://arxiv.org/abs/2312.14880)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |NIPS 2023| SutraNets: Sub-series Autoregressive Networks for Long-Sequence, Probabilistic Forecasting | None |
24-01-17|[RWKV-TS](https://arxiv.org/abs/2401.09093)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2024| RWKV-TS: Beyond Traditional Recurrent Neural Network for Time Series Tasks | [RWKV-TS](https://github.com/howard-hou/RWKV-TS) |
24-06-04|[TGLRN](https://arxiv.org/abs/2406.02726)| ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) |Arxiv 2024| Temporal Graph Learning Recurrent Neural Network for Traffic Forecasting | None |
24-06-29|[CONTIME](https://arxiv.org/abs/2407.01622)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |KDD 2024| Addressing Prediction Delays in Time Series Forecasting: A Continuous GRU Approach with Derivative Regularization | [CONTIME](https://github.com/sheoyon-jhin/CONTIME) |
24-07-14|[xLSTMTime](https://arxiv.org/abs/2407.10240)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2024| xLSTMTime: Long-term Time Series Forecasting With xLSTM | [xLSTMTime](https://github.com/muslehal/xLSTMTime) |
24-07-29|[TFFM](https://arxiv.org/abs/2407.19697)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |CIKM 2024| Multiscale Representation Enhanced Temporal Flow Fusion Model for Long-Term Workload Forecasting | None |
24-08-19|[P-sLSTM](https://arxiv.org/abs/2408.10006)| ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) |Arxiv 2024| Unlocking the Power of LSTM for Long Term Time Series Forecasting | None |
24-10-22|[xLSTM-Mixer](https://arxiv.org/abs/2410.16928)| ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) |Arxiv 2024| xLSTM-Mixer: Multivariate Time Series Forecasting by Mixing via Scalar Memories | [xlstm-mixer](https://github.com/mauricekraus/xlstm-mixer) |


</details>

<details><summary><h2 style="display: inline;">MLP.</h2></summary>

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
| 23-07-06 | [FITS](https://arxiv.org/abs/2307.03756)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | ICLR 2024  | [FITS: Modeling Time Series with 10k Parameters](https://zhuanlan.zhihu.com/p/669221150) | [FITS](https://anonymous.4open.science/r/FITS) |
| 23-08-14 | [ST-MLP](https://arxiv.org/abs/2308.07496) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2023  | ST-MLP: A Cascaded Spatio-Temporal Linear Framework with Channel-Independence Strategy for Traffic Forecasting | None |
| 23-08-25 | [TFDNet](https://arxiv.org/abs/2308.13386) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | TFDNet: Time-Frequency Enhanced Decomposed Network for Long-term Time Series Forecasting | None |
| 23-11-10 | [FreTS](https://arxiv.org/abs/2311.06184) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2023 | Frequency-domain MLPs are More Effective Learners in Time Series Forecasting | [FreTS](https://github.com/aikunyi/FreTS)
| 23-12-11 | [MoLE](https://arxiv.org/abs/2312.06786) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AISTATS 2024 | Mixture-of-Linear-Experts for Long-term Time Series Forecasting | [MoLE](https://github.com/RogerNi/MoLE) |
| 23-12-22 | [STL](https://arxiv.org/abs/2312.14869) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023  | Spatiotemporal-Linear: Towards Universal Multivariate Time Series Forecasting | None |
| 24-01-04 | [U-Mixer](https://arxiv.org/abs/2401.02236) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2024 | U-Mixer: An Unet-Mixer Architecture with Stationarity Correction for Time Series Forecasting | None |
| 24-01-16 | [TimeMixer](https://openreview.net/forum?id=7oLshfEIC2) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2024 | [TimeMixer: Decomposable Multiscale Mixing for Time Series Forecasting](https://zhuanlan.zhihu.com/p/686772622) | [TimeMixer](https://github.com/kwuking/TimeMixer) |
| 24-02-16 | [RPMixer](https://arxiv.org/abs/2402.10487) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2024 | Random Projection Layers for Multidimensional Time Sires Forecasting | None |
| 24-02-20 | [IDEA](https://arxiv.org/abs/2402.12767) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2024 | When and How: Learning Identifiable Latent States for Nonstationary Time Series Forecasting | None |
| 24-03-04 | [CATS](https://arxiv.org/abs/2403.01673) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICML 2024 | CATS: Enhancing Multivariate Time Series Forecasting by Constructing Auxiliary Time Series as Exogenous Variables | None |
| 24-03-21 | [OLS](https://arxiv.org/abs/2403.14587) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICML 2024 | An Analysis of Linear Time Series Forecasting Models | None |
| 24-03-24 | [HDMixer](https://ojs.aaai.org/index.php/AAAI/article/view/29155) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2024 | HDMixer: Hierarchical Dependency with Extendable Patch for Multivariate Time Series Forecasting | [HDMixer](https://github.com/hqh0728/HDMixer) |
| 24-04-22 | [SOFTS](https://arxiv.org/abs/2404.14197) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2024 | SOFTS: Efficient Multivariate Time Series Forecasting with Series-Core Fusion | [SOFTS](https://github.com/secilia-cxy/softs) |
| 24-05-02 | [SparseTSF](https://arxiv.org/abs/2405.00946) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | ICML 2024 | [SparseTSF: Modeling Long-term Time Series Forecasting with 1k Parameters](https://zhuanlan.zhihu.com/p/701070533) | [SparseTSF](https://github.com/lss-1138/SparseTSF) |
| 24-05-10 | [TEFN](https://arxiv.org/abs/2405.06419) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Time Evidence Fusion Network: Multi-source View in Long-Term Time Series Forecasting | [TEFN](https://github.com/ztxtech/Time-Evidence-Fusion-Network) |
| 24-05-22 | [PDMLP](https://arxiv.org/abs/2405.13575) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | PDMLP: Patch-based Decomposed MLP for Long-Term Time Series Forecasting | None |
| 24-06-06 | [AMD](https://arxiv.org/abs/2406.03751) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Adaptive Multi-Scale Decomposition Framework for Time Series Forecasting | [AMD](https://github.com/TROUBADOUR000/AMD) |
| 24-06-07 | [TimeSieve](https://arxiv.org/abs/2406.05036) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | TimeSieve: Extracting Temporal Dynamics through Information Bottlenecks | [TimeSieve](https://github.com/xll0328/TimeSieve) |
| 24-06-29 | [DERITS](https://arxiv.org/abs/2407.00502) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | IJCAI 2024 | Deep Frequency Derivative Learning for Non-stationary Time Series Forecasting | None |
| 24-07-15 | [ODFL](https://arxiv.org/abs/2407.10419) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2024 | Omni-Dimensional Frequency Learner for General Time Series Analysis | None |
| 24-07-17 | [FreDF](https://arxiv.org/abs/2407.12415) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | MM 2024 | Not All Frequencies Are Created Equal: Towards a Dynamic Fusion of Frequencies in Time-Series Forecasting | None |
| 24-09-26 | [PGN](https://arxiv.org/abs/2409.17703) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | NIPS 2024 | PGN: The RNN's New Successor is Effective for Long-Range Time Series Forecasting | [TPGN](https://github.com/Water2sea/TPGN) |
| 24-09-27 | [CycleNet](https://arxiv.org/abs/2409.18479) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | NIPS 2024 | [CycleNet: Enhancing Time Series Forecasting through Modeling Periodic Patterns](https://zhuanlan.zhihu.com/p/778345073) | [CycleNet](https://github.com/ACAT-SCUT/CycleNet) |
| 24-10-02 | [MMFNet](https://arxiv.org/abs/2410.02070) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2024 | MMFNet: Multi-Scale Frequency Masking Neural Network for Multivariate Time Series Forecasting | None |
| 24-10-02 | [MixLinear](https://arxiv.org/abs/2410.02081) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2024 | MixLinear: Extreme Low Resource Multivariate Time Series Forecasting with 0.1K Parameters | None |
| 24-10-07 | [NFM](https://arxiv.org/abs/2410.04703) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Neural Fourier Modelling: A Highly Compact Approach to Time-Series Analysis | [NFM](https://github.com/minkiml/NFM) |
| 24-10-13 | [TFPS](https://arxiv.org/abs/2410.09836) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2024 | Learning Pattern-Specific Experts for Time Series Forecasting Under Patch-level Distribution Shift | [TFPS](https://github.com/syrGitHub/TFPS) |
| 24-10-21 | [LTBoost](https://dl.acm.org/doi/pdf/10.1145/3627673.3679527) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | CIKM 2024 | LTBoost: Boosted Hybrids of Ensemble Linear and Gradient Algorithms for the Long-term Time Series Forecasting | [LTBoost](https://github.com/hubtru/LTBoost) |
| 24-10-22 | [LiNo](https://arxiv.org/abs/2410.17159) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | LiNo: Advancing Recursive Residual Decomposition of Linear and Nonlinear Patterns for Robust Time Series Forecasting | [LiNo](https://github.com/Levi-Ackman/LiNo) |
| 24-11-03 | [FilterNet](https://arxiv.org/abs/2411.01623) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | NIPS 2024 | FilterNet: Harnessing Frequency Filters for Time Series Forecasting | [FilterNet](https://github.com/aikunyi/FilterNet) |
| 24-11-26 | [DiPE-Linear](https://arxiv.org/abs/2411.17257) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Disentangled Interpretable Representation for Efficient Long-term Time Series Forecasting | [DiPE-Linear](https://github.com/wintertee/DiPE-Linear) |
| 24-12-02 | [FSMLP](https://arxiv.org/abs/2412.01654) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | FSMLP: Modelling Channel Dependencies With Simplex Theory Based Multi-Layer Perceptions In Frequency Domain | [FSMLP](https://github.com/fmlyd/fsmlp) |
| 24-12-09 | [LMS-AutoTSF](https://arxiv.org/abs/2412.06866) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | LMS-AutoTSF: Learnable Multi-Scale Decomposition and Integrated Autocorrelation for Time Series Forecasting | [LMS-TSF](https://github.com/mribrahim/LMS-TSF) |
| 24-12-14 | [DUET](https://arxiv.org/abs/2412.10859) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2025 | DUET: Dual Clustering Enhanced Multivariate Time Series Forecasting | [DUET](https://github.com/decisionintelligence/duet) |
| 24-12-18 | [PreMixer](https://arxiv.org/abs/2412.13607) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2024 | PreMixer: MLP-Based Pre-training Enhanced MLP-Mixers for Large-scale Traffic Forecasting | None |
| 24-12-22 | [WPMixer](https://arxiv.org/abs/2412.17176) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | AAAI 2025 | WPMixer: Efficient Multi-Resolution Mixing for Long-Term Time Series Forecasting | None |
| 24-12-30 | [AverageLinear](https://arxiv.org/abs/2412.20727) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | AverageLinear: Enhance Long-Term Time series forcasting with simple averaging | None |
| 25-01-25 | [FreqMoE](https://arxiv.org/abs/2501.15125) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AISTATS 2025 | FreqMoE: Enhancing Time Series Forecasting through Frequency Decomposition Mixture of Experts | [FreqMoE](https://github.com/sunbus100/FreqMoE-main) |
| 25-01-27 | [SWIFT](https://arxiv.org/abs/2501.16178) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | SWIFT: Mapping Sub-series with Wavelet Decomposition Improves Time Series Forecasting | [swift](https://github.com/lancelotxwx/swift) |
| 25-01-28 | [Amplifier](https://arxiv.org/abs/2501.17216) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2025 | Amplifier: Bringing Attention to Neglected Low-Energy Components in Time Series Forecasting | [amplifier](https://github.com/aikunyi/amplifier) |
| 25-01-31 | [BEAT](https://arxiv.org/abs/2501.19065) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | BEAT: Balanced Frequency Adaptive Tuning for Long-Term Time-Series Forecasting | None |
| 25-02-05 | [MTLinear](https://arxiv.org/abs/2502.03571) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | A Multi-Task Learning Approach to Linear Multivariate Forecasting | [MTLinear](https://github.com/azencot-group/MTLinear) |
| 25-02-14 | [HADL](https://arxiv.org/abs/2502.10569) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | HADL Framework for Noise Resilient Long-Term Time Series Forecasting | [HADL](https://github.com/forgee-master/HADL) |
| 25-02-17 | [IMTS-Mixer](https://arxiv.org/abs/2502.11816) | ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | Arxiv 2025 | IMTS-Mixer: Mixer-Networks for Irregular Multivariate Time Series Forecasting | [IMTS-Mixer](https://anonymous.4open.science/r/IMTS-Mixer-D63C/) |
| 25-02-20 | [TimeDistill](https://arxiv.org/abs/2502.15016) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | TimeDistill: Efficient Long-Term Time Series Forecasting with MLP via Cross-Architecture Distillation | None |
| 25-02-24 | [ReFocus](https://arxiv.org/abs/2502.16890) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | ReFocus: Reinforcing Mid-Frequency and Key-Frequency Modeling for Multivariate Time Series Forecasting | [refocus](https://github.com/levi-ackman/refocus) |
| 25-02-27 | [FIA-Net](https://arxiv.org/abs/2502.19983) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Efficient Time Series Forecasting via Hyper-Complex Models and Frequency Aggregation | [FIA-Net](https://anonymous.4open.science/r/research-1803/) |
| 25-02-28 | [UltraSTF](https://arxiv.org/abs/2502.20634) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2025 | A Compact Model for Large-Scale Time Series Forecasting | None |
| 25-03-04 | [CDFM](https://arxiv.org/abs/2503.02609) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | Lightweight Channel-wise Dynamic Fusion Model: Non-stationary Time Series Forecasting via Entropy Analysis | None |
| 25-03-30 | [SFNN](https://arxiv.org/abs/2503.23621) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | Simple Feedfoward Neural Networks are Almost All You Need for Time Series Forecasting | None |
| 25-04-02 | [DRAN](https://arxiv.org/abs/2504.01531) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2025 | DRAN: A Distribution and Relation Adaptive Network for Spatio-temporal Forecasting | None |
| 25-04-11 | [FilterTS](https://doi.org/10.1609/aaai.v39i20.35438) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2025 | FilterTS: Comprehensive Frequency Filtering for Multivariate Time Series Forecasting | [FilterTS](https://github.com/wyl010607/FilterTS) |
| 25-05-01 | [AiT](https://arxiv.org/abs/2505.00590) | ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | Arxiv 2025 | Unlocking the Potential of Linear Networks for Irregular Multivariate Time Series Forecasting | None |
| 25-05-07 | [RAFT](https://arxiv.org/abs/2505.04163) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICML 2025 | Retrieval Augmented Time Series Forecasting | [RAFT](https://github.com/archon159/RAFT) |
| 25-05-12 | [OLinear](https://arxiv.org/abs/2505.08550) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | OLinear: A Linear Model for Time Series Forecasting in Orthogonally Transformed Domain | [OLinear](https://anonymous.4open.science/r/OLinear) |
| 25-05-13 | [MDMixer](https://arxiv.org/abs/2505.08199) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | A Multi-scale Representation Learning Framework for Long-Term Time Series Forecasting | None |
| 25-05-15 | [ALinear](https://arxiv.org/abs/2505.10172) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | Does Scaling Law Apply in Time Series Forecasting? | None |
| 25-05-16 | [APN](https://arxiv.org/abs/2505.11250) | ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | Arxiv 2025 | Rethinking Irregular Time Series Forecasting: A Simple yet Effective Baseline | None |
| 25-05-17 | [WaveTS](https://arxiv.org/abs/2505.11781) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | Multi-Order Wavelet Derivative Transform for Deep Time Series Forecasting | None |
| 25-05-20 | [CRAFT](https://arxiv.org/abs/2505.13896) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | CRAFT: Time Series Forecasting with Cross-Future Behavior Awareness | [CRAFT](https://github.com/CRAFTinTSF/CRAFT) |


</details>

<details><summary><h2 style="display: inline;">TCN/CNN.</h2></summary>

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
| 23-10-01 | [PatchMixer](https://browse.arxiv.org/abs/2310.00655) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | PatchMixer: A Patch-Mixing Architecture for Long-Term Time Series Forecasting | [PatchMixer](https://github.com/Zeying-Gong/PatchMixer) |
| 23-11-01 | [WinNet](https://arxiv.org/abs/2311.00214) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2023 | WinNet:time series forecasting with a window-enhanced period extracting and interacting | None |
| 23-11-27 | [ModernTCN](https://openreview.net/forum?id=vpJMJerXHU)🌟 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2024 | [ModernTCN: A Modern Pure Convolution Structure for General Time Series Analysis](https://zhuanlan.zhihu.com/p/668946041) | [ModernTCN](https://github.com/luodhhh/ModernTCN) |
| 23-11-27 | [UniRepLKNet](https://arxiv.org/abs/2311.15599) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2023 | UniRepLKNet: A Universal Perception Large-Kernel ConvNet for Audio, Video, Point Cloud, Time-Series and Image Recognition | [UniRepLKNet](https://github.com/ailab-cvc/unireplknet) |
| 24-03-03 | [ConvTimeNet](https://arxiv.org/abs/2403.01493) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | WWW 2025 | ConvTimeNet: A Deep Hierarchical Fully Convolutional Model for Multivariate Time Series Analysis | [ConvTimeNet](https://github.com/Mingyue-Cheng/ConvTimeNet) |
| 24-05-20 | [ATVCNet](https://arxiv.org/abs/2405.12038) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024| ATVCNet: Adaptive Extraction Network for Multivariate Long Sequence Time-Series Forecasting |  None  |
| 24-05-24 | [FTMixer](https://arxiv.org/abs/2405.15256) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024| FTMixer: Frequency and Time Domain Representations Fusion for Time Series Modeling | [FTMixer](https://github.com/FMLYD/FTMixer)  |
| 24-10-07 | [TimeCNN](https://arxiv.org/abs/2410.04853) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024| TimeCNN: Refining Cross-Variable Interaction on Time Point for Time Series Forecasting | None |
| 24-10-21 | [TimeMixer++](https://arxiv.org/abs/2410.16032) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2025| [TimeMixer++: A General Time Series Pattern Machine for Universal Predictive Analysis](https://zhuanlan.zhihu.com/p/12926871013) | None |
| 24-11-07 | [EffiCANet](https://arxiv.org/abs/2411.04669) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024| EffiCANet: Efficient Time Series Forecasting with Convolutional Attention | None |
| 24-12-23 | [xPatch](https://arxiv.org/abs/2412.17323) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | AAAI 2025 | xPatch: Dual-Stream Time Series Forecasting with Exponential Seasonal-Trend Decomposition | [xPatch](https://github.com/stitsyuk/xpatch) |
| 25-01-23 | [TVNet](https://openreview.net/forum?id=MZDdTzN6Cy) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2025 | TVNet: A Novel Time Series Analysis Method Based on Dynamic Convolution and 3D-Variation | None |

</details>


<details><summary><h2 style="display: inline;">GNN.</h2></summary>

Date | Method | Type | Conference | Paper Title and Paper Interpretation (In Chinese) | Code |
| ---- | ------ | ------ | ---------- | ------------------------------------------------- | ---- |
| 17-09-14 | [STGCN](https://arxiv.org/abs/1709.04875)🌟🔥 | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | IJCAI 2018 | Spatio-Temporal Graph Convolutional Networks: A Deep Learning Framework for Traffic Forecasting | [STGCN](https://github.com/VeritasYin/STGCN_IJCAI-18) |
| 19-05-31 | [Graph WaveNet](https://arxiv.org/abs/1906.00121)🔥 | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | IJCAI 2019 | Graph WaveNet for Deep Spatial-Temporal Graph Modeling | [Graph-WaveNet](https://github.com/nnzhan/Graph-WaveNet) |
| 19-07-17 | [ASTGCN](https://ojs.aaai.org/index.php/AAAI/article/view/3881)🔥 | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2019 | Attention Based Spatial-Temporal Graph Convolutional Networks for Traffic Flow Forecasting | [ASTGCN](https://github.com/guoshnBJTU/ASTGCN-r-pytorch) |
| 20-04-03 | [SLCNN](https://ojs.aaai.org/index.php/AAAI/article/view/5470)🔥 | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2020 | Spatio-Temporal Graph Structure Learning for Traffic Forecasting | None |
| 20-04-03 | [GMAN](https://ojs.aaai.org/index.php/AAAI/article/view/5477)🔥 | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2020 | GMAN: A Graph Multi-Attention Network for Traffic Prediction | [GMAN](https://github.com/zhengchuanpan/GMAN) |
| 20-05-03 | [MTGNN](https://arxiv.org/abs/2005.01165)🌟🔥 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2020 | Connecting the Dots: Multivariate Time Series Forecasting with Graph Neural Networks | [MTGNN](https://github.com/nnzhan/MTGNN)  |
| 20-09-26 | [AGCRN](https://proceedings.neurips.cc/paper_files/paper/2020/file/ce1aad92b939420fc17005e5461e6f48-Paper.pdf)🌟🔥 | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | NIPS 2020 | Adaptive Graph Convolutional Recurrent Network for Traffic Forecasting | [AGCRN](https://github.com/LeiBAI/AGCRN) |
| 21-03-13 | [StemGNN](https://arxiv.org/abs/2103.07719)🌟🔥 | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2020 | Spectral Temporal Graph Neural Network for Multivariate Time-series Forecasting | [StemGNN](https://github.com/microsoft/StemGNN) |
| 22-05-16 | [TPGNN](https://openreview.net/forum?id=pMumil2EJh) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2022 | Multivariate Time-Series Forecasting with Temporal Polynomial Graph Neural Networks | [TPGNN](https://github.com/zyplanet/TPGNN) |
| 22-06-18 | [D2STGNN](https://arxiv.org/abs/2206.09112) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | VLDB 2022 | Decoupled Dynamic Spatial-Temporal Graph Neural Network for Traffic Forecasting | [D2STGNN](https://github.com/zezhishao/d2stgnn) |  
| 23-05-12 | [DDGCRN](https://www.sciencedirect.com/science/article/abs/pii/S0031320323003710?via%3Dihub) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | PR 2023 | A Decomposition Dynamic graph convolutional recurrent network for traffic forecasting | [DDGCRN](https://github.com/wengwenchao123/DDGCRN) |
| 23-05-30 | [HiGP](https://arxiv.org/abs/2305.19183) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | ICML 2024 | Graph-based Time Series Clustering for End-to-End Hierarchical Forecasting | None |
| 23-07-10 | [NexuSQN](https://arxiv.org/abs/2307.01482) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2023 | Nexus sine qua non: Essentially connected neural networks for spatial-temporal forecasting of multivariate time series | None |
| 23-11-10 | [FourierGNN](https://arxiv.org/abs/2311.06190) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2023 | FourierGNN: Rethinking Multivariate Time Series Forecasting from a Pure Graph Perspective | [FourierGNN](https://github.com/aikunyi/FourierGNN) |
| 23-12-05 | [SAMSGL](https://arxiv.org/abs/2312.02646) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | TETCI 2023 | SAMSGL: Series-Aligned Multi-Scale Graph Learning for Spatio-Temporal Forecasting | None |
| 23-12-27 | [TGCRN](https://arxiv.org/abs/2312.16403) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | ICDE 2024 | Learning Time-aware Graph Structures for Spatially Correlated Time Series Forecasting | None |
| 23-12-27 | [FCDNet](https://arxiv.org/abs/2312.16450) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2023 | FCDNet: Frequency-Guided Complementary Dependency Modeling for Multivariate Time-Series Forecasting | [FCDNet](https://github.com/oncecwj/fcdnet) |
| 23-12-31 | [MSGNet](https://arxiv.org/abs/2401.00423) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2024 | MSGNet: Learning Multi-Scale Inter-Series Correlations for Multivariate Time Series Forecasting | [MSGNet](https://github.com/YoZhibo/MSGNet) |
| 24-01-15 | [RGDAN](https://www.sciencedirect.com/science/article/abs/pii/S0893608023007542?via%3Dihub) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | NN 2024 | RGDAN: A random graph diffusion attention network for traffic prediction | [RGDAN](https://github.com/wengwenchao123/RGDAN) |
| 24-01-16 | [BiTGraph](https://openreview.net/forum?id=O9nZCwdGcG) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | ICLR 2024 | Biased Temporal Convolution Graph Network for Time Series Forecasting with Missing Values | [BiTGraph](https://github.com/chenxiaodanhit/BiTGraph) |
| 24-01-24 | [TMP](https://arxiv.org/abs/2401.13157) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | AAAI 2024 | Time-Aware Knowledge Representations of Dynamic Objects with Multidimensional Persistence | None |
| 24-02-16 | [HD-TTS](https://arxiv.org/abs/2402.10634) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | ICML 2024 | Graph-based Forecasting with Missing Data through Spatiotemporal Downsampling | [hdtts](https://github.com/marshka/hdtts) |
| 24-05-02 | [T-PATCHGNN](https://openreview.net/forum?id=UZlMXUGI6e) | ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | ICML 2024 | Irregular Multivariate Time Series Forecasting: A Transformable Patching Graph Neural Networks Approach | [t-PatchGNN](https://github.com/usail-hkust/t-PatchGNN) |
| 24-05-17 | [HimNet](https://arxiv.org/abs/2405.10800) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | KDD 2024 | Heterogeneity-Informed Meta-Parameter Learning for Spatiotemporal Time Series Forecasting | [HimNet](https://github.com/XDZhelheim/HimNet) |
| 24-05-28 | [GFC-GNN](https://arxiv.org/abs/2405.18036) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | ForecastGrapher: Redefining Multivariate Time Series Forecasting with Graph Neural Networks | None |
| 24-06-18 | [SAGDFN](https://arxiv.org/abs/2406.12282) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | ICDE 2024 | SAGDFN: A Scalable Adaptive Graph Diffusion Forecasting Network for Multivariate Time Series Forecasting | None |
| 24-10-17 | [GNeuralFlow](https://arxiv.org/abs/2410.14030) | ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | NIPS 2024 | Graph Neural Flows for Unveiling Systemic Interactions Among Irregularly Sampled Time Series | [GNeuralFlow](https://github.com/gmerca/GNeuralFlow) |
| 24-10-24 | [TEAM](https://arxiv.org/abs/2410.19192) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | VLDB 2025 | TEAM: Topological Evolution-aware Framework for Traffic Forecasting | [TEAM](https://github.com/kvmduc/TEAM-topo-evo-traffic-forecasting) |
| 25-01-22 | [TimeFilter](https://arxiv.org/abs/2501.13041) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2025 | TimeFilter: Patch-Specific Spatial-Temporal Graph Filtration for Time Series Forecasting | [TimeFilter](https://github.com/troubadour000/timefilter) |
| 25-02-21 | [EVTSF](https://arxiv.org/abs/2502.15296) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2025 | Beyond Fixed Variables: Expanding-variate Time Series Forecasting via Flat Scheme and Spatio-temporal Focal Learning | None |
| 25-05-07 | [STRGCN](https://arxiv.org/abs/2505.04167) | ![Irregular_time_series](https://img.shields.io/badge/-Irregular-orange) | Arxiv 2025 | STRGCN: Capturing Asynchronous Spatio-Temporal Dependencies for Irregular Multivariate Time Series Forecasting | None |


</details>


<details><summary><h2 style="display: inline;">SSM (State Space Model).</h2></summary>

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 18-05-18 | [DSSM](https://papers.nips.cc/paper/8004-deep-state-space-models-for-time-series-forecasting) | NIPS 2018 | Deep State Space Models for Time Series Forecasting | None   |
| 19-08-10 | [DSSMF](https://arxiv.org/abs/2102.00397) | IJCAI 2019 | Learning Interpretable Deep State Space Model for Probabilistic Time Series Forecasting | None   |
| 22-08-19 | [SSSD](https://arxiv.org/abs/2208.09399) | TMLR 2022 | Diffusion-based Time Series Imputation and Forecasting with Structured State Space Models | [SSSD](https://github.com/AI4HealthUOL/SSSD) |
| 22-09-22 | [SpaceTime](https://arxiv.org/abs/2303.09489) | ICLR 2023 | Effectively Modeling Time Series with Simple Discrete State Spaces | [SpaceTime](https://github.com/hazyresearch/spacetime)   |
| 22-12-24 | [LS4](https://arxiv.org/abs/2212.12749) | ICML 2023 | Deep Latent State Space Models for Time-Series Generation | [LS4](https://github.com/alexzhou907/ls4) |
| 24-02-18 | [Attraos](https://arxiv.org/abs/2402.11463) | Arxiv 2024 | Attractor Memory for Long-Term Time Series Forecasting: A Chaos Perspective | None |
| 24-03-14 | [TimeMachine](https://arxiv.org/abs/2403.09898) | Arxiv 2024 | TimeMachine: A Time Series is Worth 4 Mambas for Long-term Forecasting | None |
| 24-03-17 | [S-D-Mamba](https://arxiv.org/abs/2403.11144) | Arxiv 2024 | Is Mamba Effective for Time Series Forecasting? | [S-D-Mamba](https://github.com/wzhwzhwzh0921/S-D-Mamba) |
| 24-03-22 | [SiMBA](https://arxiv.org/abs/2403.15360) | Arxiv 2024 | SiMBA: Simplified Mamba-Based Architecture for Vision and Multivariate Time series | [Simba](https://github.com/badripatro/Simba) |
| 24-03-29 | [TSM2](https://arxiv.org/abs/2403.19888) | Arxiv 2024 | MambaMixer: Efficient Selective State Space Models with Dual Token and Channel Selection | [M2](https://github.com/MambaMixer/M2) |
| 24-04-23 | [Mambaformer](https://arxiv.org/abs/2404.14757) | Arxiv 2024 | Integrating Mamba and Transformer for Long-Short Range Time Series Forecasting | [Mambaformer](https://github.com/XiongxiaoXu/Mambaformer-in-Time-Series) |
| 24-04-24 | [Bi-Mamba4TS](https://arxiv.org/abs/2404.15772) | Arxiv 2024 | Bi-Mamba4TS: Bidirectional Mamba for Time Series Forecasting | None |
| 24-05-11 | [DTMamba](https://arxiv.org/abs/2405.07022) | Arxiv 2024 | DTMamba: Dual Twin Mamba for Time Series Forecasting | None |
| 24-05-25 | [Time-SSM](https://arxiv.org/abs/2405.16312) | Arxiv 2024 | Time-SSM: Simplifying and Unifying State Space Models for Time Series Forecasting | None |
| 24-05-26 | [MambaTS](https://arxiv.org/abs/2405.16440) | Arxiv 2024 | MambaTS: Improved Selective State Space Models for Long-term Time Series Forecasting | None |
| 24-06-06 | [Chimera](https://arxiv.org/abs/2406.04320) | Arxiv 2024 | Chimera: Effectively Modeling Multivariate Time Series with 2-Dimensional State Space Models | None |
| 24-06-08 | [C-Mamba](https://arxiv.org/abs/2406.05316) | Arxiv 2024 | C-Mamba: Channel Correlation Enhanced State Space Models for Multivariate Time Series Forecasting | [CMamba](https://github.com/Prismadic/simple-CMamba) |
| 24-07-20 | [FMamba](https://arxiv.org/abs/2407.14814) | Arxiv 2024 | FMamba: Mamba based on Fast-attention for Multivariate Time-series Forecasting | None |
| 24-08-17 | [SSL](https://arxiv.org/abs/2408.09120) | Arxiv 2024 | Time Series Analysis by State Space Learning | None |
| 24-08-22 | [SAMBA](https://arxiv.org/abs/2408.12068) | Arxiv 2024 | Simplified Mamba with Disentangled Dependency Encoding for Long-Term Time Series Forecasting | None |
| 24-08-28 | [MoU](https://arxiv.org/abs/2408.15997) | Arxiv 2024 | Mamba or Transformer for Time Series Forecasting? Mixture of Universals (MoU) Is All You Need | [MoU](https://github.com/lunaaa95/mou) |
| 24-09-26 | [SAMBA](https://arxiv.org/abs/2410.03707) | Arxiv 2024 | Mamba Meets Financial Markets: A Graph-Mamba Approach for Stock Price Prediction | [SAMBA](https://github.com/Ali-Meh619/SAMBA) |
| 24-10-04 | [LinOSS](https://arxiv.org/abs/2410.03943) | Arxiv 2024 | Oscillatory State-Space Models | None |
| 24-10-15 | [UmambaTSF](https://arxiv.org/abs/2410.11278) | Arxiv 2024 | UmambaTSF: A U-shaped Multi-Scale Long-Term Time Series Forecasting Method Using Mamba | None |
| 24-10-28 | [FACTS](https://arxiv.org/abs/2410.20922) | Arxiv 2024 | FACTS: A Factored State-Space Framework For World Modelling | [FACTS](https://github.com/nanboli/facts) |
| 24-10-30 | [SOR-Mamba](https://arxiv.org/abs/2410.23356) | Arxiv 2024 | Sequential Order-Robust Mamba for Time Series Forecasting | [SOR-Mamba](https://github.com/seunghan96/SOR-Mamba) |
| 24-11-26 | [MTS-UNMixers](https://arxiv.org/abs/2411.17770) | Arxiv 2024 | MTS-UNMixers: Multivariate Time Series Forecasting via Channel-Time Dual Unmixing | [MTS-UNMixers](https://github.com/zhu-0108/mts-unmixers) |
| 24-12-01 | [DSSRNN](https://arxiv.org/abs/2412.00994) | Arxiv 2024 | DSSRNN: Decomposition-Enhanced State-Space Recurrent Neural Network for Time-Series Analysis | [DSSRNN](https://github.com/ahmad-shirazi/DSSRNN) |
| 25-01-23 | [ACSSM](https://openreview.net/forum?id=8zJRon6k5v) | ICLR 2025 | Amortized Control of Continuous State Space Feynman-Kac Model for Irregular Time Series | None |
| 25-01-23 | [S4M](https://openreview.net/forum?id=BkftcwIVmR) | ICLR 2025 | S4M: S4 for multivariate time series forecasting with Missing values | [S4M](https://github.com/WINTERWEEL/S4M) |
| 25-03-13 | [Mamba-ProbTSF](https://arxiv.org/abs/2503.10873) | Arxiv 2025 | Mamba time series forecasting with uncertainty propagation | [mamba-probtsf](https://github.com/pessoap/mamba-probtsf) |
| 25-04-02 | [Attention Mamba](https://arxiv.org/abs/2504.02013) | Arxiv 2025 | Attention Mamba: Time Series Modeling with Adaptive Pooling Acceleration and Receptive Field Enhancements | None |
| 25-04-10 | [ms-Mamba](https://arxiv.org/abs/2504.07654) | Arxiv 2025 | ms-Mamba: Multi-scale Mamba for Time-Series Forecasting | None |


</details>

<details><summary><h2 style="display: inline;">Generation Model.</h2></summary>

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 20-02-14 | [MAF](https://arxiv.org/abs/2002.06103)🌟 | ICLR 2021 | [Multivariate Probabilitic Time Series Forecasting via Conditioned Normalizing Flows](https://zhuanlan.zhihu.com/p/615795048) | [MAF](https://github.com/zalandoresearch/pytorch-ts)   |
| 21-01-18 | [TimeGrad](https://arxiv.org/abs/2101.12072)🌟 | ICML 2021 | [Autoregressive Denoising Diffusion Models for Multivariate Probabilistic Time Series Forecasting](https://zhuanlan.zhihu.com/p/615858953) | [TimeGrad](https://github.com/zalandoresearch/pytorch-ts)   |
| 21-07-07 | [CSDI](https://arxiv.org/abs/2107.03502) | NIPS 2021 | [CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series Imputation](https://zhuanlan.zhihu.com/p/615998087) | [CSDI](https://github.com/ermongroup/csdi)   |
| 22-05-16 | [MANF](https://arxiv.org/abs/2205.07493)| Arxiv 2022 |Multi-scale Attention Flow for Probabilistic Time Series Forecasting| None |
| 22-05-16 | [D3VAE](https://arxiv.org/abs/2301.03028) | NIPS 2022 | Generative Time Series Forecasting with Diffusion, Denoise, and Disentanglement | [D3VAE](https://github.com/paddlepaddle/paddlespatial)   |
| 22-12-28 | [Hier-Transformer-CNF](https://arxiv.org/abs/2212.13706) | Arxiv 2022 | End-to-End Modeling Hierarchical Time Series Using Autoregressive Transformer and Conditional Normalizing Flow based Reconciliation | None   |
| 23-03-13 | [HyVAE](https://arxiv.org/abs/2303.07048) | Arxiv 2023 | Hybrid Variational Autoencoder for Time Series Forecasting | None   |
| 23-06-03 | [DYffusion](https://arxiv.org/abs/2306.01984) | NIPS 2023 | DYffusion: A Dynamics-informed Diffusion Model for Spatiotemporal Forecasting | [DYffusion](https://github.com/Rose-STL-Lab/dyffusion)   |
| 23-06-05 | [WIAE](https://arxiv.org/abs/2306.03782) | Arxiv 2023 | Non-parametric Probabilistic Time Series Forecasting via Innovations Representation | None   |
| 23-06-08 | [TimeDiff](https://arxiv.org/abs/2306.05043)🌟 | ICML 2023 | Non-autoregressive Conditional Diffusion Models for Time Series Prediction | None |
| 23-07-21 | [TSDiff](https://arxiv.org/abs/2307.11494) | NIPS 2023 | Predict, Refine, Synthesize: Self-Guiding Diffusion Models for Probabilistic Time Series Forecasting | [TSDiff](https://github.com/amazon-science/unconditional-time-series-diffusion) |
| 24-01-16 | [FTS-Diffusion](https://openreview.net/forum?id=CdjnzWsQax) | ICLR 2024 | Generative Learning for Financial Time Series with Irregular and Scale-Invariant Patterns | None |
| 24-01-16 | [MG-TSD](https://openreview.net/forum?id=CZiY6OLktd) | ICLR 2024 | MG-TSD: Multi-Granularity Time Series Diffusion Models with Guided Learning Process | [MG-TSD](https://github.com/Hundredl/MG-TSD) |
| 24-01-16 | [TMDM](https://openreview.net/forum?id=qae04YACHs) | ICLR 2024 | Transformer-Modulated Diffusion Models for Probabilistic Multivariate Time Series Forecasting | None |
| 24-01-16 | [mr-Diff](https://openreview.net/forum?id=mmjnr0G8ZY) | ICLR 2024 | Multi-Resolution Diffusion Models for Time Series Forecasting | None |
| 24-01-16 | [Diffusion-TS](https://openreview.net/forum?id=4h1apFjO99) | ICLR 2024 | Diffusion-TS: Interpretable Diffusion for General Time Series Generation | [Diffusion-TS](https://github.com/Y-debug-sys/Diffusion-TS) | 
| 24-01-16 | [SpecSTG](https://arxiv.org/abs/2401.08119) | Arxiv 2024 | SpecSTG: A Fast Spectral Diffusion Framework for Probabilistic Spatio-Temporal Traffic Forecasting | [SpecSTG](https://anonymous.4open.science/r/SpecSTG) | 
| 24-02-03 | [GenFormer](https://arxiv.org/abs/2402.02010) | Arxiv 2024 | GenFormer: A Deep-Learning-Based Approach for Generating Multivariate Stochastic Processes | [GenFormer](https://github.com/Zhaohr1990/GenFormer) | 
| 24-01-30 | [IN-Flow](https://arxiv.org/abs/2401.16777) | Arxiv 2024 | Addressing Distribution Shift in Time Series Forecasting with Instance Normalization Flows | None | 
| 24-03-24 | [LDT](https://ojs.aaai.org/index.php/AAAI/article/view/29085) | AAAI 2024 | Latent Diffusion Transformer for Probabilistic Time Series Forecasting | None | 
| 24-06-04 | [GPD](https://arxiv.org/abs/2406.02212) | Arxiv 2024 | Generative Pre-Trained Diffusion Paradigm for Zero-Shot Time Series Forecasting | None |
| 24-06-05 | [StochDiff](https://arxiv.org/abs/2406.02827) | Arxiv 2024 | Stochastic Diffusion: A Diffusion Probabilistic Model for Stochastic Time Series Forecasting | None | 
| 24-06-11 | [ProFITi](https://arxiv.org/abs/2406.07246) | Arxiv 2024 | Marginalization Consistent Mixture of Separable Flows for Probabilistic Irregular Time Series Forecasting | None | 
| 24-09-03 | [TimeDiT](https://arxiv.org/abs/2409.02322) | Arxiv 2024 | TimeDiT: General-purpose Diffusion Transformers for Time Series Foundation Model | None |
| 24-09-18 | [SI](https://arxiv.org/abs/2409.11684) | Arxiv 2024 | Recurrent Interpolants for Probabilistic Time Series Prediction | None |
| 24-09-27 | [Bim-Diff](https://arxiv.org/abs/2409.18491) | Arxiv 2024 | Treating Brain-inspired Memories as Priors for Diffusion Model to Forecast Multivariate Time Series | None |
| 24-10-03 | [CCDM](https://arxiv.org/abs/2410.02168) | Arxiv 2024 | Channel-aware Contrastive Conditional Diffusion for Multivariate Probabilistic Time Series Forecasting | [CCDM](https://github.com/LSY-Cython/CCDM) |
| 24-10-03 | [TSFlow](https://arxiv.org/abs/2410.03024) | ICLR 2025 | Flow Matching with Gaussian Process Priors for Probabilistic Time Series Forecasting | None |
| 24-10-08 | [TimeDART](https://arxiv.org/abs/2410.05711) | Arxiv 2024 | Diffusion Auto-regressive Transformer for Effective Self-supervised Time Series Forecasting | [TimeDART](https://github.com/Melmaphother/TimeDART) |
| 24-10-17 | [FDF](https://arxiv.org/abs/2410.13253) | Arxiv 2024 | FDF: Flexible Decoupled Framework for Time Series Forecasting with Conditional Denoising and Polynomial Modeling | [FDF](https://github.com/zjt-gpu/FDF) |
| 24-10-18 | [ANT](https://arxiv.org/abs/2410.14488) | NIPS 2024 | ANT: Adaptive Noise Schedule for Time Series Diffusion Models | [ANT](https://github.com/seunghan96/ANT) |
| 24-10-21 | [REDI](https://dl.acm.org/doi/abs/10.1145/3627673.3679808) | CIKM 2024 | REDI: Recurrent Diffusion Model for Probabilistic Time Series Forecasting | None |
| 24-10-24 | [RATD](https://arxiv.org/abs/2410.18712) | NIPS 2024 | Retrieval-Augmented Diffusion Models for Time Series Forecasting | [RATD](https://github.com/stanliu96/RATD) |
| 24-11-02 | [ProGen](https://arxiv.org/abs/2411.01267) | Arxiv 2024 | ProGen: Revisiting Probabilistic Spatial-Temporal Time Series Forecasting from a Continuous Generative Perspective Using Stochastic Differential Equations | None |
| 24-11-07 | [S2DBM](https://arxiv.org/abs/2411.04491) | Arxiv 2024 | Series-to-Series Diffusion Bridge Model | None |
| 24-11-12 | [FM-TS](https://arxiv.org/abs/2411.07506) | Arxiv 2024 | FM-TS: Flow Matching for Time Series Generation | [FMTS](https://github.com/UNITES-Lab/FMTS) |
| 24-12-12 | [ARMD](https://arxiv.org/abs/2412.09328) | AAAI 2025 | Auto-Regressive Moving Diffusion Models for Time Series Forecasting | [ARMD](https://github.com/daxin007/ARMD) |
| 25-01-23 | [D3U](https://openreview.net/forum?id=HdUkF1Qk7g) | ICLR 2025 | Diffusion-based Decoupled Deterministic and Uncertain Framework for Probabilistic Multivariate Time Series Forecasting | None |
| 25-02-16 | [LDM4TS](https://arxiv.org/abs/2502.14887) | Arxiv 2025 | Vision-Enhanced Time Series Forecasting via Latent Diffusion Models | None |
| 25-02-26 | [Hutch++](https://arxiv.org/abs/2502.18808) | AISTATS 2025 | Optimal Stochastic Trace Estimation in Generative Modeling | [Hutch++](https://github.com/xinyangATK/GenHutch-plus-plus) |
| 25-03-02 | [DyDiff](https://arxiv.org/abs/2503.00951) | ICLR 2025 | Dynamical Diffusion: Learning Temporal Dynamics with Diffusion Models | [DyDiff](https://github.com/thuml/dynamical-diffusion) |
| 25-05-07 | [NsDiff](https://arxiv.org/abs/2505.04278) | ICML 2025 | Non-stationary Diffusion For Probabilistic Time Series Forecasting | [NsDiff](https://github.com/wwy155/NsDiff) |
| 25-05-16 | [FALDA](https://arxiv.org/abs/2505.11306) | Arxiv 2025 | Effective Probabilistic Time Series Forecasting with Fourier Adaptive Noise-Separated Diffusion | None |


</details>


<details><summary><h2 style="display: inline;">Time-index.</h2></summary>

Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|----|-----|-----|-----
| 17-05-25 | [ND](https://arxiv.org/abs/1705.09137) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | TNNLS 2017 | [Neural Decomposition of Time-Series Data for Effective Generalization](https://zhuanlan.zhihu.com/p/574742701)  | None |
| 17-08-25 | [Prophet](https://peerj.com/preprints/3190/)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen)  | TAS 2018 | [Forecasting at Scale](https://facebook.github.io/prophet/) | [Prophet](https://github.com/facebook/prophet)   |
| 22-07-13 | [DeepTime](https://arxiv.org/abs/2207.06046) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICML 2023 | [Learning Deep Time-index Models for Time Series Forecasting](https://blog.salesforceairesearch.com/deeptime-meta-learning-time-series-forecasting/) | [DeepTime](https://github.com/salesforce/DeepTime) |
| 23-06-09 | [TimeFlow](https://arxiv.org/abs/2306.05880) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen)  | Arxiv 2023 | Time Series Continuous Modeling for Imputation and Forecasting with Implicit Neural Representations | None |
| 24-01-16 | [DAM](https://openreview.net/forum?id=4NhMhElWqP)🌟 | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen)  | ICLR 2024 | DAM: Towards A Foundation Model for Time Series Forecasting | None |

</details>


<details><summary><h2 style="display: inline;">Plug and Play (Model-Agnostic).</h2></summary>

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 19-02-21 | [DAIN](https://arxiv.org/abs/1902.07892)🌟 | TNNLS 2020 | Deep Adaptive Input Normalization for Time Series Forecasting | [DAIN](https://github.com/passalis/dain)   |
| 19-09-19 | [DILATE](https://arxiv.org/abs/1909.09020) | NIPS 2019 | Shape and Time Distortion Loss for Training Deep Time Series Forecasting Models | [DILATE](https://github.com/vincent-leguen/DILATE) |
| 21-07-19 | [TAN](https://arxiv.org/abs/2107.09031) | NIPS 2021 | Topological Attention for Time Series Forecasting | [TAN](https://github.com/plus-rkwitt/TAN)   |
| 21-09-29 | [RevIN](https://openreview.net/forum?id=cGDAkQo1C0p)🌟🔥 | ICLR 2022 | [Reversible Instance Normalization for Accurate Time-Series Forecasting against Distribution Shift](https://zhuanlan.zhihu.com/p/473553126) | [RevIN](https://github.com/ts-kim/RevIN)   |
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
| 23-10-23 | [EDAIN](https://arxiv.org/abs/2310.14720) | Arxiv 2023 | Extended Deep Adaptive Input Normalization for Preprocessing Time Series Data for Neural Networks | [EDAIN](https://github.com/marcusGH/edain_paper) |
| 23-10-23 | [SOLID](https://arxiv.org/abs/2310.14838) | KDD 2024 | Calibration of Time-Series Forecasting: Detecting and Adapting Context-Driven Distribution Shift | [SOLID](https://github.com/half111/calibration_cds) |
| 23-11-19 | [TimeSQL](https://arxiv.org/abs/2311.11285) | Arxiv 2023 | TimeSQL: Improving Multivariate Time Series Forecasting with Multi-Scale Patching and Smooth Quadratic Loss | None |
| 24-01-16 | [LIFT](https://openreview.net/forum?id=JiTVtCUOpS) | ICLR 2024 | Rethinking Channel Dependence for Multivariate Time Series Forecasting: Learning from Leading Indicators | None |
| 24-01-16 | [RobustTSF](https://openreview.net/forum?id=ltZ9ianMth) | ICLR 2024 | RobustTSF: Towards Theory and Design of Robust Time Series Forecasting with Anomalies | [RobustTSF](https://openreview.net/attachment?id=ltZ9ianMth&name=supplementary_material) |
| 24-02-04 | [FreDF](https://arxiv.org/abs/2402.02399)🌟 | ICLR 2025 | [Label Correlation Biases Direct Time Series Forecast](https://zhuanlan.zhihu.com/p/12974123752) | [FreDF](https://github.com/Master-PLC/FreDF) |
| 24-02-14 | [PDLS](https://arxiv.org/abs/2402.09373) | ICML 2024 | Transformers with Loss Shaping Constraints for Long-Term Time Series Forecasting | None |
| 24-02-20 | [Leddam](https://arxiv.org/abs/2402.12694) | ICML 2024 | Revitalizing Multivariate Time Series Forecasting: Learnable Decomposition with Inter-Series Dependencies and Intra-Series Variations Modeling | None |
| 24-03-01 | [InfoTime](https://arxiv.org/abs/2403.00869) | Arxiv 2024 | Enhancing Multivariate Time Series Forecasting with Mutual Information-driven Cross-Variable and Temporal Modeling | None |
| 24-03-13 | [wavelet-ML](https://arxiv.org/abs/2403.08630) | Arxiv 2024 | Leveraging Non-Decimated Wavelet Packet Features and Transformer Models for Time Series Forecasting | None |
| 24-03-31 | [CCM](https://arxiv.org/abs/2404.01340) | NIPS 2024 | From Similarity to Superiority: Channel Clustering for Time Series Forecasting | [TimeSeriesCCM](https://github.com/Graph-and-Geometric-Learning/TimeSeriesCCM) |
| 24-05-24 | [UnitNorm](https://arxiv.org/abs/2405.15903) | Arxiv 2024 | UnitNorm: Rethinking Normalization for Transformers in Time Series | [UnitNorm](https://anonymous.4open.science/r/UnitNorm-5B84) |
| 24-05-29 | [HCAN](https://arxiv.org/abs/2405.18975) | Arxiv 2024 | Hierarchical Classification Auxiliary Network for Time Series Forecasting | [HCAN](https://github.com/syrGitHub/HCAN) |
| 24-05-30 | [S3](https://arxiv.org/abs/2405.20082) | NIPS 2024 | Segment, Shuffle, and Stitch: A Simple Layer for Improving Time-Series Representations | [S3](https://github.com/shivam-grover/S3-TimeSeries) |
| 24-06-07 | [TDT](https://arxiv.org/abs/2406.04777) | Arxiv 2024 | TDT Loss Takes It All: Integrating Temporal Dependencies among Targets into Non-Autoregressive Time Series Forecasting | None |
| 24-06-19 | [Gaussian](https://arxiv.org/abs/2406.13871) | Arxiv 2024 | Robust Time Series Forecasting with Non-Heavy-Tailed Gaussian Loss-Weighted Sampler | None |
| 24-07-21 | [TimeInf](https://arxiv.org/abs/2407.15247) | Arxiv 2024 | TimeInf: Time Series Data Contribution via Influence Functions | [TimeInf](https://github.com/yzhang511/timeinf) |
| 24-07-24 | [C-LoRA](https://arxiv.org/abs/2407.17246) | CIKM 2024 | Channel-Aware Low-Rank Adaptation in Time Series Forecasting | [C-LoRA](https://github.com/tongnie/C-LoRA) |
| 24-08-20 | [Wave-Augs](https://arxiv.org/abs/2408.10951) | Arxiv 2024 | Wave-Mask/Mix: Exploring Wavelet-Based Augmentations for Time Series Forecasting | [Wave-Augs](https://github.com/jafarbakhshaliyev/Wave-Augs) |
| 24-08-27 | [Channel-wise Influence](https://arxiv.org/abs/2408.14763) | Arxiv 2024 | Channel-wise Influence: Estimating Data Influence for Multivariate Time Series | None |
| 24-09-10 | [AutoTSAug](https://arxiv.org/abs/2409.06282) | Arxiv 2024 | Learning Augmentation Policies from A Model Zoo for Time Series Forecasting | None |
| 24-09-23 | [MotifDisco](https://arxiv.org/abs/2409.15219) | Arxiv 2024 | MotifDisco: Motif Causal Discovery For Time Series Motifs | None |
| 24-09-25 | [FBM](https://openreview.net/forum?id=BAfKBkr8IP) | NIPS 2024 | Rethinking Fourier Transform from A Basis Functions Perspective for Long-term Time Series Forecasting | [FBM](https://github.com/runze1223/Fourier-Basis-Mapping) |
| 24-09-27 | [GLAFF](https://arxiv.org/abs/2409.18696) | NIPS 2024 | Rethinking the Power of Timestamps for Robust Time Series Forecasting: A Global-Local Fusion Perspective | [GLAFF](https://github.com/ForestsKing/GLAFF) |
| 24-09-30 | [FAN](https://arxiv.org/abs/2409.20371) | NIPS 2024 | Frequency Adaptive Normalization For Non-stationary Time Series Forecasting | [FAN](https://github.com/wayne155/FAN) |
| 24-10-02 | [FredNormer](https://arxiv.org/abs/2410.01860) | Arxiv 2024 | FredNormer: Frequency Domain Normalization for Non-stationary Time Series Forecasting | None |
| 24-10-03 | [FAN](https://arxiv.org/abs/2410.02675) | Arxiv 2024 | FAN: Fourier Analysis Networks | [FAN](https://github.com/YihongDong/FAN) |
| 24-10-04 | [GAS-Norm](https://arxiv.org/abs/2410.03935) | CIKM 2024 | GAS-Norm: Score-Driven Adaptive Normalization for Non-Stationary Time Series Forecasting in Deep Learning | [GAS-Norm](https://github.com/edo-urettini/GAS_Norm) |
| 24-10-09 | [TOI](https://arxiv.org/abs/2410.06652) | NIPS 2024 | Task-oriented Time Series Imputation Evaluation via Generalized Representers | [TOI](https://github.com/hkuedl/Task-Oriented-Imputation) |
| 24-10-19 | [FGL](https://arxiv.org/abs/2410.15217) | Arxiv 2024 | Future-Guided Learning: A Predictive Approach To Enhance Time-Series Forecasting | None |
| 24-10-28 | [BSA](https://arxiv.org/abs/2410.20772) | NIPS 2024 | Introducing Spectral Attention for Long-Range Dependency in Time Series Forecasting | [BSA](https://github.com/djlee1208/bsa_2024) |
| 24-10-30 | [DisenTS](https://arxiv.org/abs/2410.22981) | Arxiv 2024 | DisenTS: Disentangled Channel Evolving Pattern Modeling for Multivariate Time Series Forecasting | None |
| 24-10-30 | [CM](https://arxiv.org/abs/2410.23222) | Arxiv 2024 | Partial Channel Dependence with Channel Masks for Time Series Foundation Models | [CM](https://github.com/seunghan96/CM) |
| 24-10-31 | [MLP replace attention](https://arxiv.org/abs/2410.24023) | Arxiv 2024 | Approximate attention with MLP: a pruning strategy for attention-based model in multivariate time series forecasting | None |
| 24-11-12 | [RAF](https://arxiv.org/abs/2411.08249) | Arxiv 2024 | Retrieval Augmented Time Series Forecasting | [RAF](https://github.com/kutaytire/retrieval-augmented-time-series-forecasting) |
| 24-11-24 | [Freq-Synth](https://arxiv.org/abs/2411.15743) | Arxiv 2024 | Beyond Data Scarcity: A Frequency-Driven Framework for Zero-Shot Forecasting | None |
| 24-12-02 | [TimeLinear](https://arxiv.org/abs/2412.01557) | Arxiv 2024 | How Much Can Time-related Features Enhance Time Series Forecasting? | [TimeLinear](https://github.com/zclzcl0223/TimeLinear) |
| 24-12-06 | [WaveToken](https://arxiv.org/abs/2412.05244) | Arxiv 2024 | Enhancing Foundation Models for Time Series Forecasting via Wavelet-based Tokenization | None |
| 24-12-21 | [LDM](https://arxiv.org/abs/2412.16572) | Arxiv 2024 | Breaking the Context Bottleneck on Long Time Series Forecasting | [LDM](https://github.com/houyikai/ldm-logsparse-decomposable-multiscaling) |
| 24-12-27 | [PCA](https://arxiv.org/abs/2412.19423) | Arxiv 2024 | Revisiting PCA for time series reduction in temporal dimension | None |
| 24-12-30 | [TimeRAF](https://arxiv.org/abs/2412.20810) | Arxiv 2024 | TimeRAF: Retrieval-Augmented Foundation model for Zero-shot Time Series Forecasting | None |
| 25-01-02 | [HPO](https://arxiv.org/abs/2501.01394)| Arxiv 2025 | A Unified Hyperparameter Optimization Pipeline for Transformer-Based Time Series Forecasting Models | [HPO](https://github.com/jingjing-unilu/HPO_transformer_time_series) |
| 25-01-06 | [Sequence Complementor](https://arxiv.org/abs/2501.02735)| AAAI 2025 | Sequence Complementor: Complementing Transformers For Time Series Forecasting with Learnable Sequences | None |
| 25-01-09 | [TAFAS](https://arxiv.org/abs/2501.04970)| AAAI 2025 | Battling the Non-stationarity in Time Series Forecasting via Test-time Adaptation | [TAFAS](https://github.com/kimanki/TAFAS) |
| 25-01-23 | [PN-Train](https://openreview.net/forum?id=a9vey6B54y)| ICLR 2025 | Investigating Pattern Neurons in Urban Time Series Forecasting | [PN-Train](https://anonymous.4open.science/r/PN-Train) |
| 25-01-23 | [CoMRes](https://openreview.net/forum?id=bRa4JLPzii)| ICLR 2025 | CoMRes: Semi-Supervised Time Series Forecasting Utilizing Consensus Promotion of Multi-Resolution | None |
| 25-02-20 | [SCAM](https://arxiv.org/abs/2502.14704)| Arxiv 2025 | Not All Data are Good Labels: On the Self-supervised Labeling for Time Series Forecasting | [SCAM](https://anonymous.4open.science/r/SCAM-BDD3) |
| 25-03-02 | [PS_Loss](https://arxiv.org/abs/2503.00877)| Arxiv 2025 | Patch-wise Structural Loss for Time Series Forecasting | [PS_Loss](https://github.com/Dilfiraa/PS_Loss) |
| 25-03-25 | [Rejection](https://arxiv.org/abs/2503.19656)| Arxiv 2025 | Towards Reliable Time Series Forecasting under Future Uncertainty: Ambiguity and Novelty Rejection Mechanisms | None |
| 25-04-19 | [Pets](https://arxiv.org/abs/2504.14209)| Arxiv 2025 | Pets: General Pattern Assisted Architecture For Time Series Analysis | None |
| 25-04-24 | [Training Policy](https://arxiv.org/abs/2504.17493)| Arxiv 2025 | Goal-Oriented Time-Series Forecasting: Foundation Framework Design | None |
| 25-05-13 | [SPAT](https://arxiv.org/abs/2505.08768)| Arxiv 2025 | SPAT: Sensitivity-based Multihead-attention Pruning on Time Series Forecasting Models | [SPAT](https://anonymous.4open.science/r/SPAT-6042) |
| 25-05-16 | [X-Freq](https://arxiv.org/abs/2505.11567)| Arxiv 2025 | Beyond Time: Cross-Dimensional Frequency Supervision for Time Series Forecasting | None |
| 25-05-16 | [KNN-MTS](https://arxiv.org/abs/2505.11625)| TNNLS 2024 | Nearest Neighbor Multivariate Time Series Forecasting | [KNN-MTS](https://github.com/hlhang9527/KNN-MTS) |
| 25-05-20 | [Adaptive tokenization](https://arxiv.org/abs/2505.14411)| Arxiv 2025 | Byte Pair Encoding for Efficient Time Series Forecasting | None |
| 25-05-20 | [Toto](https://arxiv.org/abs/2505.14766)| Arxiv 2025 | This Time is Different: An Observability Perspective on Time Series Foundation Models | [Toto](https://github.com/datadog/toto) |
| 25-05-21 | [Adaptive Optimization](https://arxiv.org/abs/2505.15354)| Arxiv 2025 | Human in the Loop Adaptive Optimization for Improved Time Series Forecasting | None |



</details>


<details><summary><h2 style="display: inline;">LLM (Large Language Model).</h2></summary>

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 22-09-20 | [PromptCast](https://arxiv.org/abs/2210.08964) | TKDE 2023 | PromptCast: A New Prompt-based Learning Paradigm for Time Series Forecasting | [PISA](https://github.com/HaoUNSW/PISA) |
| 23-02-23 | [FPT](https://arxiv.org/abs/2302.11939) 🌟 | NIPS 2023 | [One Fits All: Power General Time Series Analysis by Pretrained LM](https://zhuanlan.zhihu.com/p/661884836) | [One-Fits-All](https://github.com/DAMO-DI-ML/NeurIPS2023-One-Fits-All)   |
| 23-05-17 | [LLMTime](https://arxiv.org/abs/2310.07820) | NIPS 2023 | [Large Language Models Are Zero-Shot Time Series Forecasters](https://zhuanlan.zhihu.com/p/661526823) | [LLMTime](https://github.com/ngruver/llmtime) |
| 23-08-16 | [TEST](https://arxiv.org/abs/2308.08241) | ICLR 2024 | TEST: Text Prototype Aligned Embedding to Activate LLM's Ability for Time Series | None |
| 23-08-16 | [LLM4TS](https://arxiv.org/abs/2308.08469) | Arxiv 2023 | LLM4TS: Two-Stage Fine-Tuning for Time-Series Forecasting with Pre-Trained LLMs | None |
| 23-10-03 | [Time-LLM](https://arxiv.org/abs/2310.01728) | ICLR 2024 | [Time-LLM: Time Series Forecasting by Reprogramming Large Language Models](https://zhuanlan.zhihu.com/p/676256783) | None |
| 23-10-08 | [TEMPO](https://arxiv.org/abs/2310.04948) | ICLR 2024 | TEMPO: Prompt-based Generative Pre-trained Transformer for Time Series Forecasting | [TEMPO](https://github.com/dc-research/tempo) |
| 23-10-12 | [Lag-Llama](https://arxiv.org/abs/2310.08278) | Arxiv 2023 | Lag-Llama: Towards Foundation Models for Time Series Forecasting | [Lag-Llama](https://github.com/time-series-foundation-models/lag-llama) |
| 23-10-15 | [UniTime](https://arxiv.org/abs/2310.09751) | WWW 2024 | UniTime: A Language-Empowered Unified Model for Cross-Domain Time Series Forecasting | [UniTime](https://github.com/liuxu77/UniTime) |
| 23-11-03 | [ForecastPFN](https://arxiv.org/abs/2311.01933) | NIPS 2023 | ForecastPFN: Synthetically-Trained Zero-Shot Forecasting | [ForecastPFN](https://github.com/abacusai/forecastpfn) |
| 23-11-24 | [FPT++](https://arxiv.org/abs/2311.14782) 🌟 | Arxiv 2023 | One Fits All: Universal Time Series Analysis by Pretrained LM and Specially Designed Adaptors | [GPT4TS_Adapter](https://github.com/PSacfc/GPT4TS_Adapter) |
| 24-01-18 | [ST-LLM](https://arxiv.org/abs/2401.10134) | Arxiv 2024 | Spatial-Temporal Large Language Model for Traffic Prediction | None |
| 24-02-01 | [LLMICL](https://arxiv.org/abs/2402.00795) | Arxiv 2024 | LLMs learn governing principles of dynamical systems, revealing an in-context neural scaling law | [LLMICL](https://github.com/AntonioLiu97/llmICL) |
| 24-02-04 | [AutoTimes](https://arxiv.org/abs/2402.02370) | Arxiv 2024 | AutoTimes: Autoregressive Time Series Forecasters via Large Language Models | [AutoTimes](https://github.com/thuml/AutoTimes) |
| 24-02-05 | [Position](https://openreview.net/attachment?id=iroZNDxFJZ&name=pdf) | ICML 2024 | Position: What Can Large Language Models Tell Us about Time Series Analysis | None |
| 24-02-07 | [aLLM4TS](https://arxiv.org/abs/2402.04852) | ICML 2024 | Multi-Patch Prediction: Adapting LLMs for Time Series Representation Learning | None |
| 24-02-16 | [TSFwithLLM](https://arxiv.org/abs/2402.10835) | Arxiv 2024 | Time Series Forecasting with LLMs: Understanding and Enhancing Model Capabilities | None |
| 24-02-25 | [LSTPrompt](https://arxiv.org/abs/2402.16132) | Arxiv 2024 | LSTPrompt: Large Language Models as Zero-Shot Time Series Forecasters by Long-Short-Term Prompting | [LSTPrompt](https://github.com/AdityaLab/lstprompt) |
| 24-03-09 | [S2IP-LLM](https://arxiv.org/abs/2403.05798) | ICML 2024 | S2IP-LLM: Semantic Space Informed Prompt Learning with LLM for Time Series Forecasting | None |
| 24-03-12 | [CALF](https://arxiv.org/abs/2403.07300) | Arxiv 2024 | CALF: Aligning LLMs for Time Series Forecasting via Cross-modal Fine-Tuning | [CALF](https://github.com/Hank0626/CALF) |
| 24-03-12 | [Chronos](https://arxiv.org/abs/2403.07815) 🌟 | Arxiv 2024 | Chronos: Learning the Language of Time Series | [Chronos](https://github.com/amazon-science/chronos-forecasting) |
| 24-04-17 | [TSandLanguage](https://arxiv.org/abs/2404.11757) | Arxiv 2024 | Language Models Still Struggle to Zero-shot Reason about Time Series | [TSandLanguage](https://github.com/behavioral-data/TSandLanguage) |
| 24-05-22 | [TGForecaster](https://arxiv.org/abs/2405.13522) | Arxiv 2024 | Beyond Trend and Periodicity: Guiding Time Series Forecasting with Textual Cues | [tgtsf](https://github.com/vewoxic/tgtsf) |
| 24-05-23 | [Time-FFM](https://arxiv.org/abs/2405.14252) | NIPS 2024 | Time-FFM: Towards LM-Empowered Federated Foundation Model for Time Series Forecasting | None |
| 24-06-03 | [TimeCMA](https://arxiv.org/abs/2406.01638) | Arxiv 2024 | TimeCMA: Towards LLM-Empowered Time Series Forecasting via Cross-Modality Alignment | None |
| 24-06-07 | [LMGF](https://arxiv.org/abs/2406.05249) | Arxiv 2024 | A Language Model-Guided Framework for Mining Time Series with Distributional Shifts | None |
| 24-06-12 | [Time-MMD](https://arxiv.org/abs/2406.08627) | Arxiv 2024 | Time-MMD: A New Multi-Domain Multimodal Dataset for Time Series Analysis | [Time-MMD](https://github.com/AdityaLab/Time-MMD) |
| 24-06-20 | [LTSM-bundle](https://arxiv.org/abs/2406.14045) | Arxiv 2024 | Understanding Different Design Choices in Training Large Time Series Models | [ltsm](https://github.com/daochenzha/ltsm) |
| 24-06-22 | [AreLLMUseful](https://arxiv.org/abs/2406.16964)🌟 | NIPS 2024 | Are Language Models Actually Useful for Time Series Forecasting? | [ts_models](https://github.com/bennytmt/ts_models) |
| 24-07-30 | [FedTime](https://arxiv.org/abs/2407.20503) | Arxiv 2024 | A federated large language model for long-term time series forecasting | None |
| 24-08-22 | [LLMGeovec](https://arxiv.org/abs/2408.12116) | Arxiv 2024 | Geolocation Representation from Large Language Models are Generic Enhancers for Spatio-Temporal Learning | None |
| 24-08-24 | [RePST](https://arxiv.org/abs/2408.14505) | Arxiv 2024 | Empowering Pre-Trained Language Models for Spatio-Temporal Forecasting via Decoupling Enhanced Discrete Reprogramming | None |
| 24-09-17 | [MLLM](https://arxiv.org/abs/2409.11376) | Arxiv 2024 | Towards Time Series Reasoning with LLMs | None |
| 24-10-04 | [MetaTST](https://arxiv.org/abs/2410.03806) | Arxiv 2024 | Metadata Matters for Time Series: Informative Forecasting with Transformers | None |
| 24-10-15 | [LLM-Mixer](https://arxiv.org/abs/2410.11674) | Arxiv 2024 | LLM-Mixer: Multiscale Mixing in LLMs for Time Series Forecasting | [LLMMixer](https://github.com/Kowsher/LLMMixer) |
| 24-10-16 | [NoLLM](https://arxiv.org/abs/2410.12326) | Arxiv 2024 | Revisited Large Language Model for Time Series Analysis through Modality Alignment | None |
| 24-10-18 | [XForecast](https://arxiv.org/abs/2410.14180) | Arxiv 2024 | XForecast: Evaluating Natural Language Explanations for Time Series Forecasting | None |
| 24-10-21 | [LLM-TS](https://arxiv.org/abs/2410.16489) | Arxiv 2024 | LLM-TS Integrator: Integrating LLM for Enhanced Time Series Modeling | None |
| 24-10-28 | [Strada-LLM](https://arxiv.org/abs/2410.20856) | Arxiv 2024 | Strada-LLM: Graph LLM for traffic prediction | None |
| 24-10-29 | [Fourier-Head](https://arxiv.org/abs/2410.22269) | Arxiv 2024 | Fourier Head: Helping Large Language Models Learn Complex Probability Distributions | [Fourier-Head](https://github.com/nate-gillman/fourier-head) |
| 24-11-24 | [LeMoLE](https://arxiv.org/abs/2412.00053) | Arxiv 2024 | LeMoLE: LLM-Enhanced Mixture of Linear Experts for Time Series Forecasting | None |
| 24-12-03 | [LLMForecaster](https://arxiv.org/abs/2412.02525) | Arxiv 2024 | LLMForecaster: Improving Seasonal Event Forecasts with Unstructured Textual Data | None |
| 24-12-06 | [NNCL-TLLM](https://arxiv.org/abs/2412.04806) | Arxiv 2024 | Rethinking Time Series Forecasting with LLMs via Nearest Neighbor Contrastive Learning | None |
| 24-12-16 | [Apollo-Forecast](https://arxiv.org/abs/2412.12226) | AAAI 2025 | Apollo-Forecast: Overcoming Aliasing and Inference Speed Challenges in Language Models for Time Series Forecasting | None |
| 24-12-21 | [TimeRAG](https://arxiv.org/abs/2412.16643) | ICASSP 2025 | TimeRAG: Boosting LLM Time Series Forecasting via Retrieval-Augmented Generation | None |
| 25-01-07 | [DECA](https://arxiv.org/abs/2501.03747) | ICLR 2025 | Context-Alignment: Activating and Enhancing LLM Capabilities in Time Series | None |
| 25-01-10 | [MPT](https://arxiv.org/abs/2501.06386) | Arxiv 2025 | Using Pre-trained LLMs for Multivariate Time Series Forecasting | None |
| 25-02-04 | [LASTS](https://arxiv.org/abs/2502.01922) | Arxiv 2025 | LAST SToP For Modeling Asynchronous Time Series | None |
| 25-02-09 | [FinSeer](https://arxiv.org/abs/2502.05878) | Arxiv 2025 | Retrieval-augmented Large Language Models for Financial Time Series Forecasting | None |
| 25-03-05 | [SMETimes](https://arxiv.org/abs/2503.03594) | Arxiv 2025 | Small but Mighty: Enhancing Time Series Forecasting with Lightweight LLMs | [SMETimes](https://github.com/xiyan1234567/SMETimes) |
| 25-03-10 | [LTM](https://arxiv.org/abs/2503.07682) | Arxiv 2025 | A Time Series Multitask Framework Integrating a Large Language Model, Pre-Trained Time Series Model, and Knowledge Graph | None |
| 25-03-11 | [LangTime](https://arxiv.org/abs/2503.08271) | Arxiv 2025 | LangTime: A Language-Guided Unified Model for Time Series Forecasting with Proximal Policy Optimization | [LangTime](https://github.com/niuwz/LangTime) |
| 25-03-12 | [LLM-PS](https://arxiv.org/abs/2503.09656) | Arxiv 2025 | LLM-PS: Empowering Large Language Models for Time Series Forecasting with Temporal Patterns and Semantics | None |
| 25-04-02 | [ModelSelection](https://arxiv.org/abs/2504.02119) | Arxiv 2025 | Efficient Model Selection for Time Series Forecasting via LLMs | None |
| 25-04-10 | [MLTA](https://arxiv.org/abs/2504.07360) | DASFAA 2025 | Enhancing Time Series Forecasting via Multi-Level Text Alignment with LLMs | [MLTA](https://github.com/ztb-35/MLTA) |
| 25-05-04 | [TimeKD](https://arxiv.org/abs/2505.02138) | ICDE 2025 | Efficient Multivariate Time Series Forecasting via Calibrated Language Models with Privileged Knowledge Distillation | [timekd](https://github.com/chenxiliu-hnu/timekd) |
| 25-05-16 | [Logo-LLM](https://arxiv.org/abs/2505.11017) | Arxiv 2025 | Logo-LLM: Local and Global Modeling with Large Language Models for Time Series Forecasting | None |
| 25-05-19 | [SGCMA](https://arxiv.org/abs/2505.13175) | Arxiv 2025 | Enhancing LLMs for Time Series Forecasting via Structure-Guided Cross-Modal Alignment | None |


</details>


<details><summary><h2 style="display: inline;">Multi-Agent.</h2></summary>

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 24-08-18 | [Agentic-RAG](https://arxiv.org/abs/2408.14484) | KDDW 2024 | Agentic Retrieval-Augmented Generation for Time Series Analysis | None |
| 25-04-14 | [IA_news_model](https://arxiv.org/abs/2504.10210) | Arxiv 2025 | Can Competition Enhance the Proficiency of Agents Powered by Large Language Models in the Realm of News-driven Time Series Forecasting? | [IA_news_model](https://anonymous.4open.science/r/IA_news_model-D7D6) |


</details>


<details><summary><h2 style="display: inline;">Representation Learning.</h2></summary>

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 20-10-06 | [TST](https://arxiv.org/abs/2010.02803) | KDD 2021 | A Transformer-based Framework for Multivariate Time Series Representation Learning | [mvts_transformer](https://github.com/gzerveas/mvts_transformer)   |
| 21-06-26 | [TS-TCC](https://arxiv.org/abs/2106.14112) | IJCAI 2021 | Time-Series Representation Learning via Temporal and Contextual Contrasting | [TS-TCC](https://github.com/emadeldeen24/TS-TCC) |
| 21-09-29 | [CoST](https://openreview.net/forum?id=PilZY3omXV2) | ICLR 2022 | CoST: Contrastive Learning of Disentangled Seasonal-Trend Representations for Time Series Forecasting | [CoST](https://github.com/salesforce/CoST)   |
| 22-05-16 | [LaST](https://openreview.net/pdf?id=C9yUwd72yy) | NIPS 2022 | LaST: Learning Latent Seasonal-Trend Representations for Time Series Forecasting | [LaST](https://github.com/zhycs/LaST)   |
| 22-06-18 | [STEP](https://arxiv.org/abs/2206.09113) | KDD 2022 | Pre-training Enhanced Spatial-temporal Graph Neural Network for Multivariate Time Series Forecasting | [STEP](https://github.com/zezhishao/step)   |
| 22-06-28 | [TS2Vec](https://ojs.aaai.org/index.php/AAAI/article/view/20881) | AAAI 2022 | TS2Vec: Towards Universal Representation of Time Series | [ts2vec](https://github.com/zhihanyue/ts2vec) |
| 23-02-02 | [SimMTM](https://arxiv.org/abs/2302.00861) | NIPS 2023 | SimMTM: A Simple Pre-Training Framework for Masked Time-Series Modeling | [SimMTM](https://github.com/thuml/simmtm) |
| 23-02-07 | [DBPM](https://arxiv.org/abs/2302.03357) | ICLR 2024 | Towards Enhancing Time Series Contrastive Learning: A Dynamic Bad Pair Mining Approach | None |
| 23-03-01 | [TimeMAE](https://arxiv.org/abs/2303.00320) | Arxiv 2023 | TimeMAE: Self-Supervised Representations of Time Series with Decoupled Masked Autoencoders | [TimeMAE](https://github.com/Mingyue-Cheng/TimeMAE) |
| 23-08-02 | [Floss](https://arxiv.org/abs/2308.01011) | Arxiv 2023 | Enhancing Representation Learning for Periodic Time Series with Floss: A Frequency Domain Regularization Approach | [floss](https://github.com/agustdd/floss) |
| 23-12-01 | [STD_MAE](https://arxiv.org/abs/2312.00516) | Arxiv 2023 | Spatio-Temporal-Decoupled Masked Pre-training for Traffic Forecasting | [STD_MAE](https://github.com/jimmy-7664/std_mae) |
| 23-12-25 | [TimesURL](https://arxiv.org/abs/2312.15709) | AAAI 2024 | TimesURL: Self-supervised Contrastive Learning for Universal Time Series Representation Learning | None |
| 24-01-16 | [SoftCLT](https://openreview.net/forum?id=pAsQSWlDUf) | ICLR 2024 | Soft Contrastive Learning for Time Series | None |
| 24-01-16 | [PITS](https://openreview.net/forum?id=WS7GuBDFa2) | ICLR 2024 | Learning to Embed Time Series Patches Independently | [PITS](https://github.com/seunghan96/pits) |
| 24-01-16 | [T-Rep](https://openreview.net/forum?id=3y2TfP966N) | ICLR 2024 | T-Rep: Representation Learning for Time Series using Time-Embeddings | None |
| 24-01-16 | [AutoTCL](https://openreview.net/forum?id=EIPLdFy3vp) | ICLR 2024 | Parametric Augmentation for Time Series Contrastive Learning | None |
| 24-01-16 | [AutoCon](https://openreview.net/forum?id=nBCuRzjqK7) | ICLR 2024 | Self-Supervised Contrastive Learning for Long-term Forecasting | [AutoCon](https://github.com/junwoopark92/self-supervised-contrastive-forecsating) |
| 24-02-04 | [TimeSiam](https://arxiv.org/abs/2402.02475) | ICML 2024 | TimeSiam: A Pre-Training Framework for Siamese Time-Series Modeling | None |
| 24-03-19 | [CrossTimeNet](https://arxiv.org/abs/2403.12372) | WSDM 2025 | Cross-Domain Pre-training with Language Models for Transferable Time Series Representations | [crosstimenet](https://github.com/mingyue-cheng/crosstimenet) |
| 24-03-31 | [SimTS](https://arxiv.org/abs/2303.18205) | ICASSP 2024 | Rethinking Contrastive Representation Learning for Time Series Forecasting | [simTS](https://github.com/xingyu617/SimTS_Representation_Learning) |


</details>


<details><summary><h2 style="display: inline;">Foundation Model.</h2></summary>

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 23-10-14 | [TimesFM](https://arxiv.org/abs/2310.10688) | ICML 2024 | A decoder-only foundation model for time-series forecasting | [timesfm](https://github.com/google-research/timesfm) |
| 24-01-08 | [TTMs](https://arxiv.org/abs/2401.03955) | Arxiv 2024 | Tiny Time Mixers (TTMs): Fast Pre-trained Models for Enhanced Zero/Few-Shot Forecasting of Multivariate Time Series | [TTMs](https://huggingface.co/ibm-granite/granite-timeseries-ttm-v1) |
| 24-02-04 | [Timer](https://arxiv.org/abs/2402.02368) | ICML 2024 | [Timer: Generative Pre-trained Transformers Are Large Time Series Models](https://zhuanlan.zhihu.com/p/698842899) | [Timer](https://github.com/thuml/Large-Time-Series-Model) |
| 24-02-04 | [Moirai](https://arxiv.org/abs/2402.02592) | ICML 2024 | [Unified Training of Universal Time Series Forecasting Transformers](https://zhuanlan.zhihu.com/p/698842899) | [Moirai](https://github.com/SalesforceAIResearch/uni2ts) |
| 24-02-06 | [MOMENT](https://arxiv.org/abs/2402.03885) | ICML 2024 | [MOMENT: A Family of Open Time-series Foundation Models](https://zhuanlan.zhihu.com/p/698842899) | [MOMENT](https://anonymous.4open.science/r/BETT-773F/README.md) |
| 24-02-14 | [GTT](https://dl.acm.org/doi/10.1145/3627673.3679931) | CIKM 2024 | General Time Transformer: an Encoder-only Foundation Model for Zero-Shot Multivariate Time Series Forecasting | [GTT](https://github.com/cfeng783/gtt) |
| 24-02-19 | [UniST](https://arxiv.org/abs/2402.11838) | KDD 2024 | UniST: A Prompt-Empowered Universal Model for Urban Spatio-Temporal Prediction | [UniST](https://github.com/tsinghua-fib-lab/UniST) |
| 24-02-26 | [TOTEM](https://arxiv.org/abs/2402.16412) | TMLR 2024 | TOTEM: TOkenized Time Series EMbeddings for General Time Series Analysis | [TOTEM](https://github.com/SaberaTalukder/TOTEM) |
| 24-02-26 | [GPHT](https://arxiv.org/abs/2402.16516) | KDD 2024 | Generative Pretrained Hierarchical Transformer for Time Series Forecasting | [GPHT](https://github.com/icantnamemyself/GPHT) |
| 24-02-29 | [UniTS](https://arxiv.org/abs/2403.00131) | NIPS 2024 | UniTS: Building a Unified Time Series Model | [UniTS](https://github.com/mims-harvard/UniTS) |


</details>


<details><summary><h2 style="display: inline;">Pretrain & Representation.</h2></summary>

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 24-05-02 | [UP2ME](https://openreview.net/forum?id=aR3uxWlZhX) | ICML 2024 | UP2ME: Univariate Pre-training to Multivariate Fine-tuning as a General-purpose Framework for Multivariate Time Series Analysis | [UP2ME](https://github.com/Thinklab-SJTU/UP2ME) |
| 24-05-17 | [UniCL](https://arxiv.org/abs/2405.10597) | Arxiv 2024 | UniCL: A Universal Contrastive Learning Framework for Large Time Series Models | [UniTS](https://github.com/mims-harvard/UniTS) |
| 24-05-24 | [NuwaTS](https://arxiv.org/abs/2405.15317) | Arxiv 2024 | NuwaTS: a Foundation Model Mending Every Incomplete Time Series | [NuwaTS](https://github.com/Chengyui/NuwaTS) |
| 24-05-24 | [ROSE](https://arxiv.org/abs/2405.17478) | Arxiv 2024 | ROSE: Register Assisted General Time Series Forecasting with Decomposed Frequency Learning | None |
| 24-05-28 | [TSRM](https://arxiv.org/abs/2405.18165) | Arxiv 2024 | Time Series Representation Models | [TSRM](https://github.com/RobertLeppich/TSRM) |
| 24-07-10 | [ViTime](https://arxiv.org/abs/2407.07311) | Arxiv 2024 | ViTime: A Visual Intelligence-Based Foundation Model for Time Series Forecasting | [ViTime](https://github.com/IkeYang/ViTime) |
| 24-07-10 | [Toto](https://arxiv.org/abs/2407.07874) | Arxiv 2024 | Toto: Time Series Optimized Transformer for Observability | None |
| 24-08-16 | [OpenCity](https://arxiv.org/abs/2408.10269) | Arxiv 2024 | OpenCity: Open Spatio-Temporal Foundation Models for Traffic Prediction | [OpenCity](https://github.com/hkuds/opencity) |
| 24-08-24 | [TSDE](https://dl.acm.org/doi/abs/10.1145/3637528.3671673) | KDD 2024 | Self-Supervised Learning of Time Series Representation via Diffusion Process and Imputation-Interpolation-Forecasting Mask | [TSDE](https://github.com/llcresearch/TSDE) |
| 24-08-30 | [VisionTS](https://arxiv.org/abs/2408.17253) 🌟 | Arxiv 2024 | VisionTS: Visual Masked Autoencoders Are Free-Lunch Zero-Shot Time Series Forecasters | [VisionTS](https://github.com/Keytoyze/VisionTS) |
| 24-09-17 | [ImplicitReason](https://arxiv.org/abs/2409.10840) | Arxiv 2024 | Implicit Reasoning in Deep Time Series Forecasting | None |
| 24-09-24 | [Time-MoE](https://arxiv.org/abs/2409.16040) | ICLR 2025 | [Time-MoE: Billion-Scale Time Series Foundation Models with Mixture of Experts](https://zhuanlan.zhihu.com/p/2950313586) | [Time-MoE](https://github.com/Time-MoE/Time-MoE) |
| 24-10-07 | [Timer-XL](https://arxiv.org/abs/2410.04803) | ICLR 2025 | Timer-XL: Long-Context Transformers for Unified Time Series Forecasting | [Timer-XL](https://github.com/thuml/Timer-XL) |
| 24-10-09 | [OTiS](https://arxiv.org/abs/2410.07299) | Arxiv 2024 | Towards Generalisable Time Series Understanding Across Domains | [otis](https://github.com/oetu/otis) |
| 24-10-14 | [Moirai-MoE](https://arxiv.org/abs/2410.10469) | Arxiv 2024 | Moirai-MoE: Empowering Time Series Foundation Models with Sparse Mixture of Experts | None |
| 24-10-30 | [FlexTSF](https://arxiv.org/abs/2410.23160) | Arxiv 2024 | FlexTSF: A Universal Forecasting Model for Time Series with Variable Regularities | [FlexTSF](https://github.com/jingge326/flextsf) |
| 24-10-31 | [ICF](https://arxiv.org/abs/2410.24087) | Arxiv 2024 | In-Context Fine-Tuning for Time-Series Foundation Models | None |
| 24-11-05 | [TSMamba](https://arxiv.org/abs/2411.02941) | Arxiv 2024 | A Mamba Foundation Model for Time Series Forecasting | None |
| 24-11-24 | [TableTime](https://arxiv.org/abs/2411.15737) | Arxiv 2024 | TableTime: Reformulating Time Series Classification as Zero-Shot Table Understanding via Large Language Models | [TableTime](https://github.com/realwangjiahao/TableTime) |
| 24-12-01 | [WQ4TS](https://arxiv.org/abs/2412.00772) | Arxiv 2024 | A Wave is Worth 100 Words: Investigating Cross-Domain Transferability in Time Series | None |
| 25-01-27 | [TimeHF](https://arxiv.org/abs/2501.15942) | Arxiv 2025 | TimeHF: Billion-Scale Time Series Models Guided by Human Feedback | None |
| 25-02-02 | [Sundial](https://arxiv.org/abs/2502.00816) | Arxiv 2025 | Sundial: A Family of Highly Capable Time Series Foundation Models | None |
| 25-02-05 | [TopoCL](https://arxiv.org/abs/2502.02924) | Arxiv 2025 | TopoCL: Topological Contrastive Learning for Time Series | None |
| 25-02-05 | [GTM](https://arxiv.org/abs/2502.03264) | Arxiv 2025 | General Time-series Model for Universal Knowledge Representation of Multivariate Time-Series data | None |
| 25-02-05 | [CAPE](https://arxiv.org/abs/2502.03393) | Arxiv 2025 | CAPE: Covariate-Adjusted Pre-Training for Epidemic Time Series Forecasting | None |
| 25-02-09 | [Reasoning](https://arxiv.org/abs/2502.06037) | Arxiv 2025 | Investigating Compositional Reasoning in Time Series Foundation Models | [tsfm_reasoning](https://github.com/PotosnakW/neuralforecast/tree/tsfm_reasoning) |
| 25-02-14 | [AdaPTS](https://arxiv.org/abs/2502.10235) | Arxiv 2025 | AdaPTS: Adapting Univariate Foundation Models to Probabilistic Multivariate Time Series Forecasting | [AdaPTS](https://github.com/abenechehab/AdaPTS) |
| 25-02-17 | [Robustness](https://arxiv.org/abs/2502.12226) | Arxiv 2025 | On Creating a Causally Grounded Usable Rating Method for Assessing the Robustness of Foundation Models Supporting Time Series | None |
| 25-02-22 | [TimePFN](https://arxiv.org/abs/2502.16294) | AAAI 2025 | TimePFN: Effective Multivariate Time Series Forecasting with Synthetic Data | [TimePFN](https://github.com/egetaga/timepfn) |
| 25-02-28 | [TimesBERT](https://arxiv.org/abs/2502.21245) | Arxiv 2025 | TimesBERT: A BERT-Style Foundation Model for Time Series Understanding | None |
| 25-03-04 | [SeqFusion](https://arxiv.org/abs/2503.02836) | Arxiv 2025 | SeqFusion: Sequential Fusion of Pre-Trained Models for Zero-Shot Time-Series Forecasting | [SeqFusion](https://github.com/Tingji2419/SeqFusion) |
| 25-03-06 | [TimeFound](https://arxiv.org/abs/2503.04118) | Arxiv 2025 | TimeFound: A Foundation Model for Time Series Forecasting | None |
| 25-03-06 | [TS-RAG](https://arxiv.org/abs/2503.07649) | Arxiv 2025 | TS-RAG: Retrieval-Augmented Generation based Time Series Foundation Models are Stronger Zero-Shot Forecaster | None |
| 25-03-15 | [ChronosX](https://arxiv.org/abs/2503.12107) | AISTATS 2025 | ChronosX: Adapting Pretrained Time Series Models with Exogenous Variables | [chronosx](https://github.com/amazon-science/chronos-forecasting/tree/chronosx) |
| 25-03-21 | [TRACE](https://arxiv.org/abs/2503.16991) | Arxiv 2025 | TRACE: Time SeRies PArameter EffiCient FinE-tuning | None |
| 25-05-21 | [Time Tracker](https://arxiv.org/abs/2505.15151) | Arxiv 2025 | Time Tracker: Mixture-of-Experts-Enhanced Foundation Time Series Forecasting Model with Decoupled Training Pipelines | None |


</details>

<details><summary><h2 style="display: inline;">Multimodal.</h2></summary>

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 24-06-12 | [Time-MMD](https://arxiv.org/abs/2406.08627) | Arxiv 2024 | Time-MMD: A New Multi-Domain Multimodal Dataset for Time Series Analysis | [time-mmd](https://github.com/adityalab/time-mmd) |
| 24-10-16 | [ContextFormer](https://arxiv.org/abs/2410.12672) | Arxiv 2024 | Context Matters: Leveraging Contextual Features for Time Series Forecasting | None |
| 24-11-11 | [Hybrid-MMF](https://arxiv.org/abs/2411.06735) | Arxiv 2024 | Multi-Modal Forecaster: Jointly Predicting Time Series and Textual Data | [Multimodal_Forecasting](https://github.com/Rose-STL-Lab/Multimodal_Forecasting) |
| 24-12-16 | [ChatTime](https://arxiv.org/abs/2412.11376) | AAAI 2025 | ChatTime: A Unified Multimodal Time Series Foundation Model Bridging Numerical and Textual Data | [ChatTime](https://github.com/forestsking/chattime) |
| 25-01-13 | [TextFusionHTS](https://arxiv.org/abs/2501.07048) | NIPSW 2024 | Unveiling the Potential of Text in High-Dimensional Time Series Forecasting | [TextFusionHTS](https://github.com/xinzzzhou/textfusionhts) |
| 25-02-06 | [Time-VLM](https://arxiv.org/abs/2502.04395) | Arxiv 2025 | Time-VLM: Exploring Multimodal Vision-Language Models for Augmented Time Series Forecasting | None |
| 25-02-13 | [TaTS](https://arxiv.org/abs/2502.08942) | Arxiv 2025 | Language in the Flow of Time: Time-Series-Paired Texts Weaved into a Unified Temporal Narrative | None |
| 25-03-02 | [TimeXL](https://arxiv.org/abs/2503.01013) | Arxiv 2025 | Explainable Multi-modal Time Series Prediction with LLM-in-the-Loop | None |
| 25-03-21 | [MTBench](https://arxiv.org/abs/2503.16858) | Arxiv 2025 | MTBench: A Multimodal Time Series Benchmark for Temporal Reasoning and Question Answering | [MTBench](https://github.com/Graph-and-Geometric-Learning/MTBench) |
| 25-04-28 | [MCD-TSF](https://arxiv.org/abs/2504.19669) | Arxiv 2025 | Multimodal Conditioned Diffusive Time Series Forecasting | [MCD-TSF](https://github.com/synlp/MCD-TSF) |
| 25-05-02 | [Dual-Forecaster](https://arxiv.org/abs/2505.01135) | Arxiv 2025 | Dual-Forecaster: A Multimodal Time Series Model Integrating Descriptive and Predictive Texts | None |
| 25-05-15 | [ChronoSteer](https://arxiv.org/abs/2505.10083) | Arxiv 2025 | ChronoSteer: Bridging Large Language Model and Time Series Foundation Model via Synthetic Data | None |
| 25-05-16 | [CAPTime](https://arxiv.org/abs/2505.10774) | Arxiv 2025 | Context-Aware Probabilistic Modeling with LLM for Multimodal Time Series Forecasting | None |
| 25-05-20 | [CHARM](https://arxiv.org/abs/2505.14543) | Arxiv 2025 | Time to Embed: Unlocking Foundation Models for Time Series with Channel Descriptions | None |
| 25-05-21 | [MoTime](https://arxiv.org/abs/2505.15072) | Arxiv 2025 | MoTime: A Dataset Suite for Multimodal Time Series Forecasting | None |


</details>

<details><summary><h2 style="display: inline;">Domain Adaptation.</h2></summary>

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 21-02-13 | [DAF](https://arxiv.org/abs/2102.06828) | ICML 2022 | Domain Adaptation for Time Series Forecasting via Attention Sharing | [DAF](https://github.com/leejoonhun/daf) |
| 21-06-13 | [FOIL](https://arxiv.org/abs/2406.09130) | ICML 2024 | Time-Series Forecasting for Out-of-Distribution Generalization Using Invariant Learning | [FOIL](https://github.com/adityalab/foil) |
| 24-08-24 | [STONE](https://dl.acm.org/doi/abs/10.1145/3637528.3671680) | KDD 2024 | STONE: A Spatio-temporal OOD Learning Framework Kills Both Spatial and Temporal Shifts | [STONE](https://github.com/PoorOtterBob/STONE-KDD-2024) |
| 24-12-15 | [LTG](https://arxiv.org/abs/2412.11171) | Arxiv 2024 | Learning Latent Spaces for Domain Generalization in Time Series Forecasting | None |
| 25-02-23 | [LCA](https://arxiv.org/abs/2502.16637) | Arxiv 2025 | Time Series Domain Adaptation via Latent Invariant Causal Mechanism | [LCA](https://github.com/DMIRLAB-Group/LCA) |


</details>

<details><summary><h2 style="display: inline;">Online.</h2></summary>

Date|Method|Type|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----|-----
| 22-02-23 | [FSNet](https://openreview.net/pdf?id=q-PbpHD3EOk) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2023 | Learning Fast and Slow for Online Time Series Forecasting | [FSNet](https://github.com/salesforce/fsnet)   |
| 23-09-22 | [OneNet](https://arxiv.org/abs/2309.12659) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | NIPS 2023 | OneNet: Enhancing Time Series Forecasting Models under Concept Drift by Online Ensembling | [OneNet](https://github.com/yfzhang114/OneNet)   |
| 23-09-25 | [MemDA](https://arxiv.org/abs/2309.14216) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | CIKM 2023 | MemDA: Forecasting Urban Time Series with Memory-based Drift Adaptation |  None  |
| 24-01-08 | [ADCSD](https://arxiv.org/abs/2401.04148) | ![spatio-temporal forecasting](https://img.shields.io/badge/-SpatioTemporal-blue) | Arxiv 2024 | Online Test-Time Adaptation of Spatial-Temporal Traffic Flow Forecasting | [ADCSD](https://github.com/Pengxin-Guo/ADCSD)  |
| 24-02-03 | [TSF-HD](https://arxiv.org/abs/2402.01999) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | A Novel Hyperdimensional Computing Framework for Online Time Series Forecasting on the Edge | [TSF-HD](https://github.com/tsfhd2024/tsf-hd)  |
| 24-02-20 | [SKI-CL](https://arxiv.org/abs/2402.12722) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Structural Knowledge Informed Continual Multivariate Time Series Forecasting | None  |
| 24-03-22 | [D3A](https://arxiv.org/abs/2403.14949) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Addressing Concept Shift in Online Time Series Forecasting: Detect-then-Adapt | None  |
| 24-09-29 | [EvoMSN](https://arxiv.org/abs/2409.19718) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | Arxiv 2024 | Evolving Multi-Scale Normalization for Time Series Forecasting under Distribution Shifts | [EvoMSN](https://github.com/qindalin/evomsn)  |
| 24-12-11 | [Proceed](https://arxiv.org/abs/2412.08435) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | KDD 2025 | Proactive Model Adaptation Against Concept Drift for Online Time Series Forecasting | [Proceed](https://github.com/SJTU-DMTai/OnlineTSF)  |
| 25-01-23 | [DSOF](https://openreview.net/forum?id=I0n3EyogMi) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | ICLR 2025 | Fast and Slow Streams for Online Time Series Forecasting Without Information Leakage | None  |
| 25-02-18 | [LSTD](https://arxiv.org/abs/2502.12603) | ![multivariate time series forecasting](https://img.shields.io/badge/-Multivariate-red) | AAAI 2025 | Disentangling Long-Short Term State Under Unknown Interventions for Online Time Series Forecasting | [LSTD](https://github.com/DMIRLAB-Group/LSTD)  |
| 25-02-18 | [AdapTS](https://arxiv.org/abs/2502.12920) | ![univariate time series forecasting](https://img.shields.io/badge/-Univariate-brightgreen) | Arxiv 2025 | Lightweight Online Adaption for Time Series Foundation Model Forecasts | None |

</details>

<details><summary><h2 style="display: inline;">KAN (Kolmogorov–Arnold Network).</h2></summary>

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 24-05-12 | [TKAN](https://arxiv.org/abs/2405.07344) | Arxiv 2024 | TKAN: Temporal Kolmogorov-Arnold Networks | [tkan](https://github.com/remigenet/tkan) |
| 24-05-14 | [KAN](https://arxiv.org/abs/2405.08790) | Arxiv 2024 | Kolmogorov-Arnold Networks (KANs) for Time Series Analysis | None |
| 24-06-04 | [TKAT](https://arxiv.org/abs/2406.02486) | Arxiv 2024 | A Temporal Kolmogorov-Arnold Transformer for Time Series Forecasting | [TKAT](https://github.com/remigenet/TKAT) |
| 24-06-04 | [MT-KAN](https://arxiv.org/abs/2406.02496) | Arxiv 2024 | Kolmogorov-Arnold Networks for Time Series: Bridging Predictive Power and Interpretability | None |
| 24-08-21 | [KAN4TSF](https://arxiv.org/abs/2408.11306) | Arxiv 2024 | KAN4TSF: Are KAN and KAN-based models Effective for Time Series Forecasting? | [kan4tsf](https://github.com/2448845600/kan4tsf) |
| 24-10-13 | [WormKAN](https://arxiv.org/abs/2410.10041) | NIPSW 2024 | Are KAN Effective for Identifying and Tracking Concept Drift in Time Series? | None |
| 24-10-19 | [HiPPO-KAN](https://arxiv.org/abs/2410.14939) | Arxiv 2024 | HiPPO-KAN: Efficient KAN Model for Time Series Analysis | None |
| 24-12-19 | [NBEATS-KAN](https://arxiv.org/abs/2412.17853) | NIPSW 2024 | Zero Shot Time Series Forecasting Using Kolmogorov Arnold Networks | None |
| 25-02-10 | [TimeKAN](https://arxiv.org/abs/2502.06910) | ICLR 2025 | TimeKAN: KAN-based Frequency Decomposition Learning Architecture for Long-term Time Series Forecasting | [TimeKAN](https://github.com/huangst21/TimeKAN) |
| 25-02-25 | [TSKANMixer](https://arxiv.org/abs/2502.18410) | AAAIW 2025 | TSKANMixer: Kolmogorov-Arnold Networks with MLP-Mixer Model for Time Series Forecasting | None |
| 25-04-23 | [iTFKAN](https://arxiv.org/abs/2504.16432) | Arxiv 2025 | iTFKAN: Interpretable Time Series Forecasting with Kolmogorov-Arnold Network | None |


</details>


<details><summary><h2 style="display: inline;">Theory.</h2></summary>

Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
| 22-10-25 | [WaveBound](https://arxiv.org/abs/2210.14303) | NIPS 2022 | WaveBound: Dynamic Error Bounds for Stable Time Series Forecasting | [WaveBound](https://github.com/choyi0521/WaveBound) |
| 23-05-25 | [Ensembling](https://arxiv.org/abs/2305.15786) | ICML 2023 | Theoretical Guarantees of Learning Ensembling Strategies with Applications to Time Series Forecasting | None |
| 24-12-08 | [CurseofAttention](https://arxiv.org/abs/2412.06061) | Arxiv 2024 | Curse of Attention: A Kernel-Based Perspective for Why Transformers Fail to Generalize on Time Series Forecasting and Beyond | None |
| 24-12-27 | [BAPC](https://arxiv.org/abs/2412.19897) | Arxiv 2024 | Surrogate Modeling for Explainable Predictive Time Series Corrections | None |
| 25-02-14 | [WeaKL](https://arxiv.org/abs/2502.10485) | Arxiv 2025 | Forecasting time series with constraints | [WeaKL](https://github.com/NathanDoumeche/WeaKL) |

</details>


<details><summary><h2 style="display: inline;">Other.</h2></summary>

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 16-12-05 | [TRMF](https://proceedings.neurips.cc/paper_files/paper/2016/file/85422afb467e9456013a2a51d4dff702-Paper.pdf) | NIPS 2016 | Temporal Regularized Matrix Factorization for High-dimensional Time Series Prediction | [TRMF](https://github.com/rofuyu/exp-trmf-nips16)   |
| 23-05-23 | [DF2M](https://arxiv.org/abs/2305.14543) | ICML 2024 | Deep Functional Factor Models: Forecasting High-Dimensional Functional Time Series via Bayesian Nonparametric Factorization | None  |
| 24-01-16 | [STanHop-Net](https://openreview.net/forum?id=6iwg437CZs) | ICLR 2024 | STanHop: Sparse Tandem Hopfield Model for Memory-Enhanced Time Series Prediction | None  |
| 24-02-02 | [SNN](https://arxiv.org/abs/2402.01533) | ICML 2024 | Efficient and Effective Time-Series Forecasting with Spiking Neural Networks | None  |
| 24-03-12 | [BayesNF](https://arxiv.org/abs/2403.07657) | Arxiv 2024 | Scalable Spatiotemporal Prediction with Bayesian Neural Fields | [BayesNF](https://github.com/google/bayesnf)  |
| 24-05-16 | [LaT-PFN](https://arxiv.org/abs/2405.10093) | Arxiv 2024 | LaT-PFN: A Joint Embedding Predictive Architecture for In-context Time-series Forecasting | None |
| 24-05-24 | [ScalingLaw](https://arxiv.org/abs/2405.15124) | NIPS 2024 | Scaling Law for Time Series Forecasting | [ScalingLaw](https://github.com/JingzheShi/ScalingLawForTimeSeriesForecasting) |
| 24-06-04 | [CondTSF](https://arxiv.org/abs/2406.02131) | Arxiv 2024 | CondTSF: One-line Plugin of Dataset Condensation for Time Series Forecasting | None |
| 24-06-14 | [MTL](https://arxiv.org/abs/2406.10327) | Arxiv 2024 | Analysing Multi-Task Regression via Random Matrix Theory with Application to Time Series Forecasting | None |
| 24-10-03 | [BackTime](https://arxiv.org/abs/2410.02195) | NIPS 2024 | BACKTIME: Backdoor Attacks on Multivariate Time Series Forecasting | [BackTime](https://github.com/xiaolin-cs/BackTime) |
| 24-10-30 | [SwimRNN](https://arxiv.org/abs/2410.23467) | Arxiv 2024 | Gradient-free training of recurrent neural networks | [swimrnn](https://gitlab.com/felix.dietrich/swimrnn-paper) |
| 24-11-06 | [FACTS](https://arxiv.org/abs/2411.05833) | VLDB 2025 | Fully Automated Correlated Time Series Forecasting in Minutes | [FACTS](https://github.com/ccloud0525/FACTS) |
| 24-12-23 | [EasyTime](https://arxiv.org/abs/2412.17603) | ICDE 2025 | EasyTime: Time Series Forecasting Made Easy | None |
| 24-12-27 | [RS3GP](https://arxiv.org/abs/2412.19727) | Arxiv 2024 | Learning to Forget: Bayesian Time Series Forecasting using Recurrent Sparse Spectrum Signature Gaussian Processes | None |
| 25-01-06 | [TabPFN-TS](https://arxiv.org/abs/2501.02945) | NIPSW 2024 | The Tabular Foundation Model TabPFN Outperforms Specialized Time Series Forecasting Models Based on Simple Features | [TabPFN-TS](https://github.com/liam-sbhoo/tabpfn-time-series) |
| 25-01-23 | [TS-LIF](https://openreview.net/forum?id=rDe9yQQYKt) | ICLR 2025 | TS-LIF: A Temporal Segment Spiking Neuron Network for Time Series Forecasting | [TS-LIF](https://github.com/kkking-kk/TS-LIF) |
| 25-01-23 | [LCESN](https://openreview.net/forum?id=KeRwLLwZaw) | ICLR 2025 | Locally Connected Echo State Networks for Time Series Forecasting | [LCESN](https://github.com/FloopCZ/echo-state-networks) |
| 25-01-23 | [KooNPro](https://openreview.net/forum?id=5oSUgTzs8Y) | ICLR 2025 | KooNPro: A Variance-Aware Koopman Probabilistic Model Enhanced by Neural Processes for Time Series Forecasting | [Koonpro](https://github.com/Rrh-Zheng/Koonpro) |
| 25-01-26 | [TCTNN](https://arxiv.org/abs/2501.15388) | Arxiv 2025 | Guaranteed Multidimensional Time Series Prediction via Deterministic Tensor Completion Theory | [TCTNN](https://github.com/HaoShu2000/TCTNN) |
| 25-01-27 | [METAFORS](https://arxiv.org/abs/2501.16325) | Arxiv 2025 | Tailored Forecasting from Short Time Series via Meta-learning | None |
| 25-03-26 | [PINN](https://arxiv.org/abs/2503.20144) | Arxiv 2025 | Physics-Informed Neural Networks with Unknown Partial Differential Equations: an Application in Multivariate Time Series | None |
| 25-05-09 | [FOCUS](https://arxiv.org/abs/2505.05738) | Arxiv 2025 | Accurate and Efficient Multivariate Time Series Forecasting via Offline Clustering | None |


</details>


## Star History

[![Star History Chart](https://api.star-history.com/svg?repos=ddz16/TSFpaper&type=Date)](https://star-history.com/#ddz16/TSFpaper&Date)
