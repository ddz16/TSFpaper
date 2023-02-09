# TSF Paper
Reading list for research topics in Time Series Forecasting (TSF).

## Survey.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
15-11-23|[Multi-step](https://ieeexplore.ieee.org/abstract/document/7422387)|ACOMP 2015|Comparison of Strategies for Multi-step-Ahead Prediction of Time Series Using Neural Network|None
20-09-27|[DL](https://arxiv.org/abs/2004.13408)|Arxiv 2020|Time Series Forecasting With Deep Learning: A Survey|None

## Transformer.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
19-06-29|[LogTrans](https://arxiv.org/abs/1907.00235)|NIPS 2019|Enhancing the Locality and Breaking the Memory Bottleneck of Transformer on Time Series Forecasting|[flowforecast](https://github.com/AIStream-Peelout/flow-forecast/blob/master/flood_forecast/transformer_xl/transformer_bottleneck.py)
20-06-05|[AST](https://proceedings.neurips.cc/paper/2020/file/c6b8c8d762da15fa8dbbdfb6baf9e260-Paper.pdf)|NIPS 2020|Adversarial Sparse Transformer for Time Series Forecasting|[AST](https://github.com/hihihihiwsf/AST)
20-12-14|[Informer](https://arxiv.org/abs/2012.07436)|AAAI 2021|[Informer: Beyond Efficient Transformer for Long Sequence Time-Series Forecasting](https://zhuanlan.zhihu.com/p/467523291)|[Informer](https://github.com/zhouhaoyi/Informer2020)
21-06-24|[Autoformer](https://arxiv.org/abs/2106.13008)|NIPS 2021|[Autoformer: Decomposition Transformers with Auto-Correlation for Long-Term Series Forecasting](https://zhuanlan.zhihu.com/p/385066440)|[Autoformer](https://github.com/thuml/Autoformer)
21-10-05|[Pyraformer](https://openreview.net/pdf?id=0EXmFzUn5I)|ICLR 2022|[Pyraformer: Low-complexity Pyramidal Attention for Long-range Time Series Modeling and Forecasting](https://zhuanlan.zhihu.com/p/467765457)|[Pyraformer](https://github.com/alipay/Pyraformer)
22-01-30|[FEDformer](https://arxiv.org/abs/2201.12740)|ICML 2022|[FEDformer: Frequency Enhanced Decomposed Transformer for Long-term Series Forecasting](https://zhuanlan.zhihu.com/p/528131016)|[FEDformer](https://github.com/MAZiqing/FEDformer)
22-02-23|[Preformer](https://arxiv.org/abs/2202.11356)|Arxiv 2022|[Preformer: Predictive Transformer with Multi-Scale Segment-wise Correlations for Long-Term Time Series Forecasting](https://zhuanlan.zhihu.com/p/536398013)|[Preformer](https://github.com/ddz16/Preformer)
22-04-28|[Triformer](https://arxiv.org/abs/2204.13767)|IJCAI 2022|[Triformer: Triangular, Variable-Specific Attentions for Long Sequence Multivariate Time Series Forecasting](https://blog.csdn.net/zj_18706809267/article/details/125048492)| [Triformer](https://github.com/razvanc92/triformer)
22-05-16|[MANF](https://arxiv.org/abs/2205.07493)|Arxiv 2022|Multi-scale Attention Flow for Probabilistic Time Series Forecasting|None
22-05-24|[FreDo](https://arxiv.org/abs/2205.12301)|Arxiv 2022|FreDo: Frequency Domain-based Long-Term Time Series Forecasting|None
22-05-27|[TDformer](https://arxiv.org/abs/2212.08151)|NIPSW 2022|[First De-Trend then Attend: Rethinking Attention for Time-Series Forecasting](https://zhuanlan.zhihu.com/p/596022160)|[TDformer](https://github.com/BeBeYourLove/TDformer)
22-05-28|[Non-stationary Transformer](https://arxiv.org/abs/2205.14415)|NIPS 2022|[Non-stationary Transformers: Rethinking the Stationarity in Time Series Forecasting](https://zhuanlan.zhihu.com/p/535931701)|[Non-stationary Transformers](https://github.com/thuml/Nonstationary_Transformers)
22-06-08|[Scaleformer](https://arxiv.org/abs/2206.04038)|ICLR 2023|[Scaleformer: Iterative Multi-scale Refining Transformers for Time Series Forecasting](https://zhuanlan.zhihu.com/p/535556231)|[Scaleformer](https://github.com/BorealisAI/scaleformer)
22-08-30|[Persistence Initialization](https://arxiv.org/abs/2208.14236)|Arxiv 2022|[Persistence Initialization: A novel adaptation of the Transformer architecture for Time Series Forecasting](https://zhuanlan.zhihu.com/p/582419707)|None
22-09-08|[W-Transformers](https://arxiv.org/abs/2209.03945)|Arxiv 2022|[W-Transformers: A Wavelet-based Transformer Framework for Univariate Time Series Forecasting](https://zhuanlan.zhihu.com/p/582419707)|[w-transformer](https://github.com/capwidow/w-transformer)
22-11-27|[PatchTST](https://arxiv.org/abs/2211.14730)|ICLR 2023|[A Time Series is Worth 64 Words: Long-term Forecasting with Transformers](https://zhuanlan.zhihu.com/p/602332939)|[PatchTST](https://github.com/yuqinie98/patchtst)



## RNN.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
17-04-13|[DeepAR](https://arxiv.org/abs/1704.04110)|IJoF 2019|[DeepAR: Probabilistic Forecasting with Autoregressive Recurrent Networks](https://zhuanlan.zhihu.com/p/542066911)|[DeepAR](https://github.com/brunoklein99/deepar)
22-05-16|[C2FAR](https://openreview.net/forum?id=lHuPdoHBxbg)|NIPS 2022|[C2FAR: Coarse-to-Fine Autoregressive Networks for Precise Probabilistic Forecasting](https://zhuanlan.zhihu.com/p/600602517)|[C2FAR](https://github.com/huaweicloud/c2far_forecasting)


## MLP.

| Date     | Method                                        | Conference | Paper Title and Paper Interpretation (In Chinese)            | Code                                           |
| -------- | --------------------------------------------- | ---------- | ------------------------------------------------------------ | ---------------------------------------------- |
| 17-05-25 | [ND](https://arxiv.org/abs/1705.09137)   | TNNLS 2017 | [Neural Decomposition of Time-Series Data for Effective Generalization](https://zhuanlan.zhihu.com/p/574742701)  | None |
| 19-05-24 | [NBeats](https://arxiv.org/abs/1905.10437)   | ICLR 2020 | [N-BEATS: Neural Basis Expansion Analysis For Interpretable Time Series Forecasting](https://zhuanlan.zhihu.com/p/572850227)      | [NBeats](https://github.com/philipperemy/n-beats) |
| 21-04-12 | [NBeatsX](https://arxiv.org/abs/2104.05522)   | IJoF 2022 | [Neural basis expansion analysis with exogenous variables: Forecasting electricity prices with NBEATSx](https://zhuanlan.zhihu.com/p/572955881)      | [NBeatsX](https://github.com/cchallu/nbeatsx) |
| 22-01-30 | [N-HiTS](https://arxiv.org/abs/2201.12886)   | Arxiv 2022 | [N-HiTS: Neural Hierarchical Interpolation for Time Series Forecasting](https://zhuanlan.zhihu.com/p/573203887)      | [N-HiTS](https://github.com/cchallu/n-hits) |
| 22-05-15 | [DEPTS](https://arxiv.org/abs/2203.07681)   | ICLR 2022 | [DEPTS: Deep Expansion Learning for Periodic Time Series Forecasting](https://zhuanlan.zhihu.com/p/572984932)      | [DEPTS](https://github.com/weifantt/depts) |
| 22-05-26 | [DLinear](https://arxiv.org/abs/2205.13504)   | Arxiv 2022 | [Are Transformers Effective for Time Series Forecasting?](https://zhuanlan.zhihu.com/p/569194246)      | [DLinear](https://github.com/cure-lab/DLinear) |
| 22-06-24 | [TreeDRNet](https://arxiv.org/abs/2206.12106) | Arxiv 2022 | TreeDRNet: A Robust Deep Model for Long Term Time Series Forecasting | None                                           |
| 22-07-04 | [LightTS](https://arxiv.org/abs/2207.01186) | Arxiv 2022 | Less Is More: Fast Multivariate Time Series Forecasting with Light Sampling-oriented MLP Structures | [LightTS](https://tinyurl.com/5993cmus)            |

## TCN.

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
| 22-09-22 | [MICN](https://openreview.net/forum?id=zt53IDUR1U) | ICLR 2023 | [MICN: Multi-scale Local and Global Context Modeling for Long-term Series Forecasting](https://zhuanlan.zhihu.com/p/603468264) | [MICN](https://github.com/whq13018258357/MICN)            |
| 22-09-22 | [TimesNet](https://arxiv.org/abs/2210.02186) | ICLR 2023 | [TimesNet: Temporal 2D-Variation Modeling for General Time Series Analysis](https://zhuanlan.zhihu.com/p/604100426) | None            |


## GNN.

| Date | Method | Conference | Paper Title and Paper Interpretation (In Chinese) | Code |
| ---- | ------ | ---------- | ------------------------------------------------- | ---- |
|      |        |            |                                                   |      |

## Normalizing Flow.

Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----

## Plug and Play.
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----
22-05-18|[FiLM](https://arxiv.org/abs/2205.08897)|NIPS 2022|FiLM: Frequency improved Legendre Memory Model for Long-term Time Series Forecasting|[FiLM](https://github.com/tianzhou2011/FiLM)


## Theory.
Date|Method|Conference| Paper Title and Paper Interpretation (In Chinese) |Code
-----|----|-----|-----|-----

## Other.
Date|Method|Conference|Paper Title and Paper Interpretation (In Chinese)|Code
-----|----|-----|-----|-----
