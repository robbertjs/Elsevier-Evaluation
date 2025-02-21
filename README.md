# Reproducibility, Replicability, and Insights into Visual Document Retrieval with Late Interaction 👀

This repository contains the code used to reproduce the training of the vision retrievers with late interaction in the [*ColPali: Efficient Document Retrieval with Vision Language Models*](https://arxiv.org/abs/2407.01449) paper. The original code is available [Here](https://github.com/illuin-tech/colpali)

## Introduction
Visual Document Retrieval (VDR) is an emerging research area that focuses on encoding and retrieving document images directly, bypassing the dependence on Optical Character Recognition (OCR) for document search. A recent advancement in VDR was introduced by Colpali, which significantly improved retrieval effectiveness through a late interaction mechanism. Colpali's approach demonstrated substantial performance gains over existing baselines that do not use late interaction on an established benchmark. In this study, we investigate the reproducibility and replicability of VDR methods with and without late interaction mechanisms by systematically evaluating their performance across multiple pre-trained vision-language models.

## Setup
The reproducibility used Python 3.11.6 and PyTorch 2.5 to train and test models. All other environment setup is followed by Colpali's setting, you can find more deatiled information in [Here](https://github.com/illuin-tech/colpali). Don't use 'pip install colpali-engine' directly for this reproducibility, the code has changed a bit from the original. 

To install the training package, run:

```bash
pip install -e .
```
To install the inference package, run:

```bash
cd vidore-benchmark
pip install -e .
```

## Reproducibility

* RQ1.1 Can we completely reproduce CoPali and achieve the same effectiveness on visual document retrieval?

* RQ1.2 Does CoPali significantly outperform the single-vector variants in terms of the effectiveness?

Results of RQ1.1 and RQ1.2 in Table 1 are the outputs of the experiments run using scripts in bash_RQ1.

## Replicability

* RQ2.1 Does the image document embedding consistently outperform the text for first-stage retrieval? 

Results of RQ2.1 in table 2 are the outputs of the experiments run using scripts in bash_RQ21.


*  RQ2.2 Does the image consistently outperform the OCR-based text document for first-state retrieval when the the embedding index size increases? 

Results of RQ2.2 in figure 2 are the outputs of the experiments run using scripts in bash_RQ22.

## Insights

*  RQ3.1 Do significant differences exist in the visual features of retrieved image documents compared to those not retrieved?

Results of RQ3.1 in figure 3 and 4 are the outputs of the experiments run using scripts in bash_RQ31.


*  RQ3.2 To What Extent Does Semantic Matching Rely on Special and Query Token Matching? Reproduce the results of RQ3.2, just simply run the following bash script in bash_RQ32.

Results of RQ3.2 in figure 5, table 3 and 4 are the outputs of the experiments run using scripts in bash_RQ32.
