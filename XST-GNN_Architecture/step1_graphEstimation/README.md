# Directory Composition

## Graph Representations
We have three folders within the `estimatedGraphs` directory, each containing graph estimations generated using correlations, dtw-hgd, and smoothness. The folders are organized as follows:
* `correlations`
* `smoothness`
* `dtw-hgd`

The files used to generate these representations are:
* `HGD-DTW.ipynb`
* `Correlations&Smoothness.ipynb`

## hgd-dtw.py

This file contains the code for the proposed metric based on Gower distance, named Heterogeneous Gower Distance. It is used to estimate a graph for each time point and includes its implementation combined with DTW, which allows for graph estimation across all time points.
