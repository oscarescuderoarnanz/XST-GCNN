# XST-GNN Architecture Directory Structure

This directory is organized into three main steps, representing the core blocks of our proposed architecture.

## step1_graphEstimation
This folder contains graph estimation methods, which form our architecture's foundational step. Various techniques for estimating graphs from data are included here: i) correlations; ii) smoothness; and iii) hgd-dtw. 

## step2_graphRepresentation
This folder includes approaches for graph representation. Once the graphs are estimated, these methods define how they are structured and represented, preparing them for further analysis and processing.

## step3_GCNNs
This folder contains the proposed GCNNs used to perform inference tasks and explainability analysis. These models are designed to leverage the estimated and represented graphs to extract meaningful insights and provide interpretability in the context of our research goals.
