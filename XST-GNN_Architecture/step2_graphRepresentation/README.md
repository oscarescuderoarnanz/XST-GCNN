# Directory Composition

## Graph Representations
The directory contains three folders with graph representations derived from estimation methods using correlations, dtw-hgd, and smoothness. Each folder includes representations for both STG and CPG. The folders are as follows:
* `correlations`
* `smoothness`
* `dtw-hgd`

The files used to generate these representations are:
* `graphRepresentation_as_CPG.ipynb`
* `graphRepresentation_as_STG.ipynb`

These files produce visual figures, which can be found in the `Figures` folder.

Additionally, there is a file for visualizing the generated results:
* `EstimatedGraphsAnalysis-Visualizations.ipynb`

By adjusting specific parameters, you can explore each graph estimation and representation.

## Graph Metrics

The file `EstimatedGraphsAnalysis-Visualizations.ipynb` also facilitates the evaluation of complexity metrics, including edge density and edge entropy, for both CPG and STG across various thresholds and three data splits.

Results are presented in a table format suitable for LaTeX display.
