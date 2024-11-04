# Directory Composition

## Models
This directory contains two `.py` files:
* `models.py`: This file includes the two types of GCNN described in the paper: GCNN-I (`standard_gcnn`) and GCNN-II (`higher_order_polynomial_gcnn`).
* `utils.py`: This file provides the functions necessary for data loading, model training, and evaluation.

**Data Note**: The data are private and anonymized to ensure patient confidentiality. Ethical approval has been obtained from the UHF Research Ethics Committee (reference: 24/22, EC2091).

To request access to the original data, please contact oscar.escudero@urjc.es.

## Experiments: Inference Task
The following directories contain experiments related to GCNN-I and GCNN-II, employing different methods for graph estimation (correlations, smoothness, and dtw-hgd) and various graph representation techniques (STG and CPG):
* `Exp1_correlations`
* `Exp2_smoothness`
* `Exp3_dtw-hgd`

Each directory includes five files to ensure the reproducibility of the conducted experiments:

* `E1_GCNN-SingleGraph.ipynb`: Note that this experiment is associated with GCNN-I and does not utilize any graph representation. It employs the estimated graph for all time steps based on correlations, smoothness, or dtw-hgd (depending on the specific experiment). This serves as the baseline.

* `E2.1-CartesianProductGraph.ipynb`

* `E3.1-SpatioTemporalGraph.ipynb`

* `E4.1-CartesianProductGraph.ipynb`

* `E5.1-SpatioTemporalGraph.ipynb`

Additionally, there is a supplementary folder, `Exp4_OthersExp_Performance_by_threshold`, which contains experiments with various threshold values (0.6, 0.725, 0.85, 0.975).

Other relevant folders include:
* `hyperparameters`: This folder contains the optimal hyperparameters for each experiment, ensuring reproducibility.

## Experiments: Explainability Task
Each of the following folders contains a file for explainability analysis related to the various experiments available within:
* `Exp1_correlations`
* `Exp2_smoothness`
* `Exp3_dtw-hgd`

As an example, we provide files associated with GCNN-II and STG representation. You can replicate these interpretability experiments without needing access to the original data. To run all experiments in each folder, adjust the following hyperparameters:

### Parameters to Set:
- `way_to_build_graph = "dtw-hgd"`
- `typeOfGraph = "SpaceTimeGraph"` # Other option: `ProdGraph`

### Model:
- `typeGCN = "higher_order_polynomial_gcnn"` # Alternative option: `standard_gcnn`
- `K = [2,3]` # Alternative option: `K=[0]`

Some results from the experiments are stored in the `Interpretability` folder.

### Additional Material
The file `explainability_funct.py` includes various functions used in the interpretability analysis.
