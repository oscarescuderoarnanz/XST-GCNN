from __future__ import annotations

import types
import torch, numpy as np, pandas as pd, json, sys
import torch
from pathlib import Path
import utils_explainability as utils
from torch_geometric.data import Data
from torch_geometric.utils import dense_to_sparse
import matplotlib.pyplot as plt
from torch_geometric.explain import Explainer, GNNExplainer
from torch_geometric.explain import ModelConfig
import torch.nn as nn
import time 
import json

sys.path.append("../step3_GCNNs/")
import models # type: ignore


class Wrapper(nn.Module):
    def __init__(self, core, pool_graph: bool = True):
        super().__init__()
        self.core = core
        self.pool_graph = pool_graph

    def forward(self, x, edge_index=None, batch=None):
        
        output, *_ = self.core(x)
        
        if self.pool_graph:
        
            if batch is None:
                output = output.mean(0, keepdim=True)
            else:
                output = torch.zeros(batch.max()+1, 1, device=output.device) \
                         .index_add_(0, batch, output) / torch.bincount(batch).unsqueeze(1)
        return output


    
if __name__ == "__main__":
    
    args = types.SimpleNamespace(
        folder="s3",      # "s1", "s2" o "s3"
        gpu=0,             # -1 for CPU
    )

    device = torch.device(f"cuda:{args.gpu}" if args.gpu >= 0 else "cpu")
    print("Dispositivo:", device)

    way_graph = "dtw-hgd"
    param_file = Path(str("../step3_GCNNs/hyperparameters/" + way_graph + "/#E5.1-SpaceGraph_th_0.975.json"))
    with open(param_file) as f:
        best_split = json.load(f)[args.folder]

    norm = "robustNorm"
    A_path = Path("../step2_graphRepresentation") / way_graph / args.folder / f"SpaceTimeGraph_Xtr_{norm}_th_0.975.csv"
    A = torch.tensor(pd.read_csv(A_path).values, dtype=torch.float32, device=device)

    params = {
        "h_dropout": [0.0, 0.15, 0.3],
        "h_hid_lay": [4, 8, 16, 32, 64],
        "h_layers" : [1, 2, 3, 4, 5, 6],
        "seed"     : [42, 76, 124, 163, 192, 205, 221, 245, 293],
        "K"        : [2, 3],
        "fc_layer" : [[1120, 1]],
    }

    n_layers = params["h_layers"][best_split["n_lay"]]
    hid_dim  = params["h_hid_lay"][best_split["hid_lay"]]
    dropout  = params["h_dropout"][best_split["dropout"]]
    K        = params["K"][best_split["K"]]
    seed     = params["seed"][best_split["seed"]]
    fc_layer = params["fc_layer"]

    torch.manual_seed(seed)
    core = models.higher_order_polynomial_gcnn(n_layers, dropout, hid_dim, A, 1, 1, K, fc_layer, seed).to(device)
    model = Wrapper(core).to(device)
    model.eval()

    start_time = time.time()

    X_train, _, _, y_train, _, _ = utils.load_data(norm, device, args.folder, False)
    edge_index, _ = dense_to_sparse(A)
    edge_index = edge_index.to(device)  

    explainer = Explainer(
        model=model,
        algorithm=GNNExplainer(epochs=500, 
                               lr=0.01),
        explanation_type='model',
        edge_mask_type=None,
        node_mask_type='object',
        model_config=ModelConfig(
            mode='binary_classification',
            task_level='graph',
            return_type='probs',
        ),
    )

    idx_mdr = (y_train == 1).nonzero(as_tuple=True)[0].tolist()
    idx_nonmdr = (y_train == 0).nonzero(as_tuple=True)[0].tolist()

    all_masks_mdr = []
    all_masks_nonmdr = []


    def get_mask(features):
        # features: [FT] or [FT, 1], needs to be on device
        features = torch.as_tensor(features, dtype=torch.float32, device=device)
        if features.dim() == 1:
            features = features.unsqueeze(-1)  # [FT, 1]
        # batch in device too
        batch = torch.zeros(features.shape[0], dtype=torch.long, device=device)
        data = Data(
            x=features,
            edge_index=edge_index,
            batch=batch
        )
        explanation = explainer(x=data.x, edge_index=data.edge_index, batch=data.batch)
        for name in ['node_mask', 'node_feat_mask', 'feature_mask']:
            if hasattr(explanation, name):
                return getattr(explanation, name).detach().cpu().numpy().flatten()
        return None

    # Process MDR patients
    print("MDR patients:", len(idx_mdr), "- X_train:", len(X_train))
    for idx in idx_mdr:
        print("Processing patient:", idx)
        mask = get_mask(X_train[idx])
        if mask is not None:
            all_masks_mdr.append(mask)

    # Process non-MDR patients
    print("non-MDR patients:", len(idx_nonmdr))
    for idx in idx_nonmdr:
        print("Processing patient:", idx)
        mask = get_mask(X_train[idx])
        if mask is not None:
            all_masks_nonmdr.append(mask)

    print("Total MDR masks:", len(all_masks_mdr))
    print("Total non-MDR masks:", len(all_masks_nonmdr))

    # Compute class-wise averages
    mean_mask_mdr = np.mean(np.stack(all_masks_mdr), axis=0)
    mean_mask_nonmdr = np.mean(np.stack(all_masks_nonmdr), axis=0)

    # 2D reshaping for visualization
    mean_mask_mdr_2d = mean_mask_mdr.reshape(80, 14)
    mean_mask_nonmdr_2d = mean_mask_nonmdr.reshape(80, 14)

    # Plot per-class importance maps
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].imshow(mean_mask_nonmdr_2d, aspect='auto', cmap='coolwarm')
    axes[0].set_title("Average Importance (non-MDR)")
    axes[0].set_xlabel("Timestep")
    axes[0].set_ylabel("Feature")

    axes[1].imshow(mean_mask_mdr_2d, aspect='auto', cmap='coolwarm')
    axes[1].set_title("Average Importance (MDR)")
    axes[1].set_xlabel("Timestep")
    axes[1].set_ylabel("Feature")

    plt.tight_layout()
    plt.savefig("classwise_importance_map.pdf")
    plt.show()

    # Difference between classes
    diff_mask_2d = mean_mask_mdr_2d - mean_mask_nonmdr_2d
    plt.figure(figsize=(8, 6))
    plt.imshow(
        diff_mask_2d, aspect='auto', cmap='bwr',
        vmin=-np.max(np.abs(diff_mask_2d)), vmax=np.max(np.abs(diff_mask_2d))
    )
    plt.colorbar(label="Importance (MDR - non-MDR)")
    plt.title("Class-wise Importance Difference")
    plt.xlabel("Timestep")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig("importance_difference_map.pdf")
    plt.show()

    # Save masks
    np.save("mean_mask_MDR.npy", mean_mask_mdr)
    np.save("mean_mask_nonMDR.npy", mean_mask_nonmdr)

    # Save total runtime
    total_time = time.time() - start_time
    print(f"Total explanation time: {total_time:.2f} seconds")

    # Save summary to JSON
    results = {
        "total_time_seconds": round(total_time, 2),
        "num_patients_MDR": len(all_masks_mdr),
        "num_patients_nonMDR": len(all_masks_nonmdr),
        "mask_dimension": len(mean_mask_mdr)
    }

    with open("explanation_summary.json", "w") as f:
        json.dump(results, f, indent=4)
