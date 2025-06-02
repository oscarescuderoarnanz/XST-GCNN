import numpy as np, pandas as json, sys
import matplotlib.pyplot as plt
import json
from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.ticker import FormatStrFormatter  # <-- Importar el formateador

all_keys =  ['AMG', 'ATF', 'ATI', 'ATP', 'CAR', 'CF1', 'CF2', 'CF3', 'CF4', 'Others',
        'GCC', 'GLI', 'LIN', 'LIP', 'MAC', 'MON', 'NTI', 'OTR', 'OXA', 'PAP',
        'PEN', 'POL', 'QUI', 'SUL', 'TTC',
        'MV hours', '# pat$_{atb}$', '# pat$_{MDR}$',
        'CAR$_{n}$', 'PAP$_{n}$', 'Others$_{n}$',
        'QUI$_{n}$', 'ATF$_{n}$', 'OXA$_{n}$', 'PEN$_{n}$',
        'CF3$_{n}$', 'GLI$_{n}$', 'CF4$_{n}$', 'SUL$_{n}$',
        'NTI$_{n}$', 'LIN$_{n}$', 'AMG$_{n}$', 'MAC$_{n}$',
        'CF1$_{n}$', 'GCC$_{n}$', 'POL$_{n}$', 'ATI$_{n}$',
        'MON$_{n}$', 'LIP$_{n}$', 'TTC$_{n}$', 'OTR$_{n}$',
        'CF2$_{n}$', 'ATP$_{n}$', 
        '# pat$_{tot}$',
        'Post change',
        'Insulin', 'Art nutrition', 'Sedation', 'Relax', 'Hepatic$_{fail}$',
        'Renal$_{fail}$', 'Coagulation$_{fail}$', 'Hemodynamic$_{fail}$',
        'Respiratory$_{fail}$', 'Multiorganic$_{fail}$',  '# transfusions',
        'Vasoactive drug', 'NEMS', 'Tracheo$_{hours}$', 'Ulcer$_{hours}$',
        'Hemo$_{hours}$', 'C01 PICC 1',
        'C01 PICC 2', 'C02 CVC - RJ',
        'C02 CVC - RS', 'C02 CVC - LS', 'C02 CVC - RF',
        'C02 CVC - LJ', 'C02 CVC - LF', '# catheters']
    
if __name__ == "__main__":
    
    # Load masks
    mean_mask_mdr = np.load("mean_mask_MDR.npy")
    mean_mask_nonmdr = np.load("mean_mask_nonMDR.npy")

    # Load summary
    with open("explanation_summary.json", "r") as f:
        results = json.load(f)

    # 2D reshaping for visualization
    mean_mask_mdr_2d = mean_mask_mdr.reshape(80, 14)
    mean_mask_nonmdr_2d = mean_mask_nonmdr.reshape(80, 14)

    max_mdr = np.max(mean_mask_mdr_2d)
    max_nonmdr = np.max(mean_mask_nonmdr_2d)

    # Compares and adjusts the one with the lowest maximum
    if max_mdr > max_nonmdr:
        mean_mask_nonmdr_2d = mean_mask_nonmdr_2d * (max_mdr / max_nonmdr) if max_nonmdr != 0 else mean_mask_nonmdr_2d
        print("The maximum is in mean_mask_mdr_2d (value = {:.3f})".format(max_mdr))
    else:
        mean_mask_mdr_2d = mean_mask_mdr_2d * (max_nonmdr / max_mdr) if max_mdr != 0 else mean_mask_mdr_2d
        print("The maximum is in mean_mask_nonmdr_2d (value = {:.3f})".format(max_nonmdr))

    vmin = min(np.min(mean_mask_nonmdr_2d), np.min(mean_mask_mdr_2d))
    vmax = max(np.max(mean_mask_nonmdr_2d), np.max(mean_mask_mdr_2d))

    diff_mask_2d = mean_mask_nonmdr_2d - mean_mask_mdr_2d
    diff_abs_max = np.max(np.abs(diff_mask_2d))

    timesteps = np.arange(1, 15)

    fig, axes = plt.subplots(1, 3, figsize=(20, 18))

    # --- non-MDR ---
    im0 = axes[0].imshow(mean_mask_nonmdr_2d, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    axes[0].set_xticks(np.arange(14))
    axes[0].set_xticklabels(timesteps, rotation=90, fontsize=14)
    axes[0].set_yticks(np.arange(len(all_keys)))
    axes[0].set_yticklabels(all_keys, fontsize=14)
    divider0 = make_axes_locatable(axes[0])
    cax0 = divider0.append_axes("right", size="3.25%", pad=0.1)
    cbar0 = plt.colorbar(im0, cax=cax0, orientation='vertical')
    cbar0.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cax0.tick_params(labelsize=14)

    # --- MDR ---
    im1 = axes[1].imshow(mean_mask_mdr_2d, aspect='auto', cmap='bwr', vmin=vmin, vmax=vmax)
    axes[1].set_xticks(np.arange(14))
    axes[1].set_xticklabels(timesteps, rotation=90, fontsize=14)
    axes[1].set_yticks(np.arange(len(all_keys)))
    axes[1].set_yticklabels(all_keys, fontsize=14)
    divider1 = make_axes_locatable(axes[1])
    cax1 = divider1.append_axes("right", size="3.25%", pad=0.1)
    cbar1 = plt.colorbar(im1, cax=cax1, orientation='vertical')
    cbar1.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cax1.tick_params(labelsize=14)

    # --- Difference MDR - non-MDR ---
    boundaries = np.linspace(-diff_abs_max, diff_abs_max, num=9)
    im2 = axes[2].imshow(diff_mask_2d, aspect='auto', cmap='bwr', vmin=-diff_abs_max, vmax=diff_abs_max)
    axes[2].set_xticks(np.arange(14))
    axes[2].set_xticklabels(timesteps, rotation=90, fontsize=14)
    axes[2].set_yticks(np.arange(len(all_keys)))
    axes[2].set_yticklabels(all_keys, fontsize=14)
    divider2 = make_axes_locatable(axes[2])
    cax2 = divider2.append_axes("right", size="3.25%", pad=0.1)
    cbar2 = plt.colorbar(im2, cax=cax2, orientation='vertical', boundaries=boundaries, ticks=boundaries)
    cbar2.ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    cax2.tick_params(labelsize=16)

    # Etiquetas (a), (b), (c)
    axes[0].set_xlabel("(a)", fontsize=18, labelpad=20)
    axes[1].set_xlabel("(b)", fontsize=18, labelpad=20)
    axes[2].set_xlabel("(c)", fontsize=18, labelpad=20)

    plt.tight_layout()
    plt.savefig("classwise_importance_maps_gnnexplainer.pdf")
    plt.show()
