# Panneau RTL éditable

Le notebook `notebooks/02_quantize_MLP.ipynb` sélectionne automatiquement la
cible et l'architecture. Les fichiers de ce dossier restent les sources
SystemVerilog modifiables; `src/rtl/generated/` n'est qu'un rendu local et n'est
plus versionné.

| Configuration | Sources de calcul | Interface carte |
| --- | --- | --- |
| Ultra96 séquentiel | `src_ultra_sequential/` | AXI-Stream |
| Ultra96 pipeline | `src_ultra_pipeline/` | AXI-Stream |
| Lattice séquentiel | `src_lattice_sequential/` | UART |
| Lattice pipeline | `src_lattice_pipeline/` | UART |

Le fichier `include/scales.vh` est un exemple lisible. À chaque quantification,
un nouveau `scales.vh` cohérent avec le modèle, les largeurs et la cible est
exporté dans `data/quantized_data/<name>/<target>/<architecture>/`, puis copié
dans le rendu RTL local.

## Règles de modification

- modifier uniquement les quatre panneaux `src_*` et `include/scales.vh`;
- ne jamais modifier ni committer `generated/`;
- garder le même contrat arithmétique entre séquentiel et pipeline;
- utiliser `ACT_W` pour les entrées, activations et sorties;
- utiliser `DSP_PW` pour le MAC et l'interpolation de LUT (48 bits sur Ultra96,
  32 bits sur Lattice).

Le mode séquentiel mutualise son MAC. Le mode pipeline instancie une couche par
étage; leur résultat numérique doit rester identique. Le simulateur Python
`src/py/fpga_sim.py` reproduit les décalages arithmétiques, la troncature en
complément à deux, l'ordre des MAC et l'interpolation des LUT.

Les deux cibles partagent désormais les mêmes cœurs numériques V2 : les fichiers
de calcul séquentiel sont identiques entre Ultra96 et Lattice, tout comme les
fichiers de calcul pipeline. Seules les enveloppes de communication diffèrent :
AXI-Stream pour Ultra96, UART pour Lattice.

Le contrat de largeur ne comporte que quatre macros : `ACT_W`, `WEIGHT_W`,
`BIAS_W` et `DSP_PW`. Les anciens alias `DATA_WIDTH*` ne sont plus acceptés.
