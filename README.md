# MLP sur FPGA : entraînement, quantification et déploiement

Ce dépôt fournit une chaîne complète pour entraîner un perceptron multicouche
(MLP) avec PyTorch, le quantifier en virgule fixe, générer son implémentation
SystemVerilog et l'exécuter sur FPGA.

Deux architectures matérielles et deux plateformes sont prises en charge :

| Architecture | Principe | Ultra96-V2 | Lattice iCE40UP5K |
|---|---|---:|---:|
| `sequential` | Un datapath partagé traite les couches successivement | AXI-Stream / DMA | UART |
| `pipeline` | Une unité matérielle est générée pour chaque couche | AXI-Stream / DMA | UART |

La recette actuelle utilise le réseau Walter suivant :

```text
6 -> 66 -> 50 -> 35 -> 28 -> 20 -> 2
```

Les poids, biais, facteurs d'échelle, LUT d'activation et vecteurs de test sont
générés automatiquement à partir du modèle PyTorch.

## Fonctionnalités

- entraînement et validation croisée d'un MLP avec PyTorch ;
- quantification en virgule fixe configurable ;
- export des poids, biais, activations et facteurs d'échelle ;
- activations directes ou par LUT ;
- génération RTL pour architectures séquentielle et pipeline ;
- génération de projets Xilinx Vivado et Lattice Radiant ;
- simulation avec comparaison automatique des sorties ;
- validation sur carte via AXI DMA ou UART ;
- export des mesures FPGA au format CSV.

## Arborescence cible du dépôt GitHub

L'arborescence ci-dessous est celle recommandée pour la version publique du
projet. Les dossiers de build, environnements locaux et grosses archives
d'expériences ne doivent pas être versionnés.

```text
.
├── README.md
├── LICENSE                         # licence du projet à ajouter
├── requirements.txt
├── pyproject.toml
├── .gitignore                      # règles Git à compléter
│
├── notebooks/
│   ├── 01_train_MLP.ipynb
│   ├── 02_quantize_MLP.ipynb
│   ├── 03_generate_TCL_lattice_pipeline.ipynb
│   ├── 03_generate_TCL_ultra96_pipeline.ipynb
│   ├── 04_push_data_lattice.py
│   └── 04_push_data_ultra96.ipynb
│
├── src/
│   ├── py/
│   │   ├── config.py                # configuration de quantification
│   │   ├── quantize.py              # quantification du modèle
│   │   └── ...                      # autres modules Python
│   ├── tcl/
│   │   ├── remodel_lattice_project.py
│   │   └── remodel_vivado_project.py
│   └── rtl/
│       ├── include/                # configuration RTL de secours
│       ├── src_lattice_sequential/ # templates Lattice séquentiels
│       ├── src_lattice_pipeline/   # templates Lattice pipeline
│       ├── src_ultra_sequential/   # templates Ultra96 séquentiels
│       ├── src_ultra_pipeline/     # templates Ultra96 pipeline
│       ├── generated/              # RTL concret généré depuis le modèle
│       └── tb/                     # testbenches séquentiel et pipeline
│
├── data/
│   ├── walter/                     # dataset et checkpoint PyTorch
│   └── quantized_data/             # artefacts quantifiés et vecteurs de test
│
├── lattice/
│   └── create_soc_mlp_project_lattice.tcl
│
├── vivado/
│   ├── create_soc_mlp_project_vivado.tcl
│   ├── Extract_hardware_send_to_ultra.sh
│   └── ip_repo/
│       └── SOC_MLP/                # IP Vivado générée ou version de référence
│
└── results/
    ├── lattice/                    # résultats UART de référence
    ├── ultra96/                    # résultats AXI DMA de référence
    └── paper/                      # résultats retenus pour publication
```

Avant publication, il est recommandé de déplacer le contenu utile de
`Safe_All_Model_for_paper/` vers `results/paper/`, puis d'exclure les fichiers
intermédiaires et les bitstreams non nécessaires.

## Organisation du workflow

```text
Dataset
  |
  v
Entraînement PyTorch
  |
  v
Quantification + export des mémoires + vecteurs de test
  |
  +-------------------------+
  |                         |
  v                         v
RTL séquentiel          RTL pipeline
  |                         |
  +------------+------------+
               |
       +-------+-------+
       |               |
       v               v
  Ultra96 / AXI    Lattice / UART
```

## Prérequis

### Logiciels communs

- Python 3.10 ou supérieur ;
- Jupyter Notebook ou JupyterLab ;
- PyTorch, NumPy, pandas et Matplotlib ;
- un simulateur SystemVerilog compatible.

### Pour Ultra96-V2

- Xilinx Vivado avec le composant `xczu3eg-sbva484-1-e` disponible ;
- une image PYNQ fonctionnelle sur la carte ;
- accès réseau à la carte pour le transfert du bitstream.

### Pour Lattice

- Lattice Radiant avec le composant `iCE40UP5K-SG48I` disponible ;
- un adaptateur UART accessible depuis la machine hôte.

## Installation

Depuis la racine du dépôt :

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -e .
```

Le mode éditable découvre les modules Python directement dans `src/py/` et les
rend disponibles dans les notebooks et les scripts du projet.

## 1. Préparer les données

Le dataset Walter doit être placé ici :

```text
data/walter/dataset_walter.npz
```

La recette actuelle attend notamment :

- `ph_val` : entrées du réseau, de forme `(N, 6)` ;
- `pos` : cibles du réseau, de forme `(N, 2)`.

Pour utiliser un autre dataset ou une autre topologie, adapter la configuration
du notebook d'entraînement avant de régénérer tous les artefacts FPGA.

## 2. Entraîner le modèle

Ouvrir et exécuter :

```text
notebooks/01_train_MLP.ipynb
```

Le notebook entraîne le modèle, réalise la validation croisée et sauvegarde le
meilleur checkpoint dans :

```text
data/walter/mlp_walter.pt
```

## 3. Quantifier et exporter

Ouvrir :

```text
notebooks/02_quantize_MLP.ipynb
```

Choisir l'architecture cible dans la configuration :

```python
qcfg = QuantConfig(
    fpga_mode="sequential",  # ou "pipeline"
)
```

### Export séquentiel

Les données de toutes les couches sont concaténées :

```text
weight_all.mem
bias_all.mem
act_all.mem
```

### Export pipeline

Les données sont séparées par couche :

```text
weight_layer0.mem
bias_layer0.mem
act_layer0.mem
...
```

### Artefacts communs

Le notebook génère également :

```text
data/quantized_data/
├── mem_layout.json
├── scales.vh
├── dataset_walter_quantized.npz
├── tb_inputs_stream.mem
├── tb_expected_outputs.mem
└── distribution_analysis_report.pdf
```

`mem_layout.json` décrit la disposition des mémoires. `scales.vh` contient la
topologie, les largeurs, les facteurs d'échelle et le type d'activation de
chaque couche. Ces deux fichiers assurent la liaison entre Python et le RTL.

## 4. Générer le RTL et les projets FPGA

Le mode sélectionné lors de la quantification doit être identique au mode
sélectionné lors de la génération du projet.

### Lattice Radiant

La méthode guidée consiste à exécuter :

```text
notebooks/03_generate_TCL_lattice_pipeline.ipynb
```

La génération peut aussi être lancée directement :

```bash
python src/tcl/remodel_lattice_project.py \
  --fpga-mode sequential \
  --template-rel src/rtl/src_lattice_sequential
```

Remplacer `sequential` par `pipeline` et le chemin du template par
`src/rtl/src_lattice_pipeline` pour générer la version pipeline.

Le RTL concret est écrit dans `src/rtl/generated/src_lattice_<mode>/` et le script
Radiant dans `lattice/create_soc_mlp_project_lattice.tcl`.

Depuis la console Tcl de Radiant :

```tcl
source lattice/create_soc_mlp_project_lattice.tcl
```

Pour créer le projet et lancer le flot complet :

```tcl
source lattice/create_soc_mlp_project_lattice.tcl -tclargs --run
```

### Ultra96-V2 avec Vivado

La méthode guidée consiste à exécuter :

```text
notebooks/03_generate_TCL_ultra96_pipeline.ipynb
```

La génération peut aussi être lancée directement :

```bash
python src/tcl/remodel_vivado_project.py --fpga-mode sequential
```

Le RTL concret est écrit dans `src/rtl/generated/src_ultra_<mode>/` et le script
Vivado dans `vivado/create_soc_mlp_project_vivado.tcl`.

Depuis la console Tcl de Vivado :

```tcl
source vivado/create_soc_mlp_project_vivado.tcl
```

Pour créer le projet, synthétiser, implémenter et générer le bitstream :

```tcl
source vivado/create_soc_mlp_project_vivado.tcl -tclargs --run
```

Le script Vivado peut également recevoir `--project-name`, `--project-dir`,
`--part`, `--board-part`, `--jobs`, `--skip-ip` ou `--skip-bd`.

## Simulation RTL

Deux testbenches sont fournis :

```text
src/rtl/tb/TB_soc_mlp_sequential.sv
src/rtl/tb/TB_soc_mlp_pipeline.sv
```

Ils lisent les vecteurs générés par la quantification :

```text
data/quantized_data/tb_inputs_stream.mem
data/quantized_data/tb_expected_outputs.mem
```

Dans Vivado :

```tcl
launch_simulation
run all
```

Le testbench pipeline ajoute automatiquement les trames nécessaires pour vider
le pipeline et réaligner les sorties avec les échantillons d'entrée.

## Validation sur Lattice par UART

Lister les ports série disponibles :

```bash
python notebooks/04_push_data_lattice.py --list-ports
```

Tester une architecture séquentielle :

```bash
python notebooks/04_push_data_lattice.py \
  --port /dev/ttyUSB0 \
  --baud 115200 \
  --fpga-mode sequential
```

Pour la version pipeline, utiliser `--fpga-mode pipeline`. Le script déduit la
taille du réseau depuis `scales.vh`, gère le délai du pipeline et écrit un
rapport CSV dans `Result/`.

## Validation sur Ultra96 par AXI DMA

Après génération du bitstream, le script suivant copie les fichiers `.bit` et
`.hwh`, puis les transfère vers la carte :

```bash
PROJECT_NAME=SOC_MLP_ultra \
REMOTE_IP=192.168.3.1 \
./vivado/Extract_hardware_send_to_ultra.sh
```

Sur l'Ultra96, ouvrir et exécuter :

```text
notebooks/04_push_data_ultra96.ipynb
```

Configurer `FPGA_MODE` avec `sequential` ou `pipeline`. Le notebook charge
l'overlay PYNQ, envoie le dataset quantifié par AXI DMA, réaligne les sorties
et produit un rapport CSV.

## Fichiers générés et versionnement

Les éléments suivants sont reproductibles et devraient généralement être
ignorés par Git :

```text
.venv/
vivado/.Xil/
vivado/build/
lattice/build/
__pycache__/
*.jou
*.log
*.str
```

Les datasets privés, checkpoints, bitstreams et rapports lourds doivent être
publiés uniquement si leur licence et leur taille le permettent. Pour les gros
fichiers nécessaires à la reproductibilité, utiliser Git LFS ou une archive de
données versionnée séparément.

## Résultats

Les résultats matériels sont exportés au format CSV. Ils contiennent notamment :

- les entrées et cibles ;
- les prédictions logicielles de référence ;
- les sorties quantifiées du FPGA ;
- les erreurs par sortie ;
- le mode FPGA et le délai de pipeline ;
- les trames brutes transmises et reçues lorsque demandé.

Les résultats actuellement présents dans `Result/` servent de références pour
les exécutions UART séquentielle et pipeline.

## Limites actuelles

- la topologie par défaut est adaptée au modèle Walter ;
- les contraintes Lattice sont spécifiques au brochage utilisé ;
- Vivado et Radiant restent nécessaires pour produire les bitstreams ;
- les versions des outils FPGA et les références exactes des cartes doivent
  être documentées avant une publication reproductible ;
- le dataset Walter doit être fourni séparément si sa diffusion est restreinte.

## Citation

Une section de citation pourra être ajoutée lors de la publication de l'article :

```bibtex
@article{a_completer,
  title   = {À compléter},
  author  = {À compléter},
  journal = {À compléter},
  year    = {À compléter}
}
```

## Licence

Ajouter un fichier `LICENSE` avant la publication du dépôt. La licence choisie
doit être compatible avec le code, les outils tiers, le dataset et les
artefacts matériels distribués.
