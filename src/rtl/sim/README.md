# Latence du MLP RTL séquentiel Ultra96

Depuis la racine du dépôt :

```bash
source /tools/xilinx/2025.2/Vivado/settings64.sh
src/rtl/sim/run_mlp_latency.sh
```

Le testbench `src/rtl/tb/TB_mlp_latency.sv` envoie quatre vecteurs INT16 :
zéros, valeurs positives, valeurs de signes alternés, et valeurs limites.
Il mesure, sur les fronts montants, le temps entre `dut.input_consume_pulse`
et `dut.mlp_done_pulse` (écriture du dernier résultat en RAM).
Cette mesure exclut le chargement des entrées et la lecture des sorties AXI.
La colonne `axis_cycles` mesure séparément le premier transfert d’entrée jusqu’au
dernier transfert de sortie. Le test vérifie les données non indéterminées et le
protocole de sortie, sans prétendre valider l’exactitude numérique des exemples.

Résultat à 100 MHz : **7674 cycles = 76,74 µs** de calcul pour chaque exemple.
Avec les échanges AXI du testbench : 7690 cycles = 76,90 µs.

Résultats : `vivado/sim/mlp_latency/mlp_latency.csv`.
Chronogrammes : `vivado/sim/mlp_latency/mlp_latency.wdb`, ouvrables dans Vivado.
Cette mesure concerne les sources Ultra96 générées, identiques au top fourni ;
elle ne constitue pas une mesure de la version Lattice.
