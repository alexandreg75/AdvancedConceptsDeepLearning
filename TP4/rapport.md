## Exercice 1 — Initialisation du TP et smoke test PyG (Cora)

### 1.a — Structure du dossier `TP4`

```text
TP4
TP4/configs
TP4/configs/baseline_mlp.yaml
TP4/configs/gcn.yaml
TP4/configs/sage_sampling.yaml
TP4/img
TP4/outputs
TP4/rapport.md
TP4/src
TP4/src/smoke_test.py
TP4/src/utils.py

```

### 1.e — Résultat du smoke test PyG sur Cora

=== Environment ===
torch: 2.10.0+cu128
cuda available: True
device: cuda
gpu: NVIDIA H100 NVL MIG 1g.12gb
gpu_total_mem_gb: 10.75

=== Dataset (Cora) ===
num_nodes: 2708
num_edges: 10556
num_node_features: 1433
num_classes: 7
train/val/test: 140 500 1000

OK: smoke test passed.


## Exercice 2 — Baseline tabulaire : MLP (features seules)

### 2.g — Pourquoi séparer train / val / test ?

On calcule les métriques séparément sur `train_mask`, `val_mask` et `test_mask` pour distinguer trois usages différents.  
Le jeu d’entraînement permet de vérifier si le modèle apprend bien sur les nœuds vus pendant l’optimisation.  
Le jeu de validation sert à suivre la généralisation pendant l’entraînement et à repérer un éventuel surapprentissage.  
Enfin, le jeu de test donne une estimation plus neutre des performances finales sur des nœuds jamais utilisés pour ajuster les poids ni guider les choix de configuration.  
D’un point de vue ingénieur, cette séparation évite de surestimer la qualité du modèle et permet de comparer proprement plusieurs approches.

---

### 2.h — Entraînement de la baseline MLP

Configuration utilisée :

```text
device: cuda
epochs: 200
config: {'seed': 42, 'device': 'cuda', 'epochs': 200, 'lr': 0.01, 'weight_decay': 0.0005, 'mlp': {'hidden_dim': 64, 'dropout': 0.5}}
```

Extrait du log d'entrainement :

epoch=001 loss=1.9512 train_acc=0.3429 val_acc=0.3580 test_acc=0.3500 train_f1=0.2493 val_f1=0.1329 test_f1=0.1357 epoch_time_s=0.8071
epoch=020 loss=0.0343 train_acc=1.0000 val_acc=0.5800 test_acc=0.5610 train_f1=1.0000 val_f1=0.5697 test_f1=0.5516 epoch_time_s=0.0016
epoch=100 loss=0.0114 train_acc=1.0000 val_acc=0.5740 test_acc=0.5740 train_f1=1.0000 val_f1=0.5681 test_f1=0.5607 epoch_time_s=0.0016
epoch=200 loss=0.0073 train_acc=1.0000 val_acc=0.5500 test_acc=0.5740 train_f1=1.0000 val_f1=0.5410 test_f1=0.5619 epoch_time_s=0.0016
total_train_time_s=1.1341
train_loop_time=2.1500

Résultats finaux sur le jeu de test :

Test Accuracy : 0.5740

Test Macro-F1 : 0.5619

Temps total d’entraînement (somme des epochs) : 1.1341 s

Temps total de boucle d’entraînement : 2.1500 s


Le MLP apprend très vite et atteint une accuracy de 100 % sur le train set, ce qui montre qu’il mémorise facilement les features des nœuds d’entraînement.
En revanche, les performances sur validation et test plafonnent autour de 0.55–0.58, ce qui suggère un surapprentissage et montre les limites d’une approche purement tabulaire sur Cora.
Cette baseline est donc utile pour mesurer ensuite ce que l’information de graphe apporte réellement avec les modèles GNN.


## Exercice 3 — Baseline GNN : GCN (full-batch) + comparaison perf/temps

### 3.e — Entraînement du GCN et comparaison avec le MLP

#### Extrait final — MLP

```text
device: cuda
model: mlp
epochs: 200
config: {'seed': 42, 'device': 'cuda', 'epochs': 200, 'lr': 0.01, 'weight_decay': 0.0005, 'mlp': {'hidden_dim': 64, 'dropout': 0.5}}

epoch=001 loss=1.9512 train_acc=0.3429 val_acc=0.3580 test_acc=0.3500 train_f1=0.2493 val_f1=0.1329 test_f1=0.1357 epoch_time_s=0.4771
epoch=100 loss=0.0114 train_acc=1.0000 val_acc=0.5740 test_acc=0.5740 train_f1=1.0000 val_f1=0.5681 test_f1=0.5607 epoch_time_s=0.0015
epoch=200 loss=0.0073 train_acc=1.0000 val_acc=0.5500 test_acc=0.5740 train_f1=1.0000 val_f1=0.5410 test_f1=0.5619 epoch_time_s=0.0015
total_train_time_s=0.7804
train_loop_time=1.8251
```

#### Extrait final — GCN

```text
device: cuda
model: gcn
epochs: 200
config: {'seed': 42, 'device': 'cuda', 'epochs': 200, 'lr': 0.01, 'weight_decay': 0.0005, 'gcn': {'hidden_dim': 64, 'dropout': 0.5}}

epoch=001 loss=1.9497 train_acc=0.8929 val_acc=0.5820 test_acc=0.5740 train_f1=0.8916 val_f1=0.5842 test_f1=0.5822 epoch_time_s=1.3172
epoch=100 loss=0.0118 train_acc=1.0000 val_acc=0.7680 test_acc=0.8070 train_f1=1.0000 val_f1=0.7550 test_f1=0.8020 epoch_time_s=0.0030
epoch=200 loss=0.0092 train_acc=1.0000 val_acc=0.7580 test_acc=0.8080 train_f1=1.0000 val_f1=0.7475 test_f1=0.8021 epoch_time_s=0.0030
total_train_time_s=1.9282
train_loop_time=3.1576
```

Le GCN dépasse nettement le MLP sur Cora : le gain est d’environ +23 points d’accuracy et +24 points de Macro-F1 sur le jeu de test.
Le coût en temps augmente, mais reste modéré sur ce dataset : l’entraînement du GCN prend environ 2.5 fois plus de temps que celui du MLP.
Le compromis est donc très favorable ici : le graphe apporte une amélioration importante des performances pour un surcoût de calcul encore faible.

### 3.f — Pourquoi GCN peut dépasser (ou non) le MLP sur Cora ?

Sur Cora, le graphe apporte un signal utile car les nœuds connectés partagent souvent des labels proches : il existe donc une certaine homophilie.
Le GCN exploite cette structure en agrégeant les features des voisins, ce qui enrichit la représentation de chaque nœud au-delà de ses seules features tabulaires.
Le MLP, au contraire, traite chaque nœud indépendamment et ignore complètement les connexions du graphe.
Dans ce contexte, l’information relationnelle aide donc fortement la classification.
En revanche, si les features seules étaient déjà très discriminantes, ou si le graphe était peu informatif, le gain du GCN pourrait être plus limité.
Un autre risque est le sur-lissage : si trop de couches ou trop d’agrégation sont appliquées, les représentations des nœuds deviennent trop similaires.
Ici, avec un GCN simple à deux couches, le compromis est bon et le signal du graphe améliore clairement les performances.


## Exercice 4 — Modèle principal : GraphSAGE + neighbor sampling (mini-batch)

### 4.e — Entraînement de GraphSAGE avec sampling

Configuration utilisée :

```text
device: cuda
model: sage
epochs: 100
config: {'seed': 42, 'device': 'cuda', 'epochs': 100, 'lr': 0.01, 'weight_decay': 0.0005, 'sage': {'hidden_dim': 64, 'dropout': 0.5}, 'sampling': {'batch_size': 256, 'num_neighbors_l1': 10, 'num_neighbors_l2': 10}}
sampling: batch_size=256, num_neighbors=[10, 10]
```

Extrait du log d'entrainement : 

epoch=001 loss=1.9487 train_acc=0.9714 val_acc=0.6540 test_acc=0.6840 train_f1=0.9710 val_f1=0.6062 test_f1=0.6433 epoch_time_s=0.4396
epoch=010 loss=0.0195 train_acc=1.0000 val_acc=0.7580 test_acc=0.7940 train_f1=1.0000 val_f1=0.7505 test_f1=0.7896 epoch_time_s=0.0051
epoch=060 loss=0.0026 train_acc=1.0000 val_acc=0.7680 test_acc=0.8130 train_f1=1.0000 val_f1=0.7605 test_f1=0.8070 epoch_time_s=0.0045
epoch=100 loss=0.0050 train_acc=1.0000 val_acc=0.7580 test_acc=0.7990 train_f1=1.0000 val_f1=0.7482 test_f1=0.7929 epoch_time_s=0.0043
total_train_time_s=0.8889
train_loop_time=1.5559

Résultats finaux GraphSAGE :

Test Accuracy : 0.7990

Test Macro-F1 : 0.7929

Temps total d’entraînement : 0.8889 s

GraphSAGE obtient des performances très proches du GCN sur Cora, tout en gardant un temps d’entraînement plus faible que le GCN full-batch dans cette configuration.
Il dépasse très nettement le MLP, ce qui confirme que l’utilisation de la structure du graphe améliore fortement la classification des nœuds.
Dans ce TP, GraphSAGE apparaît donc comme un bon compromis entre performance et coût de calcul.

### 4.f — Compromis du neighbor sampling

Le neighbor sampling accélère l’entraînement car le modèle ne traite pas tout le graphe à chaque itération : il ne charge qu’un sous-graphe local autour d’un batch de nœuds cibles.
Le coût mémoire et le temps par itération deviennent donc mieux contrôlés, ce qui est particulièrement utile sur de grands graphes.
Le fanout (num_neighbors) limite explicitement le nombre de voisins explorés à chaque couche, ce qui évite l’explosion combinatoire du voisinage.
En revanche, cette approximation introduit un risque : le modèle ne voit qu’une partie du voisinage réel, donc le gradient devient plus bruité et plus variable d’un batch à l’autre.
Si le fanout est trop faible, on peut perdre de l’information utile, en particulier autour de nœuds très connectés (hubs).
Le sampling lui-même a aussi un coût CPU non nul, qui peut devenir visible si le graphe est très grand ou si les batches sont nombreux.
On gagne donc en scalabilité, mais au prix d’une estimation plus approximative de l’agrégation complète et parfois d’une légère baisse de performance.
Sur Cora, ce compromis reste très favorable : GraphSAGE conserve de bonnes performances tout en entraînant plus vite que le GCN full-batch.


## Exercice 5 — Benchmarks ingénieur : temps d’entraînement et latence d’inférence (CPU/GPU)

### 5.d — Benchmark d’inférence

Sorties du script `benchmark.py` :

```text
model: mlp
device: cuda
avg_forward_ms: 0.051
num_nodes: 2708
ms_per_node_approx: 1.884e-05

model: gcn
device: cuda
avg_forward_ms: 0.8307
num_nodes: 2708
ms_per_node_approx: 0.00030678

model: sage
device: cuda
avg_forward_ms: 0.3794
num_nodes: 2708
ms_per_node_approx: 0.0001401
```

Le modèle MLP est de loin le plus rapide en inférence car il n’exploite pas la structure du graphe et se limite à des opérations linéaires sur les features des nœuds.
Les modèles GNN doivent agréger l’information des voisins via les arêtes du graphe, ce qui augmente le coût de calcul.
Le GCN est le plus lent en inférence dans cette expérience, car il effectue une propagation sur l’ensemble du graphe à chaque couche.
GraphSAGE reste plus rapide que GCN tout en conservant des performances proches, ce qui s’explique par une agrégation plus simple et un modèle conçu pour être scalable.
On observe donc un compromis classique : les modèles exploitant le graphe offrent de meilleures performances mais au prix d’une latence d’inférence plus élevée.

### 5.e — Warmup et synchronisation CUDA

Le warmup est utilisé pour éliminer les effets de démarrage qui peuvent biaiser les mesures de performance. Lors des premières itérations sur GPU, certaines opérations comme l’allocation mémoire, la compilation de kernels CUDA ou la mise en cache peuvent introduire un surcoût temporaire. En exécutant quelques itérations avant la mesure réelle, on obtient des temps plus représentatifs du régime stable du modèle.

La synchronisation CUDA est également nécessaire car l’exécution GPU est asynchrone : lorsque Python lance une opération CUDA, celle-ci est placée dans une file d’exécution et le programme peut continuer sans attendre la fin du calcul. Si on mesure le temps sans synchroniser, on risque de mesurer uniquement le temps de lancement du kernel et non son exécution réelle. En utilisant torch.cuda.synchronize() avant et après la mesure, on force le CPU à attendre la fin des opérations GPU, ce qui permet d’obtenir des mesures fiables et comparables.

## Exercice 6 -- 

### 6.b — Tableau comparatif des modèles


| Modèle      | test_acc | test_macro_f1 | total_train_time_s | train_loop_time | avg_forward_ms |
|-------------|----------|---------------|--------------------|-----------------|----------------|
| MLP         | 0.5740   | 0.5619        | 0.5647             | 1.1284          | 0.051          |
| GCN         | 0.8080   | 0.8021        | 0.8013             | 1.4740          | 0.8307         |
| GraphSAGE   | 0.7990   | 0.7929        | 0.5267             | 0.9543          | 0.3794         |

### 6.c — Recommandation Ingénieur

Dans ce TP, les modèles exploitant la structure du graphe offrent un gain de performance important par rapport à la baseline tabulaire. Le MLP est le modèle le plus rapide en inférence (0.051 ms) et l’un des plus simples à entraîner, mais ses performances restent nettement inférieures (test_acc ≈ 0.57). Les modèles GNN exploitent la structure du graphe et améliorent fortement la qualité de prédiction. Le GCN obtient la meilleure performance (test_acc ≈ 0.81 et macro_f1 ≈ 0.80), mais son coût d’inférence est le plus élevé (≈ 0.83 ms). GraphSAGE offre un compromis intéressant : ses performances restent proches du GCN (test_acc ≈ 0.80) tout en ayant un coût d’entraînement et d’inférence plus faible (≈ 0.38 ms). En pratique, si la priorité est la performance maximale sur un graphe de taille modérée, le GCN peut être privilégié. Si l’on souhaite un modèle plus scalable et plus rapide à entraîner sur de grands graphes, GraphSAGE constitue généralement le meilleur choix. Le MLP peut rester pertinent lorsque la structure du graphe est absente ou lorsque la latence doit être extrêmement faible.

### 6.d — Risque de protocole expérimental 

Un risque important dans ce type de comparaison est l’absence de contrôle sur l’aléa expérimental. Les modèles peuvent produire des résultats légèrement différents selon la seed aléatoire utilisée pour l’initialisation des poids ou pour le sampling des voisins dans GraphSAGE. Si l’on ne fixe pas la seed, la comparaison entre modèles peut être biaisée. Dans ce TP, ce problème est partiellement contrôlé grâce à la fonction `set_seed`. Dans un vrai projet, on exécuterait chaque expérience plusieurs fois avec différentes seeds et on rapporterait la moyenne et l’écart-type des métriques. Un autre risque est la comparaison de temps d’inférence ou d’entraînement mesurés dans des conditions différentes (CPU vs GPU ou sans synchronisation CUDA). Pour obtenir des mesures fiables, il est nécessaire d’utiliser le même matériel, d’effectuer un warmup et de synchroniser CUDA avant et après les mesures.

### 6.e — Vérification du dépôt 

Le dépôt contient bien le dossier TP4/ avec le rapport (rapport.md), les scripts dans src/ et les fichiers de configuration dans configs/. Aucun dataset, checkpoint ou fichier volumineux (logs massifs ou données) n’a été commité dans le dépôt.
