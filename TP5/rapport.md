## Exercice 1 — Exploration de Gymnasium

### Visualisation de l’agent aléatoire

![Agent aléatoire sur LunarLander](./random_agent.gif)

---

### Espaces de l’environnement

Espace d’observation (capteurs) :  
Box(..., shape=(8,), float32)

L’environnement fournit donc un vecteur de **8 variables continues** :
- position horizontale  
- position verticale  
- vitesse horizontale  
- vitesse verticale  
- angle du module  
- vitesse angulaire  
- contact jambe gauche  
- contact jambe droite  

Espace d’action (moteurs) :  
Discrete(4)

Les actions possibles sont :
- 0 : Ne rien faire  
- 1 : Moteur latéral gauche  
- 2 : Moteur principal  
- 3 : Moteur latéral droit  

---

### Rapport de vol

```text
--- RAPPORT DE VOL ---
Issue du vol : CRASH DÉTECTÉ 💥
Récompense totale cumulée : -128.62 points
Allumages moteur principal : 23
Allumages moteurs latéraux : 64
Durée du vol : 116 frames
Vidéo de la télémétrie sauvegardée sous 'random_agent.gif'
```

---

## Exercice 2 — Entraînement et évaluation PPO (Stable Baselines3)

### Évolution de `ep_rew_mean` pendant l’entraînement

Pendant l’apprentissage PPO, la métrique `ep_rew_mean` (récompense moyenne par épisode) a nettement augmenté.

- Au début (vers `total_timesteps = 2048`), on observe `ep_rew_mean ≈ -194`.
- À la fin de l’entraînement (vers `total_timesteps ≈ 500000`), `ep_rew_mean` est monté autour de **~100** (ex: `ep_rew_mean ≈ 102`).

Cette hausse indique que l’agent apprend progressivement une stratégie de pilotage plus efficace qu’un agent aléatoire, même si la performance finale n’est pas suffisante pour “résoudre” l’environnement.

---

### Visualisation de l’agent PPO entraîné

![Agent PPO entraîné sur LunarLander](./trained_ppo_agent.gif)

---

### Rapport de vol PPO

```text
--- RAPPORT DE VOL PPO ---
Issue du vol : CRASH DÉTECTÉ 💥
Récompense totale cumulée : -11.21 points
Allumages moteur principal : 147
Allumages moteurs latéraux : 138
Durée du vol : 285 frames
Vidéo de la télémétrie sauvegardée sous 'trained_ppo_agent.gif'
```

---

## Exercice 3 — Reward Engineering (Wrappers et Hacking)

### 3.a — Wrapper de pénalité carburant

On a créé un wrapper Gymnasium qui intercepte `step(action)` et modifie la récompense si l’agent utilise le moteur principal (action `2`).

Récompense modifiée :

```text
r' = r - 50 si action == 2
```

### 3.b — Exécution, télémétrie et stratégie observée

#### Sortie terminal

```text
python3 reward_hacker.py 
--- ENTRAÎNEMENT DE L'AGENT RADIN ---
Using cpu device
Wrapping the env with a `Monitor` wrapper
Wrapping the env in a DummyVecEnv.
...
Entraînement terminé.

--- ÉVALUATION ET TÉLÉMÉTRIE ---

--- RAPPORT DE VOL PPO HACKED ---
Issue du vol : CRASH DÉTECTÉ 💥
Récompense totale cumulée : -104.40 points
Allumages moteur principal : 0
Allumages moteurs latéraux : 10
Durée du vol : 79 frames
Vidéo du nouvel agent sauvegardée sous 'hacked_agent.gif'
```

### Interprétation en termes de MDP

En modifiant la récompense via le wrapper, nous avons implicitement modifié la fonction de récompense du MDP sous-jacent.  
L’agent n’interagit plus avec l’environnement original, mais avec un nouveau MDP dont la récompense pénalise fortement l’action 2.

Le comportement appris est donc parfaitement cohérent avec le nouvel objectif optimisé.  
Le problème ne vient pas de l’algorithme PPO, mais de la mauvaise spécification de la fonction de récompense.

Ce phénomène illustre un principe central en Reinforcement Learning :

> Un agent optimise exactement ce qu’on lui demande, pas ce qu’on voulait dire.

---

## Exercice 4 — Généralisation OOD

### Sortie terminal

```text
python3 ood_agent.py
--- ÉVALUATION OOD : GRAVITÉ FAIBLE ---

--- RAPPORT DE VOL PPO (GRAVITÉ MODIFIÉE) ---
Issue du vol : CRASH DÉTECTÉ 💥
Récompense totale cumulée : -53.58 points
Allumages moteur principal : 25
Allumages moteurs latéraux : 191
Durée du vol : 220 frames
Vidéo de la télémétrie sauvegardée sous 'ood_agent.gif'
```

### Observation du GIF

Dans la gravité faible, le vaisseau adopte une trajectoire diagonale vers la droite et finit par s’écraser très loin de la zone centrale.

On observe :

- Une dérive latérale continue vers la droite  
- De très nombreuses corrections latérales (191 activations)  
- Une stabilisation verticale lente  
- Une absence de recentrage vers la zone d’atterrissage centrale  

L’agent applique une stratégie adaptée à la gravité d’entraînement (-10.0), mais inadaptée à la nouvelle dynamique (-2.0).

Ce phénomène illustre un problème de généralisation Out-of-Distribution :  
la politique apprise est spécialisée pour la distribution d’entraînement et ne s’adapte pas aux nouvelles conditions physiques.

---

## Exercice 5 — Bilan Ingénieur : Sim-to-Real

### 5.a — Rendre l’agent robuste sans entraîner un modèle par lune

L’échec en gravité modifiée montre que l’agent a surappris la physique d’entraînement.

Deux stratégies pour améliorer la robustesse :

**1) Domain Randomization**

Entraîner l’agent avec :
- Gravité aléatoire à chaque épisode  
- Vent et turbulence variables  
- Paramètres physiques légèrement perturbés  

L’agent apprend ainsi une stratégie robuste valable sur plusieurs environnements.

**2) Ajouter les paramètres physiques dans l’observation**

Inclure la gravité ou le vent dans le vecteur d’état.  
L’agent peut alors adapter son comportement aux conditions rencontrées.

### Conclusion

Le Sim-to-Real Gap apparaît lorsque la dynamique réelle diffère de celle utilisée en simulation.  
En diversifiant les conditions d’entraînement et en rendant les paramètres physiques observables, on améliore la robustesse du modèle sans changer d’algorithme ni multiplier les modèles.