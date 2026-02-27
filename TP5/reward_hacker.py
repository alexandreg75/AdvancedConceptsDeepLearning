import gymnasium as gym
from stable_baselines3 import PPO
from PIL import Image

# Définition de notre Wrapper pour altérer la réalité de l'agent pendant l'entraînement
class FuelPenaltyWrapper(gym.Wrapper):
    """Un wrapper qui modifie la récompense renvoyée par l'environnement selon l'action choisie."""

    def step(self, action):
        # On récupère le résultat normal de l'environnement parent
        obs, reward, terminated, truncated, info = self.env.step(action)

        # Modification arbitraire : on taxe lourdement le moteur principal (action=2)
        if action == 2:
            reward -= 50.0

        return obs, reward, terminated, truncated, info


print("--- ENTRAÎNEMENT DE L'AGENT RADIN ---")
# 1. Création de l'environnement normal
base_env = gym.make("LunarLander-v3")

# 2. Application de notre Wrapper par-dessus l'environnement
train_env = FuelPenaltyWrapper(base_env)

# 3. Entraînement (150 000 steps suffisent pour voir ce comportement dégénéré)
# On force l'utilisation du CPU, plus efficace pour de si petits réseaux
model_cheap = PPO("MlpPolicy", train_env, verbose=1, device="cpu")
model_cheap.learn(total_timesteps=150000)
train_env.close()
print("Entraînement terminé.")

print("\n--- ÉVALUATION ET TÉLÉMÉTRIE ---")
# On évalue sur l'environnement NORMAL pour voir son vrai score
eval_env = gym.make("LunarLander-v3", render_mode="rgb_array")
obs, info = eval_env.reset()
done = False
frames = []

total_reward = 0.0
main_engine_uses = 0
side_engine_uses = 0

while not done:
    action, _states = model_cheap.predict(obs, deterministic=True)
    obs, reward, terminated, truncated, info = eval_env.step(action)

    # Mise à jour des métriques
    total_reward += reward
    if action == 2:
        main_engine_uses += 1
    elif action in [1, 3]:
        side_engine_uses += 1

    frames.append(Image.fromarray(eval_env.render()))
    done = terminated or truncated

eval_env.close()

# Analyse du vol
if reward == -100:
    issue = "CRASH DÉTECTÉ 💥"
elif reward == 100:
    issue = "ATTERRISSAGE RÉUSSI 🏆"
else:
    issue = "TEMPS ÉCOULÉ OU SORTIE DE ZONE ⚠️"

print("\n--- RAPPORT DE VOL PPO HACKED ---")
print(f"Issue du vol : {issue}")
print(f"Récompense totale cumulée : {total_reward:.2f} points")
print(f"Allumages moteur principal : {main_engine_uses}")
print(f"Allumages moteurs latéraux : {side_engine_uses}")
print(f"Durée du vol : {len(frames)} frames")

if frames:
    frames[0].save('hacked_agent.gif', save_all=True, append_images=frames[1:], duration=30, loop=0)
    print("Vidéo du nouvel agent sauvegardée sous 'hacked_agent.gif'")
