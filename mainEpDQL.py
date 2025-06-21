import os
import numpy as np
import random, torch
import torch.nn as nn
from AI_Hub.DQN import DQN
import torch.optim as optim
import torch.nn.functional as F
from AI_Hub.TatrizEnv import TetrizEnv
from threading import Semaphore, Thread

os.system('cls' if os.name == 'nt' else 'clear')

# Define o dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Usando dispositivo: {device}")

class EpData:
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []

    def add(self, state, action, reward):
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def compute_total_reward(self):
        return self.rewards

    def get_training_data(self):
        total_reward = self.compute_total_reward()
        X = torch.stack(self.states)
        y = torch.tensor(self.actions)
        rewards = torch.tensor(total_reward, dtype=torch.float32, device=device)
        return X, y, rewards

# Inicializa o ambiente e o modelo
env = TetrizEnv(janela=True)
model = DQN(1, env.action_space.n, env.jogo.tabuleiro.tabuleiro.shape).to(device)
MODEL_PATH = 'modelo.pth'
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.eval()
    print('✅ Modelo carregado com sucesso.')
else:
    print('📦 Modelo não encontrado. Treinando e salvando...')

optimizer = optim.RMSprop(model.parameters(), lr=1e-4, momentum=0.99)
# loss_fn = nn.CrossEntropyLoss()
global mutex, stop_threads, epsodes
epsodes = 0
stop_threads = False
mutex = Semaphore(1)

def producer(model, env, eplist, process_events):
    global mutex, stop_threads
    ep = EpData()
    state = env.reset(process_events=process_events)

    i = 0
    while not stop_threads:
        if i % 8 == 0:
            state, reward, done, _ = env.step(2, process_events=process_events)
            state = torch.tensor(state, dtype=torch.float32, device=device)
            ep.add(state.unsqueeze(0).unsqueeze(0), 2, reward)

        if done:
            mutex.acquire()
            eplist.append(ep)
            mutex.release()
            break

        i += 1

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=device)

        state = state.unsqueeze(0).unsqueeze(0)

        epsilon = 1 - (0.01 * epsodes)
        epsilon = 0 if epsilon < 0 else epsilon
        if random.random() < epsilon:
            action = random.randint(0, env.action_space.n - 1)
        else:
            with torch.no_grad():
                logits = model(state)
                action = torch.argmax(logits).item()

        next_state, reward, done, _ = env.step(action, process_events=process_events)

        state_tensor = torch.tensor(state.cpu().numpy(), dtype=torch.float32, device=device)
        ep.add(state_tensor, action, reward)
        state = next_state

        if done:
            mutex.acquire()
            eplist.append(ep)
            mutex.release()
            break

# Recompensas com desconto (reward-to-go)
def compute_discounted_rewards(rewards, gamma=0.99):
    discounted = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted.insert(0, R)
    return torch.tensor(discounted)


def trainer(model, eplist):
    global epsodes
    ep: EpData
    for ep in eplist:
        losses = []
        epsodes += 1
        X, y, R = ep.get_training_data()
        # R = (R - R.mean()) / (R.std() + 1e-6)
        discounted_R  = compute_discounted_rewards(R)
        baseline = discounted_R.mean()

        optimizer.zero_grad()
        for x_i, y_i, r_i in zip(X, y, R):
            r_i = r_i * 10 if r_i > 0 else r_i
            logits = model(x_i) 
            probs = F.softmax(logits, dim=1) 
            print(f"\r{probs.squeeze().tolist()}", end="    "*10)
            log_probs = torch.log(probs + 1e-8)
            action = y_i.item() if y_i.dim() > 0 else y_i
            selected_log_prob = log_probs[0, action]
            entropy = -(probs * log_probs).sum()
            loss = -(selected_log_prob) * (r_i - baseline) - 0.01 * entropy
            losses.append(loss.item())
            loss.backward()
        print()
        optimizer.step()
        
        print(f"Episódio {epsodes}, Recompensa total: {torch.sum(R).item()}, Losses: {np.mean(losses)}")

for episode in range(1000000):
    try:
        eplist = []
        threads = []
        n_threads = 0

        print(f"🎮 Jogando")
        for _ in range(n_threads):
            t = Thread(target=producer, args=[model, TetrizEnv(janela=False), eplist, False])
            t.start()
            threads.append(t)

        producer(model, env, eplist, True)

        for t in threads:
            t.join()

        print("🧠 Treinando")
        trainer(model, eplist)

        torch.save(model.state_dict(), MODEL_PATH)
    except KeyboardInterrupt:
        stop_threads = True
        for t in threads:
            t.join()
        break
