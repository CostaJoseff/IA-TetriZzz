import os
import numpy as np
import random, torch
import torch.nn as nn
from AI_Hub.DQN import DQN
import torch.optim as optim
import torch.nn.functional as F
from Jogo2.TetriZzz import TetriZzz
from threading import Semaphore, Thread

os.system('cls' if os.name == 'nt' else 'clear')

# Define o dispositivo
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"🖥️ Usando dispositivo: {device}")

class EpData:
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.states = []
        self.next_states = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.limite_de_futuro = 25

    def add(self, state, action, reward, next_state, dones):
        self.states.append(state)
        self.actions.append(action)

        r: list
        for r in self.rewards:
            if dones and len(r) < self.limite_de_futuro:
                r.append(reward)
                while len(r) < self.limite_de_futuro:
                    r.append(0)
            elif len(r) < self.limite_de_futuro:
                r.append(reward)


        if not dones:
            self.rewards.append([reward])
        else:
            self.rewards.append([reward]*self.limite_de_futuro)

        # self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(dones)

    def compute_total_reward(self):
        return [sum(r[1:]) for r in self.rewards]
    
    def get_immediate_rewards(self):
        return [r[0] for r in self.rewards]

    def get_training_data(self):
        total_reward = self.compute_total_reward()
        total_reward = torch.tensor(total_reward, dtype=torch.float32, device=device)
        X = torch.stack(self.states)
        y = torch.tensor(self.actions)
        Z = torch.stack(self.next_states)
        D = torch.tensor(self.dones)
        rewards = torch.tensor(self.get_immediate_rewards(), dtype=torch.float32, device=device)
        return X, y, rewards, Z, D, total_reward

# Inicializa o ambiente e o modelo
env: TetriZzz = TetriZzz(janela=True)
model = DQN(1, env.action_space.n, env.estado_atual.shape).to(device)

MODEL_PATH = 'modelo.pth'
if os.path.exists(MODEL_PATH):
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device), strict=False)
    print('✅ Modelo carregado com sucesso.')
else:
    print('📦 Modelo não encontrado. Treinando e salvando...')

optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)# RMSprop(model.parameters(), lr=1e-4, alpha=0.95, eps=1e-5)
loss_fn = nn.SmoothL1Loss()
global mutex, stop_threads, epsodes, epsilon
epsilon = None
epsodes = 0
stop_threads = False
mutex = Semaphore(1)

def producer(model: DQN, env: TetriZzz, eplist, process_events):
    global mutex, stop_threads
    model.eval()
    ep = EpData()
    state = env.reset(process_events=process_events)
    state = torch.tensor(state, dtype=torch.float32, device=device)
    state = state.unsqueeze(0)
    done = False
    global epsilon
    # epsilon = .45
    if random.random() < .1:
        epsilon = .45
    elif random.random() < .4:
        epsilon = .75
    elif random.random() < .5:
        epsilon = .1
    else:
        epsilon = .05
    if process_events:
        print(f"🎮 Jogando | Epsilon: {epsilon}")
        env.engine.random_background = 0 if epsilon == .05 else 100 if epsilon == .45 else 200
        env.redesenhar_tudo()
        

    i = 0
    while not stop_threads:
        if i % 5 == 0:
            next_state, reward, done, _ = env.step(2, process_events=process_events)
            next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
            next_state = next_state.unsqueeze(0)
            ep.add(state, 2, reward, next_state, done)
            state = next_state

        if done:
            mutex.acquire()
            eplist.append(ep)
            mutex.release()
            break

        i += 1

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32, device=device)
            state = state.unsqueeze(0)

        # epsilon = 1 - (0.001 * epsodes)
        # epsilon = 0 if epsilon <= 0 else epsilon

        if random.random() < epsilon:
            if epsilon <= .1 and random.random() < .7:
                action = random.choice([0, 4])
            else:
                action = random.randint(0, env.action_space.n - 1)
        else:
            with torch.no_grad():
                logits: torch.Tensor = model(state.unsqueeze(0))
                if process_events:
                    print(f"\r{logits.tolist()}", end=f"  {i}                     ")
                action = torch.argmax(logits).item()

        next_state, reward, done, _ = env.step(action, process_events=process_events)

        state_tensor = state
        next_state = torch.tensor(next_state, dtype=torch.float32, device=device)
        next_state = next_state.unsqueeze(0)
        ep.add(state_tensor, action, reward, next_state, done)
        state = next_state

        if done:
            mutex.acquire()
            eplist.append(ep)
            mutex.release()
            break


def trainer(model: DQN, eplist):
    model.train()
    global epsodes
    gamma = 0.99
    epsodes += 1
    N = 1

    for _ in range(N):
        local_ep = 1
        all_losses = []
        optimizer.zero_grad()
        ep: EpData
        for ep in eplist:
            print(f"\r🧠 Treinando | ep {episode} ex {local_ep}", end="")
            X, y, R, Z, D, TR = ep.get_training_data()

            X = X.float().to(model.device)
            Z = Z.float().to(model.device)
            y = torch.tensor(y, dtype=torch.long).to(model.device)
            R = torch.tensor(R, dtype=torch.float32).to(model.device)
            D = torch.tensor(D, dtype=torch.bool).to(model.device)
            TR = torch.tensor(TR, dtype=torch.bool).to(model.device)

            q_pred = model(X)
            q_pred_selected = q_pred.gather(1, y.unsqueeze(1)).squeeze(1)

            # try:
            with torch.no_grad():
                # model = model.eval()
                # q_next = model(Z)
                # model = model.train()
                # max_q_next = q_next.max(dim=1)[0]
                target = torch.where(D, R, R + gamma * TR)
                noise = torch.randn_like(target) * 0.05
                target = torch.clamp(target + noise, -1.0, 1.0)
            # except:
                # with torch.no_grad():
                #     model.eval()
                #     q_next = model(Z)
                #     model.train()
                #     max_q_next = q_next.max(dim=1)[0]
                #     target = torch.where(D, R, R + gamma * max_q_next)
                #     noise = torch.randn_like(target) * 0.05
                #     target = torch.clamp(target + noise, -1.0, 1.0)

            loss = loss_fn(q_pred_selected, target)
            all_losses.append(loss.item())
            loss.backward()
            local_ep += 1

        optimizer.step()
        print(f" ->   Episódio {epsodes}, Loss médio: {sum(all_losses)/len(all_losses):.4f}")

eplist = []
eplist_limit = 1000
for episode in range(1000000):
    try:
        while len(eplist) > eplist_limit:
            eplist.pop(0)

        threads = []
        n_threads = 100

        for _ in range(n_threads):
            t = Thread(target=producer, args=[model, TetriZzz(janela=False), eplist, False])
            t.start()
            threads.append(t)

        producer(model, env, eplist, True)

        i = 0
        print()
        print(f"\r⌛ Aguardando threads finalizarem: {i} de {len(threads)}", end="")
        for t in threads:
            t.join()
            i += 1
            print(f"\r⌛ Aguardando threads finalizarem: {i} de {len(threads)}", end="")
        print()
        
        trainer(model, eplist)

        torch.save(model.state_dict(), MODEL_PATH)
    except KeyboardInterrupt:
        stop_threads = True
        mutex.release()
        for t in threads:
            t.join()
        break
