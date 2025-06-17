import numpy as np
import random, torch
import torch.nn as nn
from AI_Hub.DQN import DQN
import torch.optim as optim
import torch.nn.functional as F
from AI_Hub.TatrizEnv import TetrizEnv
from threading import Semaphore, Thread

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
        return sum(self.rewards)

    def get_training_data(self):
        total_reward = self.compute_total_reward()
        X = torch.stack(self.states)
        y = torch.tensor(self.actions)
        rewards = torch.full((len(self.states),), total_reward, dtype=torch.float32)
        return X, y, rewards
    
env = TetrizEnv(janela=True)
model = DQN(1, env.action_space.n, env.jogo.tabuleiro.tabuleiro.shape)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
loss_fn = nn.CrossEntropyLoss()
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
            state = torch.tensor(state, dtype=torch.float32)
            ep.add(state.unsqueeze(0).unsqueeze(0), 2, reward)
        if done:
            mutex.acquire()
            eplist.append(ep)
            mutex.release()
            break
        i += 1

        if isinstance(state, np.ndarray):
            state = torch.tensor(state, dtype=torch.float32)

        state = state.unsqueeze(0).unsqueeze(0)

        epsilon = 0.1  # ou decaimento com o tempo
        if random.random() < epsilon:
            action = random.randint(0, env.action_space.n - 1)
        else:
            with torch.no_grad():
                logits = model(state)
                action = torch.argmax(logits).item()


        next_state, reward, done, _ = env.step(action, process_events=process_events)

        ep.add(state, action, reward)
        state = next_state

        if done:
            mutex.acquire()
            eplist.append(ep)
            mutex.release()
            break

def trainer(model, eplist):
    global epsodes
    ep: EpData
    for ep in eplist:
        epsodes += 1
        X, y, R = ep.get_training_data()
        # R = (R - R.mean()) / (R.std() + 1e-6)
        for x_i, y_i, r_i in zip(X, y, R):
            # x_i = x_i.unsqueeze(0)  # adiciona dimensão do batch: [1, C, H, W]
            # y_i = y_i.unsqueeze(0)  # também se precisar de batch p/ loss
            # logits = model(x_i)
            # loss = loss_fn(logits, y_i) * r_i  # r_i já é escalar

            logits = model(x_i) 
            probs = F.softmax(logits, dim=1) 
            log_probs = torch.log(probs + 1e-8)
            selected_log_prob = log_probs[0, y_i.item()]
            loss = -selected_log_prob * r_i

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        print(f"Episódio {epsodes}, Recompensa total: {R[0].item()}")

for episode in range(10000):
    try:
        eplist = []
        t: Thread
        n_threads = 0
        threads = []
        print("Jogando")
        for _ in range(n_threads):
            t = Thread(target=producer, args=[model, TetrizEnv(janela=False), eplist, False])
            t.start()
            threads.append(t)
        
        producer(model, env, eplist, True)

        for t in threads:
            t.join()
        print("Treinando")
        trainer(model, eplist)
    except KeyboardInterrupt:
        stop_threads = True
        for t in threads:
            t.join()
        break

    # ep = EpData()
    # state = env.reset(process_events=True)

    # i = 0

    # while True:
    #     if i % 8 == 0:
    #         state, reward, done, _ = env.step(2, process_events=True)
    #         state = torch.tensor(state, dtype=torch.float32)
    #         ep.add(state.unsqueeze(0).unsqueeze(0), 2, reward)
    #     if done:
    #         break
    #     i += 1

    #     if isinstance(state, np.ndarray):
    #         state = torch.tensor(state, dtype=torch.float32)

    #     # if state.ndim == 2:  # [H, W]
    #     state = state.unsqueeze(0).unsqueeze(0)  # -> [1, 1, H, W]
    #     # elif state.ndim == 3:  # [C, H, W]
    #     #     state = state.unsqueeze(0)  # -> [1, C, H, W]

    #     with torch.no_grad():
    #         logits = model(state)
    #         action = torch.argmax(logits).item()

    #     next_state, reward, done, _ = env.step(action, process_events=True)

    #     ep.add(state, action, reward)
    #     state = next_state

    #     if done:
    #         break

    # X, y, R = ep.get_training_data()
    # for x_i, y_i, r_i in zip(X, y, R):
    #     # x_i = x_i.unsqueeze(0)  # adiciona dimensão do batch: [1, C, H, W]
    #     y_i = y_i.unsqueeze(0)  # também se precisar de batch p/ loss
    #     logits = model(x_i)
    #     loss = loss_fn(logits, y_i) * r_i  # r_i já é escalar

    #     optimizer.zero_grad()
    #     loss.backward()
    #     optimizer.step()