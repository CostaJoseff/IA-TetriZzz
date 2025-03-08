from AI_Hub.valores import batch_size, device, next_reward_focus, learning_rate
from threading import Thread, Semaphore
from AI_Hub.TatrizEnv import TetrizEnv
from AI_Hub.Memory import Memory_
import matplotlib.pyplot as plt
import torch, random, copy
import seaborn as sns
import numpy as np

class AI_Hub():

    def __init__(self, modelos: list, otimizadores: list, ambientes: list, funcoes_de_perda: list, displays: list[bool], args: dict = {}):
        assert len(modelos) == len(otimizadores) and len(modelos) == len(ambientes) and len(modelos) == len(funcoes_de_perda)
        print("Instanciando AI_Hub")
        self.modelos: dict = {}

        i = 0
        for modelo, otimizador, ambiente, func_de_perda, display in zip(modelos, otimizadores, ambientes, funcoes_de_perda, displays):
            self.modelos[f"modelo{i}"] = {
                "modelo": modelo,
                "otimizador": otimizador(modelo.parameters(), lr=learning_rate),
                "ambiente": ambiente,
                "funcao_de_perda": func_de_perda,
                "display": display
            }
            i += 1

        self.mutex = Semaphore(1)

        self.memoria: Memory_ = Memory_()
        self.new_line_log = args.get("new_line_log", False)
        self.user_input = args.get("user_input", False)
        self.total_epsodes = args.get("total_epsodes", 10000)
        self.flag = not self.user_input

        self.pesos_anteriores = None
        self.epsode = 0
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.99
        self.maior_score = 0
        self.maior_reward = -float("inf")
        self.melhor_score_atual = 0

        self.stop_threads = False

    def start(self):
        try:
            print("\nIniciando AI_Hub")
            threads: list[Thread] = []

            while self.epsode <= self.total_epsodes:
                disp_model = None
                for nome, modelo in self.modelos.items():
                    if not modelo["display"]:
                        thread = Thread(target=self.game_loop, args=[modelo["modelo"], modelo["ambiente"], modelo["display"]])
                        thread.start()
                        threads.append(thread)

                    else:
                        disp_model = modelo

                self.game_loop(disp_model["modelo"], disp_model["ambiente"], disp_model["display"])

                for thread in threads:
                    thread.join()
                
                self.train_all(disp_model)
                self.update_plot(disp_model["modelo"])
                self.log()

        except KeyboardInterrupt:
            print("\n\nFinalizando threads\n")
            self.stop_threads = True
            for thread in threads:
                thread.join()
            exit(0)

    def game_loop(self, model, env: TetrizEnv, process_events: bool):
        state = env.reset(process_events)
        self.melhor_score_atual = 0
        while not self.memoria.is_full() and not self.stop_threads:
            if env.jogo.perdeu:
                state = env.reset(process_events)

            if random.random() < self.epsilon:
                action = env.action_space.sample()
            else:
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(device)
                with torch.no_grad():
                    action = torch.argmax(model(state_tensor)).item()

            next_state, reward, done, _ = env.step(action, process_events)
            env.jogo.tabuleiro.pontos += reward
            # print(reward)
            self.memoria.append((state, action, reward, next_state, done))
            state = next_state
            self.mutex.acquire()
            if self.melhor_score_atual < reward:
                self.melhor_score_atual = reward
            if self.maior_reward < reward:
                self.maior_reward = reward
            self.mutex.release()

            self.log()

        self.mutex.acquire()
        if self.maior_score < env.jogo.tabuleiro.pontos:
            self.maior_score = env.jogo.tabuleiro.pontos
        self.mutex.release()

    def train_all(self, modelo_base):
        if self.memoria.len_() < batch_size:
            print("Não treinou\n"*50) # Esse print nunca deve acontecer
            return
        
        batch = self.memoria.sample()
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(actions, dtype=torch.long).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        dones = torch.tensor(dones, dtype=torch.float32).to(device)

        for i in range(10):
            q_values = modelo_base["modelo"](states).gather(1, actions.unsqueeze(1)).squeeze()
            next_q_values = modelo_base["modelo"](next_states).max(1)[0].detach()
            targets = rewards + next_reward_focus * next_q_values * (1 - dones)
            loss = modelo_base["funcao_de_perda"](q_values, targets)#, reduction="sum")
            modelo_base["otimizador"].zero_grad()
            loss.backward()
            modelo_base["otimizador"].step()

        for nome, _ in self.modelos.items():
            self.modelos[nome]["modelo"].load_state_dict(modelo_base["modelo"].state_dict())
            # if modelo["display"]:
            #     self.update_plot(modelo["modelo"])

        self.epsode += 1
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def update_plot(self, model, layer="fc1.weight"):
        weights = model.state_dict()[layer].cpu().numpy()
        if self.pesos_anteriores is not None:
            heat_map = weights - self.pesos_anteriores
            self.pesos_anteriores = weights.copy()
            sns.heatmap(heat_map, annot=False, cmap="coolwarm", fmt=".4f", cbar=False, center=0)

        else:
            sns.heatmap(weights, annot=False, cmap="magma", fmt=".4f", cbar=False)
            self.pesos_anteriores = weights.copy()

        plt.title(f"Distribuição de Pesos - {layer}")
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.pause(0.1)

    def log(self):
        eps = f"\033[33mEp: \033[36m{self.epsode} "
        eps += " " * max(20 - len(eps), 0)

        scr = f"\033[33mScore: \033[32m{self.melhor_score_atual} "
        scr += " " * max(25 - len(scr), 0)

        bt_sz = f"\033[33mBatch size: \033[36m{batch_size} "
        bt_sz += " " * max(30 - len(bt_sz), 0)

        mem = f"\033[33mMemory: \033[35m{self.memoria} "
        mem += " " * max(47 - len(mem), 0)

        mem_len = f"\033[33mMemory len: \033[36m{self.memoria.len_()} "
        mem_len += " " * max(30 - len(mem_len), 0)

        btt_rwd = f"\033[33mBetter reward: \033[32m{self.maior_reward} "
        btt_rwd += " " * max(30 - len(btt_rwd), 0)

        btt_scr = f"\033[33mBetter score: \033[32m{self.maior_score} "
        btt_scr += " " * max(30 - len(btt_scr), 0)

        epsl = f"\033[33mEpsilon: \033[31m{round(self.epsilon, 3)} "
        epsl += " " * max(30 - len(epsl), 0)

        print("\r"+eps+scr+bt_sz+mem+mem_len+btt_rwd+btt_scr+epsl+f"\033[0m",end="   " if not self.new_line_log else "\n")