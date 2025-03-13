from AI_Hub.valores import batch_size, device, next_reward_focus, learning_rate, memory_size
from threading import Thread, Semaphore
from AI_Hub.TatrizEnv import TetrizEnv
from AI_Hub.Memory import Memory_
import matplotlib.pyplot as plt
import torch, random, copy
import seaborn as sns
import numpy as np

cores = {
    "Preto": {"letra": "\033[1;30m", "fundo": "\033[1;40m"},

    "Vermelho": {"letra": "\033[1;31m", "fundo": "\033[1;41m"},
    "Vermelho Claro": {"letra": "\033[1;91m", "fundo": "\033[1;101m"},

    "Verde": {"letra": "\033[1;32m", "fundo": "\033[1;42m"},
    "Verde Claro": {"letra": "\033[1;92m", "fundo": "\033[1;102m"},

    "Amarelo": {"letra": "\033[1;33m", "fundo": "\033[1;43m"},
    "Amarelo Claro": {"letra": "\033[1;93m", "fundo": "\033[1;103m"},

    "Azul": {"letra": "\033[1;34m", "fundo": "\033[1;44m"},
    "Azul Claro": {"letra": "\033[1;94m", "fundo": "\033[1;104m"},

    "Magenta": {"letra": "\033[1;35m", "fundo": "\033[1;45m"},
    "Magenta Claro": {"letra": "\033[1;95m", "fundo": "\033[1;105m"},

    "Cyan": {"letra": "\033[1;36m", "fundo": "\033[1;46m"},
    "Cyan Claro": {"letra": "\033[1;96m", "fundo": "\033[1;106m"},

    "Branco": {"letra": "", "fundo": "\033[1;107m"},
    "Cinza Claro": {"letra": "\033[1;37m", "fundo": "\033[1;47m"},
    "Cinza Escuro": {"letra": "\033[1;90m", "fundo": "\033[1;100m"},
    
    "Laranja": {"letra": "\033[38;5;214m", "fundo": "\033[48;5;214m"},

    "Amarelo-Verde": {"letra": "\033[38;5;154m", "fundo": "\033[48;5;154m"},
}

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
            if self.maior_reward < reward:
                self.maior_reward = reward
            if self.maior_score < env.jogo.tabuleiro.pontos:
                self.maior_score = env.jogo.tabuleiro.pontos
            self.log()
            self.mutex.release()

    def train_all(self, modelo_base):
        while not self.memoria.is_empty():
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

            for _ in range(5):
                q_values = modelo_base["modelo"](states).gather(1, actions.unsqueeze(1)).squeeze()
                next_q_values = modelo_base["modelo"](next_states).max(1)[0].detach()
                targets = rewards + next_reward_focus * next_q_values * (1 - dones)
                loss = modelo_base["funcao_de_perda"](q_values, targets)#, reduction="sum")
                modelo_base["otimizador"].zero_grad()
                loss.backward()
                modelo_base["otimizador"].step()
                self.train_log(loss.cpu().detach().numpy())

            del batch, states, actions, rewards, next_states, dones, q_values, next_q_values, targets

            for nome, _ in self.modelos.items():
                self.modelos[nome]["modelo"].load_state_dict(modelo_base["modelo"].state_dict())

            self.epsode += 1
            self.train_log(loss.cpu().detach().numpy())
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
        torch.cuda.empty_cache()
        torch.save(modelo_base["modelo"].state_dict(), 'TetrizDQN_model.pth')

    def update_plot(self, model, layer="fc.weight"):
        weights = model.state_dict()[layer].cpu().numpy()
        if self.pesos_anteriores is not None:
            heat_map = weights - self.pesos_anteriores
            self.pesos_anteriores = weights.copy()
            sns.heatmap(heat_map, annot=False, cmap="bwr", fmt=".4f", cbar=False, center=0)

        else:
            sns.heatmap(weights, annot=False, cmap="magma", fmt=".4f", cbar=False)
            self.pesos_anteriores = weights.copy()

        plt.title(f"Distribuição de Pesos - {layer}")
        plt.xticks([])
        plt.yticks([])
        plt.show()
        plt.pause(0.1)

    def train_log(self, loss):
        memoria_len = self.memoria.len_()
        cor_condicional = cores['Vermelho Claro']['letra'] if memoria_len <= (memory_size*1/5) else cores["Laranja"]['letra'] if memoria_len <= (memory_size*2/5) else cores["Amarelo Claro"]['letra'] if memoria_len <= (memory_size*3/5) else cores["Amarelo-Verde"]['letra'] if memoria_len <= (memory_size*4/5) else cores["Verde Claro"]['letra']
        eps = f"{cor_condicional}Ep: {cor_condicional}{self.epsode} "
        eps += " " * max(20 - len(eps), 0)

        bt_sz = f"{cor_condicional}Batch size: {cor_condicional}{batch_size} "
        bt_sz += " " * max(30 - len(bt_sz), 0)

        mem_len = f"{cor_condicional}Memory len: {cor_condicional}{memoria_len} "
        mem_len += " " * max(30 - len(mem_len), 0)

        mem = f"{cor_condicional}Memory: {cor_condicional}{self.memoria} "
        mem += " " * max(47 - len(mem), 0)

        btt_rwd = f"{cor_condicional}Better reward: {cor_condicional}{self.maior_reward} "
        btt_rwd += " " * max(30 - len(btt_rwd), 0)

        btt_scr = f"{cor_condicional}Better score: {cor_condicional}{self.maior_score} "
        btt_scr += " " * max(30 - len(btt_scr), 0)

        epsl = f"{cor_condicional}Epsilon: {cor_condicional}{round(self.epsilon, 3)} "
        epsl += " " * max(30 - len(epsl), 0)

        los = f"{cores['Cinza Claro']['fundo']}{cores['Vermelho Claro']['letra']}Loss: {loss}{cores['Preto']['fundo']}"

        print("\r"+eps+bt_sz+mem+mem_len+btt_rwd+btt_scr+epsl+los, end="   " if not self.new_line_log else "\n")

    def log(self):
        eps = f"{cores["Cinza Claro"]["letra"]}Ep: {cores['Azul Claro']['letra']}{self.epsode} "
        eps += " " * max(20 - len(eps), 0)

        bt_sz = f"{cores["Cinza Claro"]["letra"]}Batch size: {cores['Azul Claro']['letra']}{batch_size} "
        bt_sz += " " * max(30 - len(bt_sz), 0)

        memoria_len = self.memoria.len_()
        cor_condicional = cores['Vermelho Claro']['letra'] if memoria_len <= (memory_size*1/5) else cores["Laranja"]['letra'] if memoria_len <= (memory_size*2/5) else cores["Amarelo Claro"]['letra'] if memoria_len <= (memory_size*3/5) else cores["Amarelo-Verde"]['letra'] if memoria_len <= (memory_size*4/5) else cores["Verde Claro"]['letra']
        mem_len = f"{cores["Cinza Claro"]["letra"]}Memory len: {cor_condicional}{memoria_len} "
        mem_len += " " * max(30 - len(mem_len), 0)

        mem = f"{cores["Cinza Claro"]["letra"]}Memory: {cor_condicional}{self.memoria} "
        mem += " " * max(47 - len(mem), 0)

        btt_rwd = f"{cores["Cinza Claro"]["letra"]}Better reward: {cores['Verde Claro']['letra']}{self.maior_reward} "
        btt_rwd += " " * max(30 - len(btt_rwd), 0)

        btt_scr = f"{cores["Cinza Claro"]["letra"]}Better score: {cores['Verde']['letra']}{self.maior_score} "
        btt_scr += " " * max(30 - len(btt_scr), 0)

        epsl = f"{cores["Cinza Claro"]["letra"]}Epsilon: {cores['Vermelho']['letra']}{round(self.epsilon, 3)} "
        epsl += " " * max(30 - len(epsl), 0)

        print("\r"+eps+bt_sz+mem+mem_len+btt_rwd+btt_scr+epsl+f"\033[0m",end="   " if not self.new_line_log else "\n")