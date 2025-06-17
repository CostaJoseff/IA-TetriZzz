import pygame, random, time
from Memoria import Memoria
import numpy as np
from threading import Thread
import networkx as nx
import matplotlib.pyplot as plt

from AI_Hub.TatrizEnv import TetrizEnv

stop_threads = False
def feed(env, memoria):
    global stop_threads
    i = 0
    state = env.reset(process_events)
    memoria.lembrar_inicio(state)
    while not stop_threads:
        if env.jogo.perdeu:
            state = env.reset(process_events)
            memoria.lembrar_inicio(state)
        
        if i % 10 == 0:
            i = 0
            action = 2
            state, _, _, _ = env.step(action, process_events)
        
        i += 1
        action = memoria.decidir(env.jogo.tabuleiro.model_ajust(state))
        next_state, reward, _, _ = env.step(action, process_events)
        memoria.lembrar(state, action, reward, next_state)
        state = next_state

def feed_memory(memoria, n_threads):
    threads = []
    print("Gerando threads")
    for i in range(n_threads):
        print(f"\r{i}", end="")
        thread = Thread(target=feed, args=[TetrizEnv(janela=False), memoria])
        thread.start()
        threads.append(thread)
    
    print()
    return threads

class FilaDeMovimentos():
    def __init__(self, tamanho):
        self.fila = [-1]*tamanho
        self.tamanho = tamanho

    def em_fila(self, elemento):
        self.fila.append(elemento)
        self.fila.pop(0)

    def to_numpy(self):
        return np.array(self.fila)

def criar_janela():
    pygame.init()
    info = pygame.display.Info()
    pygame.display.set_caption("TetriZzz")
    return pygame.display.set_mode([info.current_w/4, info.current_h/2])


# Parâmetros
process_events = True
env = TetrizEnv(janela=True)
total_reward = 0
state = env.reset(process_events)
memoria = Memoria()
memoria.lembrar_inicio(env.jogo.tabuleiro.model_ajust(state))
user_input = True
utilizar_memoria = True
action: int = 0
debug = False
threads = feed_memory(memoria, 0)
try:
    while True:
        if env.jogo.perdeu:
            state = env.reset(process_events)
            memoria.lembrar_inicio(env.jogo.tabuleiro.model_ajust(state))
            if debug:
                debug = False

        done = False
        i = 0
        while not done:
            if i % 1000 == 0:
                i = 0
                # print("\nPropag do i")
                memoria.propagar_conhecimento()
                memoria.log()

            if i % 20 == 0:
                action = 2
                next_state, reward, _, _ = env.step(action, process_events)
                memoria.lembrar(env.jogo.tabuleiro.model_ajust(state), action, reward, env.jogo.tabuleiro.model_ajust(next_state))

            i += 1

            if user_input:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        exit()
                    elif event.type == pygame.KEYDOWN:
                        # if event.key == pygame.K_w:
                        #     action = 0
                        # elif event.key == pygame.K_a:
                        #     action = 1
                        # elif event.key == pygame.K_s:
                        #     action = 2
                        # elif event.key == pygame.K_d:
                        #     action = 3
                        # elif event.key == pygame.K_SPACE:
                        #     action = 4

                        if event.key == pygame.K_q:
                            action = 2
                            user_input = False
                            continue
                        elif event.key == pygame.K_p:
                            memoria.propagar_conhecimento()
                        elif event.key == pygame.K_m:
                            utilizar_memoria = not utilizar_memoria
                            print("\nUtilizando memória\n"*10 if utilizar_memoria else "Memória apenas armazenando\n")
                            time.sleep(0.5)
                            break
                        elif event.key == pygame.K_l:
                            debug = True
                            memoria.propagar_conhecimento()
                        elif event.key == pygame.K_t:
                            stop_threads = True
                            print("\n\n")
                            for thread in threads:
                                thread.join()
                                i += 1
                                print(f"\rThreads finalizadas: {i}", end="")
                            print("\n\n")
                        elif event.key == pygame.K_g:
                            G = nx.DiGraph()
                            # G = nx.tree_graph()
                            for estado_key, evento in memoria.eventos_presenciados.items():
                                G.add_node(evento.estado_key)
                                for acao in range(evento.n_acoes):
                                    for evento_gerado_dict_key, evento_gerado_dict in evento.eventos_gerados[acao].items():
                                        recompensa = evento.eventos_gerados[acao][evento_gerado_dict_key]["recompensa"]
                                        G.add_node(evento_gerado_dict_key)
                                        G.add_edge(
                                            evento.estado_key,
                                            evento_gerado_dict_key
                                        )

                            assert len(memoria.eventos_presenciados) == G.number_of_nodes()
                            pos = nx.planar_layout(G)
                            # edge_labels = nx.get_edge_attributes(G, 'weight')
                            # plt.figure(figsize=(12, 8))
                            nx.draw(G, pos, with_labels=False, node_size=100, node_color='lightblue')
                            # nx.draw_networkx()
                            nx.draw_networkx_edges(G, pos)
                            plt.title("Grafo de Eventos")
                            plt.show()
                            input()
                    else:
                        continue
            
            if not utilizar_memoria:
                action = random.randint(0, 4)
            
            else:
                action = memoria.decidir(env.jogo.tabuleiro.model_ajust(state))

            if action == 5:
                continue

            if action is None:
                ""

            next_state, reward, done, _ = env.step(action, process_events)
            memoria.lembrar(env.jogo.tabuleiro.model_ajust(state), action, reward, env.jogo.tabuleiro.model_ajust(next_state))
            print(f"\r{len(memoria.eventos_presenciados)}", end="")
            # if len(memoria.eventos_presenciados) == 15:
            #     memoria.propagar_conhecimento()
            env.jogo.tabuleiro.pontos += reward
            state = next_state
            total_reward += reward
            action = 5

except KeyboardInterrupt:
    stop_threads = True
    print("\n\n")
    print("Threads finalizadas: 0", end="")
    i = 0
    memoria.mutex.release()
    for thread in threads:
        thread.join()
        i += 1
        print(f"\rThreads finalizadas: {i}", end="")
        memoria.mutex.release()
    print("\n\n")