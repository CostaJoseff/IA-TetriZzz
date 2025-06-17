import random
from Evento import Evento
import numpy as np
from threading import Semaphore
from pprint import pprint

class Memoria:
    
    def __init__(self, configs: dict={}):
        self.mutex = Semaphore(1)
        self.eventos_iniciais = {}
        self.eventos_presenciados = {}
        self.random_inicial: Evento = None

    def lembrar_inicio(self, estado):
        estado_key = ''.join(map(str, estado.flatten()))

        self.mutex.acquire()
        if self.eventos_iniciais.get(estado_key, False):
            self.mutex.release()
            return
        
        self.eventos_iniciais[estado_key] = Evento(estado, estado_key)
        self.mutex.release()

    def lembrar(self, estado, acao, recompensa, estado_gerado):
        estado_key = ''.join(map(str, estado.flatten()))
        estado_gerado_key = ''.join(map(str, estado_gerado.flatten()))

        self.mutex.acquire()
        if self.eventos_presenciados.get(estado_key, False) == False:
            self.eventos_presenciados[estado_key] = Evento(estado, estado_key)
        if self.eventos_presenciados.get(estado_gerado_key, False) == False:
            self.eventos_presenciados[estado_gerado_key] = Evento(estado_gerado, estado_gerado_key)

        if self.eventos_presenciados[estado_key].eventos_gerados[acao].get(estado_gerado_key, False) == False:
            self.eventos_presenciados[estado_key].eventos_gerados[acao][estado_gerado_key] = {
                "evento_gerado": self.eventos_presenciados[estado_gerado_key],
                "recompensa": recompensa,
                "consequencia": 0,
                "conhecimento_propagado": True
            }

        if self.eventos_presenciados[estado_gerado_key].eventos_geradores.get(estado_key, False) == False:
            self.eventos_presenciados[estado_gerado_key].eventos_geradores[estado_key] = {
                "evento_gerador": self.eventos_presenciados[estado_key]
            }
            self.eventos_presenciados[estado_key].eventos_gerados[acao][estado_gerado_key]["conhecimento_propagado"] = False
        self.mutex.release()

    def propagar_conhecimento(self):
        stack = []

        self.mutex.acquire()
        for _, evento in self.eventos_iniciais.items():
            stack.append(evento)
        
        while stack:
            evento_atual: Evento = stack.pop(0)
            
            if evento_atual.contem_propagacao_pendente():
                stack_temp = []
                for _, evento_gerado_dict in evento_atual.eventos_gerados.items():
                    if not evento_gerado_dict["conhecimento_propagado"]:
                        stack_temp.append(evento_gerado_dict["evento_gerado"])

                stack_temp.append(evento_atual)
                stack = stack_temp + stack
            
            else:
                for evento_gerador_dict_key, evento_gerador_dict in evento_atual.eventos_geradores.items():
                    evento_gerador = evento_gerador_dict["evento_gerador"]

                    for acao, estados_gerados_pela_acao_dict in evento_gerador.eventos_gerados.items():
                        if not estados_gerados_pela_acao_dict.get(evento_atual.estado_key, False):
                            continue

                        evento_gerador.eventos_gerados[acao][evento_atual.estado_key]["conhecimento_propagado"] = True
                        evento_gerador.eventos_gerados[acao][evento_atual.estado_key]["consequencia"] = evento_atual.calcular_consequencia()
        self.mutex.release()

    def decidir(self, estado):
        estado_key = ''.join(map(str, estado.flatten()))

        self.mutex.acquire()
        if not self.eventos_presenciados.get(estado_key, False):
            self.mutex.release()
            return random.randint(0, 4)

        decisao = self.eventos_presenciados[estado_key].decidir()
        self.mutex.release()
        return decisao

    def log(self):
        if len(self.eventos_iniciais) == 0:
            return 
        self.mutex.acquire()
        if self.random_inicial is None:
            self.set_random_inicial()
        
        print("\n"*10)
        pprint(f"{self.random_inicial.eventos_gerados}")
        self.mutex.release()

    def set_random_inicial(self):
        for _, evento in self.eventos_iniciais.items():
            if random.randint(0, 1) or self.random_inicial is None:
                self.random_inicial = evento