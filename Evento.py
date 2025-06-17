import numpy as np
import random as rd

class Evento:

    def __init__(self, estado, estado_key):
        self.estado = estado
        self.n_acoes = 5
        self.estado_key = estado_key
        self.eventos_geradores = {}
        self.eventos_gerados = {
            0:{},
            1:{},
            2:{},
            3:{},
            4:{}
        }

    def contem_propagacao_pendente(self) -> bool:
        for _, evento_gerado_dict in self.eventos_gerados.items():
            if len(evento_gerado_dict) == 0:
                continue
            
            if not evento_gerado_dict["conhecimento_propagado"]:
                return True
            
        return False
    
    def calcular_consequencia(self):
        consequencia_final = -np.inf
        for acao, eventos_gerados_dict in self.eventos_gerados.items():
            for evento_gerado_key, evento_gerado_dict in eventos_gerados_dict.items():
                if self.estado_key == evento_gerado_key:
                    return np.inf * evento_gerado_dict["recompensa"]

                if evento_gerado_dict["consequencia"] == np.inf or evento_gerado_dict["consequencia"] == -np.inf:
                    continue
                    # return evento_gerado_dict["consequencia"]

                if evento_gerado_dict["recompensa"] == np.inf or evento_gerado_dict["recompensa"] == -np.inf:
                    continue
                    # return evento_gerado_dict["recompensa"]
                
                if (evento_gerado_dict["recompensa"] + evento_gerado_dict["consequencia"]) > consequencia_final:
                    consequencia_final = evento_gerado_dict["recompensa"] + evento_gerado_dict["consequencia"]

        return consequencia_final
    
    def decidir(self):
        maior_acao = 4
        maior_recompensa = -np.inf
        for i in range(5):
            for evento_key, evento_dict in self.eventos_gerados[i].items():
                if evento_dict["recompensa"] + evento_dict["consequencia"] > maior_recompensa:
                    maior_acao = i
                    maior_recompensa = evento_dict["recompensa"] + evento_dict["consequencia"]
        
        return maior_acao