import torch

batch_size = 2000
tamanho_fila_movimentos = 15
memory_size = int(batch_size * 1.5)
next_reward_focus = 0.8
learning_rate = 0.003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")