import torch

batch_size = 5000
tamanho_fila_movimentos = 15
memory_size = 50000
next_reward_focus = 0.1
learning_rate = 0.003

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")