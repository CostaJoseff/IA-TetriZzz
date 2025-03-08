from AI_Hub.valores import batch_size, memory_size
from threading import Semaphore
from collections import deque
import random

class Memory_():
    def __init__(self):
        self.memory_size = memory_size
        self.positive_deq_size = int(self.memory_size/2)
        self.negative_deq_size =  self.memory_size - self.positive_deq_size

        self.positive_batch_size = batch_size // 2
        self.negative_batch_size = batch_size - self.positive_batch_size

        self.dequeue_pos = deque(maxlen=self.positive_deq_size)
        self.dequeue_neg = deque(maxlen=self.negative_deq_size)
        self.mutex: Semaphore = Semaphore(1)

    def is_full(self):
        self.mutex.acquire()
        a = min(self.negative_batch_size, len(self.dequeue_neg))
        b = min(self.positive_batch_size, len(self.dequeue_pos))
        rtrn = a + b >= batch_size
        self.mutex.release()
        return rtrn 

    def __len__(self):
        self.mutex.acquire()
        rtrn = len(self.dequeue_neg) + len(self.dequeue_pos)
        self.mutex.release()
        return rtrn
    
    def len_(self):
        a = 0
        b = 0

        self.mutex.acquire()
        a = min(self.negative_batch_size, len(self.dequeue_neg))
        b = min(self.positive_batch_size, len(self.dequeue_pos))
        self.mutex.release()

        return a + b
    
    def __str__(self):
        self.mutex.acquire()
        rtrn = f"Pos - {len(self.dequeue_pos)} --- Neg - {len(self.dequeue_neg)}"
        self.mutex.release()
        return rtrn
    
    def sample(self):
        self.mutex.acquire()
        pos_sample = [self.dequeue_pos.popleft() for _ in range(min(len(self.dequeue_pos), self.positive_batch_size))]
        neg_sample = [self.dequeue_neg.popleft() for _ in range(min(len(self.dequeue_neg), self.negative_batch_size))]
        self.mutex.release()

        final_list = pos_sample+neg_sample
        random.shuffle(final_list)
        return final_list

    def append(self, data):
        self.mutex.acquire()
        if data[2] >= 0:
            self.dequeue_pos.append(data)
        else:
            self.dequeue_neg.append(data)
        self.mutex.release()