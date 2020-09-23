import torch
import random

class DataPrefetcher(object):
    def __init__(self, loader, cnt=3):
        self.list = []
        self.pre_idx = 0
        self.idx = 0
        self.cnt = cnt ###len(loader.dataset.train_scales)
        self.arr = list(range(self.cnt))
        self.loader = iter(loader)
        self.stream = torch.cuda.Stream()
        self.preload()


    def preload(self):
        try:
            if len(self.list) == 0 or len(self.list) >= self.cnt:
                self.idx = 0
                self.pre_idx = 0
                self.list = []
                random.shuffle(self.arr)
                self.next_input, self.next_depth, self.next_target, self.next_gate = next(self.loader)
        except StopIteration:
            self.next_input = None
            self.next_depth = None
            self.next_target = None
            self.next_gate = None
            return

        with torch.cuda.stream(self.stream):
            idx = self.arr[self.pre_idx]
            self.next_input[idx] = self.next_input[idx].cuda(non_blocking=True)
            self.next_depth[idx] = self.next_depth[idx].cuda(non_blocking=True)
            self.next_target[idx] = self.next_target[idx].cuda(non_blocking=True)
            self.next_gate[idx] = self.next_gate[idx].cuda(non_blocking=True)
            self.idx = idx
            self.pre_idx += 1
            self.list.append(idx)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        if self.next_input is None:
            return None, None, None, None
        input = self.next_input[self.idx]
        depth = self.next_depth[self.idx]
        target = self.next_target[self.idx]
        gate = self.next_gate[self.idx]
        self.preload()
        return input, depth, target, gate
