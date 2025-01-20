import torch
import torch.nn as nn
import torch.nn.functional as F

from numpy import dot
from numpy.linalg import norm

class OnlineVisualTokenMerging(nn.Module):
    def __init__(self, alpha_current=2, alpha_short=8, alpha_long=16, buffer_size=64, tau=0.95):
        super(OnlineVisualTokenMerging, self).__init__()
        # 파라미터 설정
        self.alpha_current = alpha_current
        self.alpha_short = alpha_short
        self.alpha_long = alpha_long
        self.buffer_size = buffer_size  # 메모리 버퍼 크기
        self.tau = tau

        # 메모리 버퍼 초기화 (각각 현재, 단기, 장기 토큰)
        self.curr_memory = None
        self.short_memory = None
        self.long_memory = None
        
        self.k = 1

    def forward(self, current_frame_tokens):
        """
        시각적 토큰을 병합하여 반환
        - current_frame_tokens : 현재 프레임에서 추출된 시각적 토큰
        """
        t = self._total_len()
        
        if t == 1:
            self.short_memory = []
            self.long_memory = []
            
        else:
            curr2short = self._grid_pooling(self.curr_memory, self.alpha_short / self.alpha_current)
            self.short_memory += [curr2short]

        # 현재 프레임을 병합된 토큰으로 변환
        current_tokens = self._grid_pooling(current_frame_tokens, self.alpha_current)

        # 단기 및 장기 정보 업데이트
        if t > self.buffer_size + 1:
            short2long = self._grid_pooling(self.short_memory[0], self.alpha_long / self.alpha_short)
            self.short_memory = self.short_memory[1:]
            s = self._cos_sim(self.long_memory[-1], short2long)
            
            if t > (self.buffer_size + 2) and s > self.tau:
                last_long = (self.k * self.long_memory[-1] + short2long) / (self.k + 1)
                self.long_memory = self.long_memory[:-1] + [last_long]
                self.k += 1
            else:
                self.long_memory += [short2long]
                self.k = 1
            
        
        # 현재 정보와 단기, 장기 메모리 결합
        merged_tokens = torch.cat([current_tokens, self.short_memory, self.long_memory], dim=1)

        return merged_tokens

    def _grid_pooling(self, tokens, alpha):
        """
        그리드 풀링을 통해 토큰을 축소하여 병합
        """
        return F.avg_pool2d(tokens, kernel_size=alpha)
    
    def _total_len(self):
        return 1 + len(self.short_memory) +len(self.long_memory)
    
    def _cos_sim(self, a, b):
        return dot(a, b) / (norm(a)*norm(b))
