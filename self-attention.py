import torch
from torch import nn
import math

class SelfAttetion_v1(nn.Module):
  def __init__(self, hiddenDim):
    super().__init__()

    self.hiddenDim = hiddenDim

    self.Q_proj = nn.Linear(hiddenDim, hiddenDim)
    self.K_proj = nn.Linear(hiddenDim, hiddenDim)
    self.V_proj = nn.Linear(hiddenDim, hiddenDim)

  def forward(self, X):
    # (b, s, h)
    Q = self.Q_proj(X)
    K = self.K_proj(X)
    V = self.V_proj(X)

    attention = torch.matmul(
      Q, K.transpose(1, 2)
    ) / math.sqrt(self.hiddenDim)
    attention_weight = torch.softmax(
      attention, dim = -1
    )

    output = torch.matmul(attention_weight, V)
    return output


X = torch.rand(3, 2, 4)
net = SelfAttetion_v1(4)
net(X)


class SelfAttetion_v2(nn.Module):
  def __init__(self, hiddenDim):
    super().__init__()

    self.hiddenDim = hiddenDim

    self.proj = nn.Linear(hiddenDim, 3 * hiddenDim)

  def forward(self, X):
    QKV = self.proj(X)
    Q, K, V = torch.split(QKV, self.dim, dim = -1)

    attention = torch.matmul(
      Q, K.transpose(1, 2)
    ) / math.sqrt(self.hiddenDim)
    attention_weight = torch.softmax(
      attention, dim = -1
    )

    output = torch.matmul(attention_weight, V)
    return output


class SelfAttetion_v3(nn.Module):
  def __init__(self, hiddenDim):
    super().__init__()

    self.hiddenDim = hiddenDim

    self.proj = nn.Linear(hiddenDim, 3 * hiddenDim)
    self.att_drop = nn.Dropout(0.1)

  def forward(self, X, attention_mask=None):
    QKV = self.proj(X)
    Q, K, V = torch.split(QKV, self.dim, dim = -1)

    attention_weight = torch.matmul(
      Q, K.transpose(1, 2)
    ) / math.sqrt(self.hiddenDim)

    if attention_mask is not None:
      attention_weight = attention_weight.masked_fill(attention_mask == 0, float('-inf'))

    attention_weight = torch.softmax(
      attention, dim = -1
    )

    attention_weight = self.att_drop(attention_weight)

    output = torch.matmul(attention_weight, V)
    return output

class SelfAttetion_v4(nn.Module):
  def __init__(self, hiddenDim, headNum):
    super().__init__()

    self.hiddenDim = hiddenDim
    self.headNum = headNum
    self.headDim = hiddenDim / headNum

    self.Q_proj = nn.Linear(hiddenDim, hiddenDim)
    self.K_proj = nn.Linear(hiddenDim, hiddenDim)
    self.V_proj = nn.Linear(hiddenDim, hiddenDim)
    self.out_proj = nn.Linear(hiddenDim, hiddenDim)
    self.atten_drop = nn.Dropout(0.1)

  def forward(self, X, attention_mask=None):
    # (b, s, dim)

    batch, seq, _ = X.size()

    # (b, headnum, s, headdim)
    Q = self.Q_proj(X).view(batch, seq, self.headNum, self.headDim).transpose(1, 2)
    K = self.K_proj(X).view(batch, seq, self.headNum, self.headDim).transpose(1, 2)
    V = self.V_proj(X).view(batch, seq, self.headNum, self.headDim).transpose(1, 2)

    # (b, headnum, s, s)
    attention_weight = torch.matmul(
      Q, K.transpose(2, 3)
    ) / math.sqrt(self.headdim)

    if attention_mask is not None:
      attention_weight = attention_weight.masked_fill(attention_mask == 0, float('-inf'))

    attention_weight = torch.softmax(
      attention, dim = -1
    )

    attention_weight = self.atten_drop(attention_weight)

    # (b, headnum, s, headDim)
    output = torch.matmul(attention_weight, V)

    output = out_put.transpose(1, 2).contiguous()
    output = out_put.view(batch, seq, -1)

    return self.out_proj(output)

# GQA
# 多个 Q 的head公用一个V的head
class GroupQueryAttetion(nn.Module):
    def __init__(self, hiddendim, headnum, keyvaluenum):
      super().__init__()
      self.hiddendim = hiddendim
      assert hiddendim % headnum == 0
      assert headnum % keyvaluenum == 0
      self.headdim = hiddendim // headnum
      self.keyvaluenum = keyvaluenum
      self.headnum = headnum

      self.Q_proj = nn.Linear(hiddendim, hiddendim)
      self.K_proj = nn.Linear(hiddendim, keyvaluenum * self.headdim)
      self.V_proj = nn.Linear(hiddendim, keyvaluenum * self.headdim)
      self.out_proj = nn.Linear(hiddendim, hiddendim)

    def forward(self, X):
      # X (batch, seq, hiddendim)
      batch, seq, _ = X.size()
      q = self.Q_proj(X)
      k = self.K_proj(X)
      v = self.V_proj(X)

      q = q.view(batch, seq, self.headnum, self.headdim)
      k = k.view(batch, seq, self.keyvaluenum, self.headdim)
      v = v.view(batch, seq, self.keyvaluenum, self.headdim)

      # (batch, self.headnum, seq, self.headdim)
      q = q.transpose(1, 2)
      # (batch, self.keyvaluenum, seq, self.headdim)
      k = k.transpose(1, 2)
      v = v.transpose(1, 2)

      k = k.repeat_interleave(self.headnum // self.keyvaluenum, dim = 1)
      v = v.repeat_interleave(self.headnum // self.keyvaluenum, dim = 1)

      # (batch, self.headnum, seq, seq)
      attetion_weight = torch.matmul(
        q, v.transpose(-1, -2)
      )

      attetion_weight = torch.softmax(attetion_weight, dim = -1)

      # (batch, self.headnum, seq, headdim)
      output = torch.matmul(attetion_weight, v)

      # 
      output = output.transpose(1, 2).contiguous()
      output = output.view(batch, seq, -1)

      return self.out_proj(output)
