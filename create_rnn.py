#create a simple RNN and export it as ONNX

from torch.autograd import Variable
import torch.onnx
import torchvision
import torch.nn as nn
import numpy

lstm = nn.LSTM(3, 5)

print(lstm)

'''
dummy_input = Variable(torch.from_numpy(numpy.arange(256)[:, None] % 64))
dummy_state = model.init_hidden(batch_size=1)
torch.onnx.export(model, (dummy_input, dummy_state), "rnn.onnx", verbose=True)
'''
