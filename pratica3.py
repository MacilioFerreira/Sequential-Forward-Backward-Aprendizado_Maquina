#-*- coding: utf-8 -*-

import numpy as np
import forward as F
import backward as B


# Lendo arquivo
data = np.genfromtxt("wine.data", delimiter=",")

print "Sequential Forward\n"
resultado_forward = F.forward_selection(data)
# Padronizando os valores para facilitar a visualização.
#OBS:. Esse processo não afeta o conjunto de dados, pois é feito de acordo com a quantidade de casas decimais original
for linha in resultado_forward:
    linha = [round(linha[i],3) for i in range(0, len(linha))]
    print linha

print "\nSequential Backward\n"
resultado_backward = B.backward(data)

for linha in resultado_backward:
    linha = [round(linha[i],3) for i in range(0, len(linha))]
    print linha


