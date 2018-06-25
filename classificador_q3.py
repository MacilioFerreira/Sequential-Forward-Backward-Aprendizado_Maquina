# -*- coding: utf-8 -*-
import numpy as np
import math

# Distância Euclediana
def euclidiana(valorTeste, centroide):
    distEucli = 0
    i = 0
    while i < len(valorTeste):
        distEucli += math.pow((np.abs(valorTeste[i] - centroide[i])), 2)
        i += 1

    return math.sqrt(distEucli)

# Quantidade de atributos classificadas corretamente
def positivo(classe, teste):
    cont = 0
    for valor in classe:
        if valor in teste:
            cont += 1
    return cont

def isOutra(classe1, classe2):
    cont = 0
    for valor in classe1:
        if (valor in classe2):
            cont += 1
    return cont


def classificadorDEM(data):

    #data = np.genfromtxt(nome_arquivo, delimiter=",")

    # Normalizar os dados
    # Médias
    dim = data.shape # dimensões
    mediaGeral = []
    i = 0
    while i < dim[1]:
        mediaGeral.append(np.mean([data[a][i] for a in range(len(data))], 0))
        i += 1

    # Desvio Padrão
    desvioGeral = []
    j = 0
    while j < dim[1]:
        desvioGeral.append(np.std([data[a][j] for a in range(len(data))], 0))
        j += 1

    # Normalizar, cada atributo recebe a diferença dele para a sua média sobre o desvio padrão
    for coluna in data:
        k = 1
        while k < dim[1]:
            j = 0
            while j < len(mediaGeral):
                coluna[k] = np.abs((coluna[k] - mediaGeral[j]) / desvioGeral[j])
                j += 1
            k += 1

    # Gerando posições aleatórias..
    posicoes = np.random.permutation(len(data))

    # Embaralhando os dados..
    data_alterado = np.zeros(data.shape)

    for i in xrange(0, len(posicoes)):
        data_alterado[i] = data[posicoes[i]]

    # Separação entre treinamento e testes..
    t_treinamento = int((0.8) * (len(data)))
    t_testes = (len(data) - t_treinamento)

    # Dados normalizados, encontrar classes
    # Conjunto de treinamento
    # treinamento = [data_alterado[i][0:6] for i in range(0, t_treinamento)]
    treinamento = np.zeros((t_treinamento, dim[1]))
    for i in range(0, t_treinamento):
        linha = data_alterado[i].tolist()
        treinamento[i] = linha

    classe1 = []  # Hernia
    classe2 = []  # Spondylolisthesis
    classe3 = []  # Normal
    for linha in treinamento:
        if int(linha[0]) == 1:
            classe1.append(linha[1:])
        if int(linha[0]) == 2:
            classe2.append(linha[1:])
        if int(linha[0]) == 3:
            classe3.append(linha[1:])

    # Calcular centroide das classes
    centroide1 = np.mean(classe1, axis=0)
    centroide2 = np.mean(classe2, axis=0)
    centroide3 = np.mean(classe3, axis=0)

    # Usar a distância euclidiana e classificar o conjunto de testes.
    # Conjunto de teste.
    teste = np.zeros((t_testes, dim[1]))
    cont = 0
    for i in range(t_treinamento, (t_testes + t_treinamento)):
        linha = data_alterado[i].tolist()
        teste[cont] = linha  # [0:6]
        cont += 1

    # Classificando
    classe_1 = []
    classe_2 = []
    classe_3 = []
    for linha in teste:
        d1 = euclidiana(linha[1:], centroide1)
        d2 = euclidiana(linha[1:], centroide2)
        d3 = euclidiana(linha[1:], centroide3)
        if (d1 < d2) and (d1 < d3):
            classe_1.append(linha.tolist())
        if (d2 < d1) and (d2 < d3):
            classe_2.append(linha.tolist())
        if (d3 < d1) and (d3 < d2):
            classe_3.append(linha.tolist())  # Inserindo a linha completa, facilita o teste

    # Matriz de confusão
    t_c1 = []  # H
    t_c2 = []  # S
    t_c3 = []  # N
    # Colocando os atributos de teste em suas respectivas classes
    for linha in teste:
        if linha[0] == 1:
            t_c1.append(linha.tolist())
        if linha[0] == 2:
            t_c2.append(linha.tolist())
        if linha[0] == 3:
            t_c3.append(linha.tolist())

    # Criando matriz de confusão
    matriz_cfs = []
    matriz_cfs.append([positivo(classe_1, t_c1), isOutra(classe_1, t_c2), isOutra(classe_1, t_c3)])
    matriz_cfs.append([isOutra(classe_2, t_c1), positivo(classe_2, t_c2), isOutra(classe_2, t_c3)])
    matriz_cfs.append([isOutra(classe_3, t_c1), isOutra(classe_3, t_c2), positivo(classe_3, t_c3)])

    return matriz_cfs

