#-*- coding: utf-8 -*-

import numpy as np
import classificador_q3 as DEM

# Efetua a soma de todos que ele classificou corretamente
def somaPrincipal(matriz):
    n = len(matriz)
    soma = 0
    i = 0
    while i < n:
        j = 0
        while j < n:
            if i == j:
                soma += matriz[i][j]
            j += 1
        i += 1
    return soma

# Calcula médias de cada subconjunto
def getMedias(dados):

    quantidade = 50
    medias = []

    # Classificando os dados de acordo com os atributos.
    i = 0
    while i < quantidade:
        matriz = DEM.classificadorDEM(np.array(dados))
        medias.append(somaPrincipal(matriz))
        i += 1

    # Calculando médias dos atributos de acordo com a classificação
    atributo = [np.mean(medias, 0), np.std(medias, 0)] # dados[0] é o atributo atual

    return atributo

# Seleciona inicialmente o melhor atributo do conjunto de dados
def selecionaAtributo(data,dimensoes):
    quantidade = 50
    dic_medias = {}
    dados = []
    # Inicializando dicionario de médias..
    k = 1
    while k < dimensoes[1]:
        dic_medias[k] = []
        k += 1

    # Classificando os dados de acordo com os atributos.
    i = 0
    while i < quantidade:
        k = 1
        while k < dimensoes[1]:
            classes = [data[j][0] for j in range(0, len(data))]
            dados = [[classes[j], data[j][k]] for j in range(0, len(data))]
            matriz = DEM.classificadorDEM(np.array(dados).reshape((len(dados), len(dados[0]))))
            dic_medias[k].append(somaPrincipal(matriz))
            k += 1
        i += 1

    # Calculando médias dos atributos de acordo com a classificação
    medias = []
    k = 1
    while k < dimensoes[1]:
        medias.append((k, np.mean(dic_medias[k], 0), np.std(dic_medias[k], 0)))
        k += 1

    medias_ordenadas = np.sort(medias, axis=0).tolist()

    # Reduzindo o conjunto de dados
    atributo = medias_ordenadas[0]
    return np.delete(data, atributo[0], 1)

# Começa com todo o conjunto de dados, elimando um atributo a cada passo.
def backward(data):
    # Melhor atributo para ser removido incialmente
    melhor_atributo = selecionaAtributo(data, data.shape)
    media_anterior = getMedias(melhor_atributo)

    proximo = selecionaAtributo(melhor_atributo, melhor_atributo.shape)
    media_proximo = getMedias(proximo)

    if media_proximo > media_anterior:
        return backward(proximo)
    else:
        return melhor_atributo



