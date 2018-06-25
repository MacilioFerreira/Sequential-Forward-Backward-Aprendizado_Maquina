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

# Auxiliar para o cálculo das médias
def gerarMatriz(base, atributo, dados):
    matriz = base.tolist()
    elementos = [dados[i][int(atributo)] for i in range(0, len(dados))]
    i = 0
    while i < len(elementos):
        matriz[i].append(elementos[i])
        i += 1


    return np.array(matriz)

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
    return np.array(dados).reshape((len(dados), 2)), medias_ordenadas[-1]

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

#Inicia com nenhum atributo e inclui um atributo por vez no conjunto
def forward_selection(data):


    dimensoes = data.shape

    # Melhor atributo inicial
    melhor_atributo, media = selecionaAtributo(data,dimensoes)

    # Atributo selecionado anteriormente
    selecionados = [int(media[0])]

    # quantidade de atributos após a escolha do primeiro
    i = 2
    while i < dimensoes[1]: # para cada atributo, efetuar a escolha do melhor daquele conjunto
        # Para adicionar um novo atributo, a média de acertos desse atributo deve ser a melhor entre os demais
        media_atual = media[1]  #0
        medias = []
        j = 1
        while j < dimensoes[1]:
            if not j in selecionados:
                nv_matriz = gerarMatriz(melhor_atributo,j,data)
                media_proximo, desvio = getMedias(nv_matriz)
                if media_proximo > media_atual:
                    selecionados.append(j)
                    media_atual = media_proximo
                    medias.append((j, media_proximo))
            j += 1
        # adicionando
        if medias != []:
            medias = np.sort(medias, axis=-1)
            melhor_atributo = gerarMatriz(melhor_atributo, medias[-1][0], data)
        i += 1

    #Resultado final
    return melhor_atributo

