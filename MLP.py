import matplotlib.pyplot as plt
import csv
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
import pickle


class MLP:
    # inicializa o classificador
    def __init__(self, model=None):
        #carga dos dados de Treino
        entrada = []
        saidaDesejada = []

       # armazena na variavel entrada e saidaDesejada os dados de treinamento
        with open("DadosDeTreino.csv", "r") as f:
            dados = f.readlines()
            for linha in dados:
                
                coluna = linha.split(",")

                entrada.append(
                    [
                        float(coluna[0]),
                      
                        float(coluna[2]),
                    ]
                     
                )
                saidaDesejada.append(coluna[4].strip("\n"))

        print("Quantidade de dados: ", len(entrada))
        for ind in range(0,len(entrada)):
            print("Features de entrada: ", entrada[ind],"-", "Classe: " + saidaDesejada[ind])

        f.close()

        # -------------------------------------------------

        # Treinamento
        
        print("Treinando...")
        
        #definindo o modelo da rede neural MLP
        model = MLPClassifier(
            hidden_layer_sizes=(10, 10, 10, 10, 10, 10, 10, 10, 10, 10),
            max_iter=5000,
            verbose=True,
            learning_rate="constant",
            learning_rate_init=0.0001,
            activation="relu",
            solver="adam",
        )
        #'''

        model.fit(entrada, saidaDesejada)#treinando o modelo

        self.model = model#armazena o modelo na variavel model

        # -------------------------------------------------

        # Carga dos dados de Teste
        entrada = []
        saidaDesejada = []

        # armazena na variavel entrada e saidaDesejada os dados de teste
        with open("DadosDeTeste.csv", "r") as f:
            dados = f.readlines()
            for linha in dados:
                # print(linha)
                coluna = linha.split(",")
                entrada.append(
                    [
                      float(coluna[0]),
                      
                      float(coluna[2]),
                    ]
                )
                saidaDesejada.append(coluna[4].strip("\n"))



        f.close()

        # -------------------------------------------------
        # Generalização - Teste com Y
        print("Testando...")

        saidaPredita = model.predict(entrada) #predizendo a saida com os dados de teste
        print("Predição: ", saidaPredita)
        print ("Saida Desejada: ", saidaDesejada)

        #rint(model.predict_proba(entrada))
        # print(model.score(entrada,saidaDesejada,sample_weight=None))

        #mat = confusion_matrix(saidaDesejada, saidaPredita)
        #print(mat)
        print("Acurácia: ", accuracy_score(saidaDesejada, saidaPredita)) #calculando a acuracia do modelo

    
   
