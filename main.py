import os
import pickle

from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# define os caminhos dos dados de treino e de teste
input_dir = 'data/train'
input_dir2 = 'data/test'

# Define as categorias de classificacao
categories = ['cats', 'dogs']

train_data = []
train_labels = []
test_data = []
test_labels = []

# Como os dados de cada categoria estao em pastas com o nome da categoria
for category_idx, category in enumerate(categories):
    # Andamos por cada uma delas adicionando seus respectivos nomes ao caminho de destino
    for file in os.listdir(os.path.join(input_dir, category)):
        # O caminho de destino de cada imagem eh dado pela soma do caminho base + categoria + nome do arquivo
        img_path = os.path.join(input_dir,category,file)
        img = imread(img_path)        # Lemos a imagem a partir de seu caminho
        img = resize(img, (64, 64))  # Modifica-se o tamanho da imagem tanto para manter
                                                 # a homogenidade quanto para otimizar a classificacao
        img = img.flatten()     # Reduz a informacao da imagem para um array

        # Um erro que estava ocorrendo era que algumas imagens estavam ficando com tamanhos diferentes, mesmo apos modificarmos o tamanho
        # por isso, a verificacao abaixo apenas deixa passar as imagens que tiverem esse tamanho exato
        # na pratica, exclui uma ou duas imagens apenas
        if len(img) == 12288:
            # O vetor que representa a imagem e o numero da sua categoria sao adicionados a lista
            train_data.append(img)
            train_labels.append(category_idx)

    # Repetimos o mesmo procedimento para os dados de teste, visto que estao em pastas separadas
    for file in os.listdir(os.path.join(input_dir2, category)):
        img_path = os.path.join(input_dir2,category,file)
        img = imread(img_path)
        img = resize(img, (64, 64))
        img = img.flatten()

        if len(img) == 12288:
            test_data.append(img.flatten())
            test_labels.append(category_idx)


print('Qnt de imagens para treino: {}'.format(len(train_data)))
print('Qnt de imagens para validacao: {}'.format(len(test_data)))

# transformamos as listas com as informacoes em um np-array
train_data = np.asarray(train_data)
test_data = np.asarray(test_data)

train_labels = np.asarray(train_labels)
test_labels = np.asarray(test_labels)

# Classificador de vetor
classificador = SVC()

# Definimos alguns parametros para o classificador
parametros = [{'gamma': [0.1, 0.01, 0.001, 0.0001], 'C': [0.1, 1, 10, 100, 1000], 'kernel':['rbf', 'poly']}]

# O conteudo de grid_search na verdade eh de 40 classificadores, um para cada combinacao de parametros que indicamos
grid_search = GridSearchCV(classificador, parametros)

# Treina-se todos os 40 classificadores de uma vez com as informacoes de treino
grid_search.fit(train_data, train_labels)

# Aqui, escolhemos o classificador que teve a melhor performance dentre os 40
best_estimator = grid_search.best_estimator_

# Retorna a previsao do classificador conforme os dados de teste
y_prediction = best_estimator.predict(test_data)

# Compara os resultados da predicao com o valor real
score = accuracy_score(y_prediction, test_labels)

print('{}% das imagens de teste foram corretamente classificadas'.format(str(score * 100)))
# A pontuacao para imagens 64x64 alcancada foi de 62.85% de imagens corretamente classificadas
# imagens com resolucao de 150x150 tiveram 56.42% de acerti

# Salva o objeto em um arquivo pickle de nome model.p para ser utilizado em outros codigos
pickle.dump(best_estimator, open('./model.p', 'wb'))




