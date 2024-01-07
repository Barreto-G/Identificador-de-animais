import pickle
import os
from skimage.io import imread
from skimage.transform import resize
import numpy as np
from sklearn.metrics import accuracy_score
from classe import Imagem
import matplotlib.pyplot as plt

arq_modelo = open('model.p', 'rb')
modelo = pickle.load(arq_modelo)

diretorio = 'data/teste_final'

categories = ['cats', 'dogs']
test_data = []
test_labels = []
imagens = []

for categories_idx, categoria in enumerate(categories):
    for file in os.listdir(os.path.join(diretorio, categoria)):
        img_path = os.path.join(diretorio, categoria, file)
        img = imread(img_path)
        img2 = resize(img, (64, 64))
        img2 = img2.flatten()
        if len(img2) == 12288:
            imagens.append(Imagem(img_path, img, img2, categoria))
            test_data.append(img2)
            test_labels.append(categories_idx)



test_data = np.asarray(test_data)
test_labels = np.asarray(test_labels)

predicao = modelo.predict(test_data)

fig, axs = plt.subplots(3,4)
axs = axs.flatten()

for i in range(0,len(predicao)):
    axs[i].imshow(imagens[i].conteudo)
    if predicao[i] == 0:
        axs[i].set_title('Gato')
    else:
        axs[i].set_title('Cachorro')
    axs[i].axis('off')
plt.show()


score = accuracy_score(predicao, test_labels)

print('{}% das imagens de teste foram corretamente classificadas'.format(str(score * 100)))
