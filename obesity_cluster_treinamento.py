import pandas as pd
import numpy as np
import math
import pickle
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist

dados = pd.read_csv('ObesityData.csv')

colunas_categoricas = ['Gender', 'family_history_with_overweight', 'FAVC', 'CAEC', 'SMOKE', 'SCC', 'CALC', 'MTRANS']
dados_features = dados.drop(columns=['NObeyesdad'])

colunas_numericas = [col for col in dados_features.columns if col not in colunas_categoricas]

for col in colunas_numericas:
    dados_features[col] = dados_features[col].fillna(dados_features[col].mean())

for col in colunas_categoricas:
    moda = dados_features[col].mode()[0]
    dados_features[col] = dados_features[col].fillna(moda)

print("Dados carregados e nulos tratados.")
print(f"Shape original: {dados.shape}")

scaler = MinMaxScaler()
dados_num_norm = pd.DataFrame(scaler.fit_transform(dados_features[colunas_numericas]), columns=colunas_numericas)

dados_cat_norm = pd.get_dummies(dados_features[colunas_categoricas], prefix_sep='_')

dados_norm = dados_num_norm.join(dados_cat_norm).astype(float)
colunas_final = dados_norm.columns.tolist()

metadata = {
    'scaler': scaler,
    'colunas_final': colunas_final,
    'colunas_numericas': colunas_numericas,
    'colunas_categoricas': colunas_categoricas
}
pickle.dump(metadata, open('metadata_obesity.pkl', 'wb'))

print(f"Shape após One-Hot Encoding: {dados_norm.shape}")

erros = []
K = range(1, 31) 

for k in K:
    modelo_temp = KMeans(n_clusters=k, random_state=42, n_init=10)
    modelo_temp.fit(dados_norm)
    erro_calculado = sum(np.min(cdist(dados_norm, modelo_temp.cluster_centers_, 'euclidean'), axis=1)) / dados_norm.shape[0]
    erros.append(erro_calculado)

x0, y0 = K[0],  erros[0]   
xn, yn = K[-1], erros[-1]  

distancias = []
for i in range(len(erros)):
    x, y = K[i], erros[i]
    numerador   = abs((yn - y0) * x - (xn - x0) * y + xn * y0 - yn * x0)
    denominador = math.sqrt((yn - y0) ** 2 + (xn - x0) ** 2)
    distancias.append(numerador / denominador)

melhor_k = K[distancias.index(np.max(distancias))]
print(f"\nNúmero de clusters sugerido pelo método do cotovelo: {melhor_k}\n")

modelo = KMeans(n_clusters=melhor_k, random_state=42, n_init=10).fit(dados_norm)

pickle.dump(modelo, open('modelo_obesity.pkl', 'wb'))

print("Modelo salvo em 'modelo_obesity.pkl'")
print("Metadados salvos em 'metadata_obesity.pkl'")

fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(K, erros, marker='o')
ax.axvline(melhor_k, color='red', linestyle='--', label=f'k ótimo = {melhor_k}')
ax.set(xlabel='Número de Clusters (k)', ylabel='Distorção média')
ax.set_title('Curva do Cotovelo — Segmentação de Obesidade')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('elbow_obesity.png', dpi=150)
print("Gráfico da curva do cotovelo salvo como 'elbow_obesity.png'")
