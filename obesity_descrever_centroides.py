import pickle
import pandas as pd
import numpy as np

modelo = pickle.load(open('modelo_obesity.pkl', 'rb'))
metadata = pickle.load(open('metadata_obesity.pkl', 'rb'))

scaler = metadata['scaler']
colunas_final = metadata['colunas_final']
colunas_numericas = metadata['colunas_numericas']
colunas_categoricas = metadata['colunas_categoricas']

centroides_norm = pd.DataFrame(modelo.cluster_centers_, columns=colunas_final)

centroides_num_norm = centroides_norm[colunas_numericas]
centroides_num_reais = pd.DataFrame(
    scaler.inverse_transform(centroides_num_norm),
    columns=colunas_numericas
)

centroides_cat_norm = centroides_norm.drop(columns=colunas_numericas)

lista_df_cat = []
for cat_orig in colunas_categoricas:
    cols_dummy = [c for c in centroides_cat_norm.columns if c.startswith(cat_orig + '_')]
    df_dummy = centroides_cat_norm[cols_dummy]
    
    mais_representativa = df_dummy.idxmax(axis=1)
    
    mais_representativa = mais_representativa.str.replace(cat_orig + '_', '', regex=False)
    
    lista_df_cat.append(pd.DataFrame({cat_orig: mais_representativa}))

centroides_cat_reais = pd.concat(lista_df_cat, axis=1)

centroides_reais = centroides_num_reais.join(centroides_cat_reais)
centroides_reais.index.name = 'Cluster'

membros = pd.Series(modelo.labels_).value_counts().sort_index()
centroides_reais['Qtd_Pacientes'] = membros.values

print("=" * 100)
print("         DESCRIÇÃO DOS SEGMENTOS (CLUSTERS) — ESTUDO DE OBESIDADE")
print("=" * 100)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.float_format', '{:.2f}'.format)
print(centroides_reais)

print("\n" + "=" * 100)
print("         PERFIL INTERPRETATIVO POR CLUSTER")
print("=" * 100)

for idx, row in centroides_reais.iterrows():
    genero = row['Gender']
    idade = row['Age']
    peso = row['Weight']
    historico = "SIM" if row['family_history_with_overweight'] == 'yes' else "NÃO"
    transporte = row['MTRANS']
    
    perfil_peso = "ALTO" if peso > 90 else "MÉDIO" if peso > 70 else "BAIXO"
    
    print(f"\n  Cluster {idx}  ({int(row['Qtd_Pacientes'])} pacientes)")
    print(f"  {'─'*60}")
    print(f"  Perfil Predominante: {genero}, Média {idade:.1f} anos, Peso médio {peso:.1f}kg ({perfil_peso})")
    print(f"  Histórico Familiar: {historico:<5} | Transporte: {transporte}")
    print(f"  Hábitos: Água (CH2O): {row['CH2O']:.1f} | Refeições (NCP): {row['NCP']:.1f} | Atividade (FAF): {row['FAF']:.1f}")

print("\n" + "=" * 100)

centroides_reais.to_csv('centroides_obesidade.csv')
print("Tabela de centroides salva em 'centroides_obesidade.csv'")
