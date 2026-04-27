import pickle
import pandas as pd
import numpy as np

CAMPOS = {
    'Gender':                       ('Gênero',                        'opcoes', ['Male', 'Female']),
    'Age':                          ('Idade',                          'numero', None),
    'Height':                       ('Altura (ex: 1.75)',              'decimal', None),
    'Weight':                       ('Peso em kg (ex: 70.5)',          'decimal', None),
    'family_history_with_overweight':('Histórico familiar de sobrepeso','opcoes', ['yes', 'no']),
    'FAVC':                         ('Consome alimentos calóricos?',   'opcoes', ['yes', 'no']),
    'FCVC':                         ('Consumo de vegetais (1-3)',      'decimal', None),
    'NCP':                          ('Nº de refeições por dia (1-4)',  'decimal', None),
    'CAEC':                         ('Come entre refeições?',          'opcoes', ['no', 'Sometimes', 'Frequently', 'Always']),
    'SMOKE':                        ('Fuma?',                          'opcoes', ['yes', 'no']),
    'CH2O':                         ('Litros de água por dia (1-3)',   'decimal', None),
    'SCC':                          ('Monitora calorias?',             'opcoes', ['yes', 'no']),
    'FAF':                          ('Frequência de atividade física (0-3)', 'decimal', None),
    'TUE':                          ('Tempo em tela por dia (0-2)',    'decimal', None),
    'CALC':                         ('Consome álcool?',                'opcoes', ['no', 'Sometimes', 'Frequently', 'Always']),
    'MTRANS':                       ('Meio de transporte principal',   'opcoes', ['Automobile', 'Bike', 'Motorbike', 'Public_Transportation', 'Walking']),
}


def coletar_dados_usuario() -> dict:
    print("\n" + "═" * 60)
    print("         INSERÇÃO MANUAL DE DADOS DO PACIENTE")
    print("═" * 60)
    print(" Preencha os campos abaixo. Para opções, digite o número")
    print(" correspondente à escolha desejada.")
    print("═" * 60)

    dados = {}
    for campo, (descricao, tipo, opcoes) in CAMPOS.items():
        while True:
            try:
                if tipo == 'opcoes':
                    print(f"\n {descricao}:")
                    for i, op in enumerate(opcoes, 1):
                        print(f"   [{i}] {op}")
                    escolha = int(input(" > "))
                    if 1 <= escolha <= len(opcoes):
                        dados[campo] = opcoes[escolha - 1]
                        break
                    print(" [!] Opção inválida. Tente novamente.")
                elif tipo == 'numero':
                    dados[campo] = int(input(f"\n {descricao}: "))
                    break
                elif tipo == 'decimal':
                    dados[campo] = float(input(f"\n {descricao}: "))
                    break
            except ValueError:
                print(" [!] Valor inválido. Tente novamente.")
    return dados


def escolher_modo(paciente_exemplo: dict) -> dict:
    print("Como deseja prosseguir?")
    print("[1] Usar dados de exemplo")
    print("[2] Inserir meus dados")

    while True:
        try:
            opcao = int(input("[1] OU [2]: "))
            if opcao == 1:
                print("\nUsando dados do paciente de exemplo.")
                return paciente_exemplo
            elif opcao == 2:
                return coletar_dados_usuario()
            else:
                print("Digite 1 ou 2.")
        except ValueError:
            print("Entrada inválida. Digite 1 ou 2.")


def inferir_paciente(dados_paciente: dict) -> None:
    modelo = pickle.load(open('modelo_obesity.pkl', 'rb'))
    metadata = pickle.load(open('metadata_obesity.pkl', 'rb'))

    scaler = metadata['scaler']
    colunas_final = metadata['colunas_final']
    colunas_numericas = metadata['colunas_numericas']
    colunas_categoricas = metadata['colunas_categoricas']

    novo_df = pd.DataFrame([dados_paciente])

    print("\n" + "═" * 60)
    print("         DADOS DO NOVO PACIENTE PARA ANÁLISE")
    print("═" * 60)
    for col, val in dados_paciente.items():
        print(f" {col:<30}: {val}")
    print("═" * 60)

    novo_num_norm = pd.DataFrame(
        scaler.transform(novo_df[colunas_numericas]),
        columns=colunas_numericas
    )

    novo_cat_norm = pd.get_dummies(novo_df[colunas_categoricas], prefix_sep='_')
    cols_dummy_treino = [c for c in colunas_final if c not in colunas_numericas]
    novo_cat_norm = novo_cat_norm.reindex(columns=cols_dummy_treino, fill_value=0)

    novo_norm = novo_num_norm.join(novo_cat_norm)

    cluster_predito = modelo.predict(novo_norm)[0]

    centroide_norm = modelo.cluster_centers_[cluster_predito]
    distancia = np.linalg.norm(novo_norm.values[0] - centroide_norm)
    afinidade = "ALTA" if distancia < 0.4 else "MÉDIA" if distancia < 0.8 else "BAIXA"

    centroide_df = pd.DataFrame([centroide_norm], columns=colunas_final)
    c_num_real = scaler.inverse_transform(centroide_df[colunas_numericas])[0]

    print("\n" + "╔" + "═" * 58 + "╗")
    print(f"║ {'RESULTADO DA SEGMENTAÇÃO':^56} ║")
    print("╠" + "═" * 58 + "╣")
    print(f"║ O paciente foi classificado no CLUSTER: {cluster_predito:<13} ║")
    print(f"║ Grau de afinidade com este grupo: {afinidade:<21} ║")
    print(f"║ (Distância ao centro do grupo: {distancia:.4f}) {' ':<19} ║")
    print("╚" + "═" * 58 + "╝")

    print("\n" + "PERFIL REPRESENTATIVO DESTE GRUPO:")
    print(f"Idade Média: {c_num_real[0]:.1f}")
    print(f"Peso Médio:  {c_num_real[2]:.1f}")
    print(f"Altura Média: {c_num_real[1]:.2f}")
    print(f"Consumo de Água (CH2O): {c_num_real[5]:.1f}")
    print(f"Atividade Física (FAF): {c_num_real[6]:.1f}")
    print(f"Consumo de Vegetais (FCVC): {c_num_real[3]:.1f}")
    print(f"Refeições Diárias (NCP): {c_num_real[4]:.1f}")
    print("-" * 60)


if __name__ == '__main__':
    paciente_exemplo = {
        'Gender': 'Male',
        'Age': 24,
        'Height': 1.80,
        'Weight': 85.0,
        'family_history_with_overweight': 'yes',
        'FAVC': 'yes',
        'FCVC': 3.0,
        'NCP': 3.0,
        'CAEC': 'Sometimes',
        'SMOKE': 'no',
        'CH2O': 2.0,
        'SCC': 'no',
        'FAF': 1.0,
        'TUE': 1.0,
        'CALC': 'Sometimes',
        'MTRANS': 'Public_Transportation'
    }

    try:
        dados = escolher_modo(paciente_exemplo)
        inferir_paciente(dados)
    except FileNotFoundError:
        print("\n[ERRO] Arquivos de modelo não encontrados. Execute 'obesity_cluster_treinamento.py' primeiro.")
    except Exception as e:
        print(f"\n[ERRO] Ocorreu um erro na inferência: {e}")