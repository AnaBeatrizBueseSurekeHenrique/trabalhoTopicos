import cv2
import glob
import numpy as np
import os
def carregar_imagem(caminho):
    img = cv2.imread(caminho)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128))
    return img
def vetor_pixels(img):
    img = cv2.resize(img, (64, 64))
    vetor = img.astype("float32").flatten()
    return vetor
    # Calcula a diferença média entre duas imagens
    # Quanto menor o valor, mais parecidas elas são
def distancia_visual(vetor1, vetor2):
    return np.mean((vetor1 - vetor2) ** 2)
    # Remove duplicatas, mas percorre o ranking inteiro
    # Assim, quando uma duplicata sai, outros candidatos podem aparecer no ranking final
def remover_duplicatas(ranking, limite_distancia=30):
    linhas_mantidas = []
    vetores_usados = []
    duplicatas_removidas = []
    PATH = "../busca_de_imagem/pasta_gatos/train/images/*.jpg"
    caminhos = glob.glob(PATH)[:30] # seleciona 30 documentos

    for arq in ranking:
        arquivo = arq["path"]
        print(arquivo)
        img = carregar_imagem(arquivo)
        vetor_atual = vetor_pixels(img)
        duplicada = False
        menor_distancia = None
        for vetor_usado in vetores_usados:
            distancia = distancia_visual(vetor_atual, vetor_usado)
            if menor_distancia is None or distancia < menor_distancia:
                menor_distancia = distancia
            if distancia < limite_distancia:
                duplicada = True
                break
        nova_linha = arq.copy()
        nova_linha["menor_distancia_visual"] = 0 if menor_distancia is None else menor_distancia
        if duplicada:
            duplicatas_removidas.append(nova_linha)
        else:
            linhas_mantidas.append(nova_linha)
        vetores_usados.append(vetor_atual)
    # ranking_novo = pd.DataFrame(linhas_mantidas).reset_index(drop=True)
    # ranking_novo["nova_posicao"] = range(1, len(ranking_novo) + 1)
    # removidas = pd.DataFrame(duplicatas_removidas)
    return  sorted(linhas_mantidas, key=lambda x: x['score'], reverse=True)
    # # Mostra as imagens do ranking
    # def mostrar_ranking(ranking, titulo, coluna_posicao):
    # total = min(10, len(ranking))
    # plt.figure(figsize=(14, 4))
    # for i in range(total):
    # linha = ranking.iloc[i]
    # arquivo = linha["arquivo"]
    # img = cv2.imread(os.path.join(pasta_imagens, arquivo))
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # plt.subplot(2, 5, i + 1)
    # plt.imshow(img)
    # plt.title(
    # f'{int(linha[coluna_posicao])}º\n{arquivo}\nscore={linha["score"]:.2f}
    # ',
    # fontsize=8
    # )
    # plt.axis("off")
    # plt.suptitle(titulo)
    # plt.tight_layout()
    # plt.show()
    # ranking_original = gerar_ranking_original()
    # ranking_sem_duplicatas, imagens_removidas = remover_duplicatas(
    # ranking_original,
    # limite_distancia=30
    # )
    # ranking_original.to_csv("ranking_original_duplicatas.csv",
    # index=False)
    # ranking_sem_duplicatas.to_csv("ranking_sem_duplicatas.csv",
    # index=False)
    # imagens_removidas.to_csv("duplicatas_removidas.csv", index=False)
    # print("Ranking original")
    # print(ranking_original.head(10).to_string(index=False))
    # print("\nImagens removidas como duplicatas")
    # print(imagens_removidas[["arquivo", "score"]].to_string(index=False))
    # print("\nRanking após remoção de duplicatas")
    # print(ranking_sem_duplicatas.head(10).to_string(index=False))
    # print("\nResumo")
    # print("Total de candidatos na base:", len(ranking_original))
    # print("Top 10 original mostrado:", min(10, len(ranking_original)))
    # print("Top 10 novo mostrado:", min(10, len(ranking_sem_duplicatas)))
    # print("Duplicatas removidas:", len(imagens_removidas))
    # mostrar_ranking(ranking_original, "Ranking original",
    # "posicao_original")
    # mostrar_ranking(ranking_sem_duplicatas, "Ranking após remoção de
    # duplicatas", "nova_posicao")