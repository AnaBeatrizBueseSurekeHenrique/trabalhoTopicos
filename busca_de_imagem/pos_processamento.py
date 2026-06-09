import cv2
import glob
import numpy as np
import os
from sklearn.cluster import KMeans

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


# CLUSTER 
def criar_mascara_objeto(roi):
    cinza = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    _, mascara = cv2.threshold(cinza, 245, 255, cv2.THRESH_BINARY_INV)
    return mascara

def calcular_formato(roi):
    mascara = criar_mascara_objeto(roi)
    contornos, _ = cv2.findContours(mascara, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contornos) == 0:
        return 0, 0, 0
        
    maior = max(contornos, key=cv2.contourArea)
    area = cv2.contourArea(maior)
    perimetro = cv2.arcLength(maior, True)
    
    circularidade = (4 * np.pi * area) / (perimetro ** 2) if perimetro > 0 else 0
    x, y, largura, altura = cv2.boundingRect(maior)
    
    proporcao = largura / altura if altura > 0 else 0
    preenchimento = area / (largura * altura) if (largura * altura) > 0 else 0
    
    return circularidade, proporcao, preenchimento

def calcular_cor_objeto(roi):
    mascara = criar_mascara_objeto(roi)
    media_bgr = cv2.mean(roi, mask=mascara)[:3]
    return media_bgr[2] / 255, media_bgr[1] / 255, media_bgr[0] / 255 

def extrair_caracteristicas_para_cluster(roi):
    vermelho, verde, azul = calcular_cor_objeto(roi)
    circularidade, proporcao, preenchimento = calcular_formato(roi)
    
    return [
        vermelho * 0.3,
        verde * 0.3,
        azul * 0.3,
        circularidade * 5,
        proporcao * 1,
        preenchimento * 5
    ]

def aplicar_cluster(img_q, query_bbox, rank_resultados, quantidade_clusters=4, bonus_cluster=0.35):

    if not rank_resultados:
        return []

    qx, qy, qw, qh = query_bbox
    roi_query = img_q[qy:qy+qh, qx:qx+qw]
    vetor_query = extrair_caracteristicas_para_cluster(roi_query)
    
    vetores = []
    
    for res in rank_resultados:
        img_res = cv2.imread(res['path'])
        rx, ry, rw, rh = res['bbox']
        roi_res = img_res[ry:ry+rh, rx:rx+rw]
        
        vetor_res = extrair_caracteristicas_para_cluster(roi_res)
        vetores.append(vetor_res)

    todos_vetores = [vetor_query] + vetores
    
    # K-Means limita a quantidade de clusters ao total de amostras disponíveis
    k = min(quantidade_clusters, len(todos_vetores))
    
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    clusters = kmeans.fit_predict(todos_vetores)
    
    cluster_query = clusters[0]
    clusters_resultados = clusters[1:]
    
    for i, res in enumerate(rank_resultados):
        res['cluster'] = clusters_resultados[i]
        if clusters_resultados[i] == cluster_query:
            res['score'] += bonus_cluster
            
    return sorted(rank_resultados, key=lambda x: x['score'], reverse=True)