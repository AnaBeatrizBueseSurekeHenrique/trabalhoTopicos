import glob
import cv2
import pre_processamento
import index_descri
import rank 
from scipy.cluster.vq import kmeans
import matplotlib.pyplot as plt
import pos_processamento
def mostrar_resultados(query_path, query_bbox, rank_resultados, top_n=5, query_id=1):
    plt.figure(figsize=(20, 10))
    
    img_q = cv2.imread(query_path)
    img_q = cv2.cvtColor(img_q, cv2.COLOR_BGR2RGB)
    
    qx, qy, qw, qh = query_bbox
    cv2.rectangle(img_q, (qx, qy), (qx+qw, qy+qh), (255, 0, 0), 3)
    
    plt.subplot(1, top_n + 1, 1)
    plt.imshow(img_q)
    plt.title(f"QUERY {query_id}\n(Busca)")
    plt.axis("off")

    # exibi os Resultados
    for i in range(min(top_n, len(rank_resultados))):
        resultado = rank_resultados[i]
        path_res = resultado['path']
        score = resultado['score']
        img_res = cv2.imread(path_res)
        img_res = cv2.cvtColor(img_res, cv2.COLOR_BGR2RGB)
        
        plt.subplot(1, top_n + 1, i + 2)
        plt.imshow(img_res)
        plt.title(f"Rank {i+1}\nScore: {score:.4f}")
        plt.axis("off")

    plt.tight_layout()
    plt.show()

PATH = "../busca_de_imagem/pasta_gatos/train/images/*.jpg"
caminhos = glob.glob(PATH)[:30] # seleciona 30 documentos

print("\n----------SIFT e Codebook---------")
des_banco = index_descri.extrair_sift_dataset(caminhos)
codebook, _ = kmeans(des_banco, 200)

print("\n--------Indexação----------")
index = index_descri.indexar_sistema(caminhos, codebook)

# 5 imagens diferentes para serem as queries
indices_das_queries = [0, 5, 10, 15, 20]

print("\n--- Buscas ---")

for id_query, indice in enumerate(indices_das_queries, start=1):
    try:
        query_path = caminhos[indice]
        print(f"Processando Query {id_query}...")
        
        img_q = cv2.imread(query_path)

        props = pre_processamento.selective_search(img_q)

        # Filtra regiões com menos de 90% da área total da imagem 
        q_bbox = max([p for p in props if p[2]*p[3] < (img_q.shape[0] * img_q.shape[1] * 0.9)], key=lambda b: b[2]*b[3])

        x, y, w, h = q_bbox
        q_desc = index_descri.gerar_histograma_bovw(cv2.cvtColor(img_q[y:y+h, x:x+w], cv2.COLOR_BGR2GRAY), codebook)

        rank_resultados = rank.buscar(q_desc, q_bbox, index, img_q.shape, peso_iou=0.3)
        rank_resultados = pos_processamento.remover_duplicatas(rank_resultados)
        mostrar_resultados(query_path, q_bbox, rank_resultados, top_n=4, query_id=id_query)
        
    except Exception as e:
        print(f"Erro na Query {id_query}: {e}")

print("Processo concluído.")