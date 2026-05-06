import cv2
import numpy as np
import glob
from scipy.cluster.vq import kmeans, vq
import pre_processamento 

sift = cv2.SIFT_create()

def extrair_sift_dataset(lista_caminhos_imagens):
    todos_descritores = []
    for path in lista_caminhos_imagens:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None: continue
        
        kp, des = sift.detectAndCompute(img, None)
        if des is not None:
            todos_descritores.extend(des)
    return np.array(todos_descritores, dtype=np.float32)

def gerar_histograma_bovw(roi, codebook):
    kp, des = sift.detectAndCompute(roi, None)
    hist = np.zeros(len(codebook))
    
    if des is not None:
        visual_words, _ = vq(des, codebook)
        for w in visual_words:
            hist[w] += 1
            
    norm = np.linalg.norm(hist)
    return hist / norm if norm > 0 else hist

def indexar_sistema(caminhos, codebook):
    banco_de_dados = []
    
    print(f"Indexando {len(caminhos)} imagens")
    for idx, path in enumerate(caminhos):
        img = cv2.imread(path)
        if img is None: continue
        
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        proposals = pre_processamento.selective_search(img) #região de interesse
        
        img_entry = {
            'id': idx,
            'path': path,
            'regions': []
        }
        
        for (x, y, w, h) in proposals:
            roi = img_gray[y:y+h, x:x+w]
            if roi.size == 0: continue
            
            descritor_visual = gerar_histograma_bovw(roi, codebook)
            
            img_entry['regions'].append({
                'descriptor': descritor_visual,
                'bbox': (x, y, w, h) #precisa para o calculo de IoU 
            })
        
        banco_de_dados.append(img_entry)
        if idx % 5 == 0: print(f"Progresso: {idx}/{len(caminhos)}")
        
    return banco_de_dados

