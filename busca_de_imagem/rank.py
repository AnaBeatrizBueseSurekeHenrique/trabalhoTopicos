import numpy as np
import cv2

def criar_mask(shape, bbox):
    x, y, w, h = bbox
    stencil = np.zeros(shape[:2], dtype=np.uint8)
    pontos = np.array([[x, y], [x+w, y], [x+w, y+h], [x, y+h]])
    cv2.fillPoly(stencil, [pontos], 255)
    return stencil

def calcular_iou(img_shape, boxA, boxB):
    mask1 = criar_mask(img_shape, boxA)
    mask2 = criar_mask(img_shape, boxB)
    
    intersection = np.logical_and(mask1, mask2)
    union = np.logical_or(mask1, mask2)
    
    iou_score = np.sum(intersection) / np.sum(union)
    return iou_score

def buscar(query_desc, query_bbox, index, img_shape, peso_iou=0.3):
    resultados = []
    for img_entry in index:
        max_score = -1
        vencedor = {}
        
        for region in img_entry['regions']:
            sim_visual = np.dot(query_desc, region['descriptor'])
            sim_espacial = calcular_iou(img_shape, query_bbox, region['bbox'])
            
            score_final = (sim_visual * (1 - peso_iou)) + (sim_espacial * peso_iou)
            
            if score_final > max_score:
                max_score = score_final
                vencedor = {
                    'path': img_entry['path'],
                    'score': score_final,
                    'bbox': region['bbox'],
                    'sim_v': sim_visual,
                    'iou': sim_espacial
                }
        resultados.append(vencedor)
    
    return sorted(resultados, key=lambda x: x['score'], reverse=True)


