from math import log
import string
from collections import defaultdict
from unicodedata import normalize
def tfidf(query, documentos):
    palavras = query.split()
    results = {}

    for doc in documentos:
        tfidfs = [
            matriz_termo_documento[doc["titulo"]].get(palavra, 0)
            * idf.get(palavra, 0)
            for palavra in palavras
        ]

        results[doc["titulo"]] = sum(tfidfs)
    #nos resultados, o primeiro val armazena o titulo do documento! e o segundo a porcentagem do calculo do tfidf
    
    results = sorted(results.items(), key=lambda x: x[1], reverse=True)

    return results

def limpezaDocs(documento):
    #palavras que não adicionam muito a frase em si
    palavras_filler = ['a','e','o']
    tabela = str.maketrans('','',string.punctuation)
    
    for doc in documento:        
            frase = doc["conteudo"]
            #se descomentar essas linhas abaixo, vai tirar caracteres como ã ç!
            # frase = normalize('NFD', str(frase)).encode('latin-1', 'ignore')
            # frase = frase.decode('latin-1')
            frase = frase.split()
            #deixa em minuscula
            frase = [palavra.lower() for palavra in frase]
            frase = [palavra.translate(tabela) for palavra in frase]
            #deixa só letras
            frase = [palavra for palavra in frase if palavra.isalpha()]
            #tira palavras filler
            frase = [palavra for palavra in frase if palavra not in palavras_filler]
            doc["conteudo"] = frase
    return documento


documentos = [
    {
        "titulo": "O gato preto",
        "conteudo": "O gato preto subiu no telhado para caçar aves.",},
    {
        "titulo": "Os cachorros grandes",
        "conteudo":"Cachorros grandes precisam de muito espaço para correr.",},
    {
        "titulo": "O gato e o cachorro",
        "conteudo": "O gato e o cachorro são animais domésticos populares.",},
    {
        "titulo": "O que são aviões?",
        "conteudo":"Aviões grandes decolam rapidamente do aeroporto.",},
    {
        "titulo": "O cachorro preto",
        "conteudo":  "O cachorro preto correu atrás do gato pelo jardim.",},
]

qntdPalavras = defaultdict(int)
matriz_termo_documento = {}
idf = {}

print("Inicio pré-processamento........")
documentos = limpezaDocs(documentos)

print("Documentos após processamento: ")
print(documentos)

for doc in documentos:
    frase = doc["conteudo"]

    matriz_termo_documento[doc["titulo"]] = defaultdict(int)
    #conta a quantidade de palavras em documentos    
    for palavra in frase:
        qntdPalavras[palavra] += 1
        matriz_termo_documento[doc["titulo"]][palavra] += 1
print("Matriz-Termo-Documento")
print(matriz_termo_documento)
for doc in documentos:
    frase = doc["conteudo"]
    #muda o valor lá em cima baseado na importancia da palavra na frase em si
    matriz_termo_documento[doc["titulo"]] = {
        palavra: count / len(frase)
        for palavra, count in matriz_termo_documento[doc["titulo"]].items()
    }
print(matriz_termo_documento)
#calculo de log pra evitar Problemas em documentos mais longos:
for palavra, count in qntdPalavras.items():
    idf[palavra] = (
        log(len(documentos) / count)
    )
print("Valor IDF:")
print(idf)

print("Realização query:")
result = tfidf("gato preto", documentos)

for i in range(0, len(result)):
    print(f"#{i+1}  : Titulo, {result[i][0]}, %:{result[i][1]}")