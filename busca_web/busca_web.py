# import itidf
import pandas as pd
import networkx as nx

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

docs =[ {
        "id": "A",
        "titulo": "Portal Tech",
        "texto": "portal com cursos de tecnologia programação python java banco de dados gratuito",
        "links": ["B"]
    },
{
        "id": "B",
        "titulo": "Curso Python",
        "texto": "curso de python gratuito para iniciantes com exercícios e projeto final",
        "links": []
    }
]
df = pd.DataFrame(docs)
print(df.values)
# df.columns = ["id","titulo","texto","links"]
print("\n1) paginas da mini-web")
print(df[["id", "titulo", "texto", "links"]].to_string(index=False))

consulta = "python"
print(f"Consulta: {consulta}")
relevantes = {"B"}

print("\n3) paginas relevantes para essa consulta")
print(sorted(relevantes))

indice = {}

for doc in docs:
    palavras = doc["texto"].lower().split()

    for palavra in palavras:
        indice.setdefault(palavra, set()).add(doc["id"])

print("\n4) exemplo de indice invertido")
for palavra in ["curso", "python", "gratuito", "java", "apostila"]:
    print(f"{palavra}: {sorted(indice.get(palavra, []))}")


textos = [doc["texto"] for doc in docs]

vectorizer = TfidfVectorizer(lowercase=True)
X = vectorizer.fit_transform(textos)

q = vectorizer.transform([consulta])
scores_texto = cosine_similarity(q, X).flatten()

ranking_texto = pd.DataFrame({
    "id": [doc["id"] for doc in docs],
    "titulo": [doc["titulo"] for doc in docs],
    "score_texto": scores_texto
})

ranking_texto = ranking_texto.sort_values("score_texto", ascending=False).reset_index(drop=True)
ranking_texto["posicao_texto"] = ranking_texto.index + 1
ranking_texto["top3_texto"] = ranking_texto["posicao_texto"] <= 3
ranking_texto["relevante"] = ranking_texto["id"].isin(relevantes)

tabela_texto = ranking_texto[
    ["posicao_texto", "id", "titulo", "score_texto", "top3_texto", "relevante"]
]

print("\n5) ranking textual")
print(tabela_texto.round(6).to_string(index=False))


# grafo da mini-web
# cada pagina vira um nó e cada link vira uma seta

G = nx.DiGraph()

for doc in docs:
    G.add_node(doc["id"], titulo=doc["titulo"])

    for destino in doc["links"]:
        G.add_edge(doc["id"], destino)

df_links = pd.DataFrame(list(G.edges()), columns=["origem", "destino"])

print("\n6) links da mini-web")
print(df_links.to_string(index=False))


# links recebidos
# isso mostra quantas paginas apontam para cada pagina

df_links_recebidos = pd.DataFrame({
    "id": list(dict(G.in_degree()).keys()),
    "links_recebidos": list(dict(G.in_degree()).values())
})

df_links_recebidos = df_links_recebidos.merge(df[["id", "titulo"]], on="id")
df_links_recebidos = df_links_recebidos[["id", "titulo", "links_recebidos"]]
df_links_recebidos = df_links_recebidos.sort_values("links_recebidos", ascending=False).reset_index(drop=True)

print("\n7) links recebidos")
print(df_links_recebidos.to_string(index=False))


# pagerank
# mede a importancia da pagina usando os links

pagerank = nx.pagerank(G, alpha=0.85)

df_pr = pd.DataFrame({
    "id": list(pagerank.keys()),
    "pagerank": list(pagerank.values())
})

df_pr = df_pr.merge(df[["id", "titulo"]], on="id")
df_pr = df_pr[["id", "titulo", "pagerank"]]
df_pr = df_pr.sort_values("pagerank", ascending=False).reset_index(drop=True)
df_pr["posicao_pagerank"] = df_pr.index + 1

print("\n8) ranking por pagerank")
print(df_pr[["posicao_pagerank", "id", "titulo", "pagerank"]].round(6).to_string(index=False))


# hits
# authority = pagina boa como fonte
# hub = pagina boa como guia de links

hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)

df_hits = pd.DataFrame({
    "id": [doc["id"] for doc in docs],
    "titulo": [doc["titulo"] for doc in docs],
    "hub": [hubs[doc["id"]] for doc in docs],
    "authority": [authorities[doc["id"]] for doc in docs]
})

# so pra nao aparecer -0.000000 por causa de arredondamento

for col in ["hub", "authority"]:
    df_hits[col] = df_hits[col].apply(lambda x: 0 if abs(x) < 0.0000001 else x)

ranking_authorities = df_hits.sort_values("authority", ascending=False).reset_index(drop=True)
ranking_authorities["posicao_authority"] = ranking_authorities.index + 1

ranking_hubs = df_hits.sort_values("hub", ascending=False).reset_index(drop=True)
ranking_hubs["posicao_hub"] = ranking_hubs.index + 1

print("\n9) hits - ranking de authorities")
print(
    ranking_authorities[["posicao_authority", "id", "titulo", "authority"]]
    .round(6)
    .to_string(index=False)
)

print("\n10) hits - ranking de hubs")
print(
    ranking_hubs[["posicao_hub", "id", "titulo", "hub"]]
    .round(6)
    .to_string(index=False)
)


# agora vem a parte mais importante
# aqui eu junto os resultados para criar um ranking final
# texto, pagerank e authority têm escalas diferentes
# por isso eu normalizo tudo antes de somar

df_rank = ranking_texto[["id", "titulo", "score_texto"]].merge(
    df_pr[["id", "pagerank"]],
    on="id"
)

df_rank = df_rank.merge(
    df_hits[["id", "hub", "authority"]],
    on="id"
)


def normalizar_coluna(df, coluna):
    maior_valor = df[coluna].max()

    if maior_valor == 0:
        return 0

    return df[coluna] / maior_valor


for col in ["score_texto", "pagerank", "authority"]:
    df_rank[col + "_norm"] = normalizar_coluna(df_rank, col)


# media ponderada
# texto pesa mais porque a pagina precisa responder a consulta
# pagerank e authority entram para dar reputacao pela estrutura de links

peso_texto = 0.60
peso_pagerank = 0.25
peso_authority = 0.15

df_rank["score_final"] = (
    peso_texto * df_rank["score_texto_norm"] +
    peso_pagerank * df_rank["pagerank_norm"] +
    peso_authority * df_rank["authority_norm"]
)

ranking_antes = df_rank.sort_values("score_final", ascending=False).reset_index(drop=True)
ranking_antes["posicao_antes"] = ranking_antes.index + 1
ranking_antes["top3_antes"] = ranking_antes["posicao_antes"] <= 3
ranking_antes["relevante"] = ranking_antes["id"].isin(relevantes)

tabela_antes = ranking_antes[
    [
        "posicao_antes",
        "id",
        "titulo",
        "score_texto_norm",
        "pagerank_norm",
        "authority_norm",
        "score_final",
        "top3_antes",
        "relevante"
    ]
]

print("\n11) ranking final antes do anti-spam")
print("\nobs: aqui ja esta tudo normalizado e unido pela media ponderada")
print(tabela_antes.round(6).to_string(index=False))


# anti-spam
# no exemplo da aula, a pagina F é suspeita
# então ela perde 70% do score final

df_rank["spam_suspeito"] = df_rank["id"].eq("F").astype(int)

df_rank["score_final_seguro"] = (
    df_rank["score_final"] * (1 - 0.70 * df_rank["spam_suspeito"])
)

ranking_depois = df_rank.sort_values("score_final_seguro", ascending=False).reset_index(drop=True)
ranking_depois["posicao_depois"] = ranking_depois.index + 1
ranking_depois["top3_depois"] = ranking_depois["posicao_depois"] <= 3
ranking_depois["relevante"] = ranking_depois["id"].isin(relevantes)

tabela_depois = ranking_depois[
    [
        "posicao_depois",
        "id",
        "titulo",
        "score_final",
        "spam_suspeito",
        "score_final_seguro",
        "top3_depois",
        "relevante"
    ]
]

print("\n12) ranking final depois do anti-spam")
print(tabela_depois.round(6).to_string(index=False))


# precision@k
# precision@3 é a metrica principal
# precision@2 entra como extra porque nesse exemplo so existem 2 paginas relevantes

def precision_k(tabela, k, coluna_id="id"):
    topk = tabela.head(k)[coluna_id].tolist()
    acertos = sum(pagina in relevantes for pagina in topk)
    valor = acertos / k

    return topk, acertos, valor


linhas_precision = []

rankings_para_avaliar = [
    ("somente texto", ranking_texto),
    ("texto + links", ranking_antes),
    ("texto + links + anti-spam", ranking_depois)
]

for nome, tabela in rankings_para_avaliar:
    top3, acertos3, p3 = precision_k(tabela, 3)
    top2, acertos2, p2 = precision_k(tabela, 2)

    linhas_precision.append({
        "ranking": nome,
        "top3": ", ".join(top3),
        "relevantes_no_top3": acertos3,
        "precision@3": p3,
        "top2_extra": ", ".join(top2),
        "relevantes_no_top2": acertos2,
        "precision@2_extra": p2,
        "spam_no_top3": "F" in top3
    })

tabela_precision = pd.DataFrame(linhas_precision)

print("\n13) tabela de avaliacao")
print(tabela_precision.round(4).to_string(index=False))


# comparacao antes e depois
# mudanca positiva quer dizer que a pagina subiu no ranking
# mudanca negativa quer dizer que ela caiu no ranking

comparacao = ranking_antes[["id", "titulo", "posicao_antes"]].merge(
    ranking_depois[["id", "posicao_depois"]],
    on="id"
)

comparacao["mudanca"] = comparacao["posicao_antes"] - comparacao["posicao_depois"]

comparacao = comparacao.merge(
    df_rank[["id", "score_final", "spam_suspeito", "score_final_seguro"]],
    on="id"
)

comparacao = comparacao.sort_values("posicao_depois").reset_index(drop=True)

print("\n14) comparacao antes e depois do anti-spam")
print(comparacao.round(6).to_string(index=False))


# resumo final

top3_texto, _, p3_texto = precision_k(ranking_texto, 3)
top3_antes, _, p3_antes = precision_k(ranking_antes, 3)
top3_depois, _, p3_depois = precision_k(ranking_depois, 3)

top2_antes, _, p2_antes = precision_k(ranking_antes, 2)
top2_depois, _, p2_depois = precision_k(ranking_depois, 2)

print("\n15) resumo rapido")
print(f"Top 3 somente texto: {top3_texto} | Precision@3 = {p3_texto:.4f}")
print(f"Top 3 antes do anti-spam: {top3_antes} | Precision@3 = {p3_antes:.4f}")
print(f"Top 3 depois do anti-spam: {top3_depois} | Precision@3 = {p3_depois:.4f}")

print("\nextra para enxergar melhor a melhora nesse exemplo:")
print(f"Top 2 antes do anti-spam: {top2_antes} | Precision@2 = {p2_antes:.4f}")
print(f"Top 2 depois do anti-spam: {top2_depois} | Precision@2 = {p2_depois:.4f}")

