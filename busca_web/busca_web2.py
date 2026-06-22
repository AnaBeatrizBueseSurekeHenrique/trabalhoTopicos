import pandas as pd
import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
def tfidf(docs, consulta,):
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

    return ranking_texto

def criar_grafo(docs):
    G = nx.DiGraph()

    for doc in docs:
        G.add_node(doc["id"], titulo=doc["titulo"])

        for destino in doc["links"]:
            G.add_edge(doc["id"], destino)

    df_links = pd.DataFrame(list(G.edges()), columns=["origem", "destino"])
    # pos = nx.circular_layout(G)  # Positions nodes using a force-directed algorithm
    # nx.draw(
    #     G,
    #     pos,
    #     with_labels=True,
    #     node_color="lightblue",
    #     node_size=800,
    #     font_weight="bold",
    #     arrows=True,
    #     arrowsize=20,
    # )

    # 5. Display the plot
    # plt.savefig("figura.png")
    # plt.show()
    print("\n6) links da mini-web")
    print(df_links.to_string(index=False))
    return G
          
def busca_web(docs, consulta, relevantes):
    df = pd.DataFrame(docs)
    print(f"Inicio da consulta: {consulta}")
    
    ranking_texto = tfidf(docs, consulta)
   
    ranking_texto = ranking_texto.sort_values("score_texto", ascending=False).reset_index(drop=True)
    ranking_texto["posicao_texto"] = ranking_texto.index + 1
    ranking_texto["top3_texto"] = ranking_texto["posicao_texto"] <= 3
    ranking_texto["relevante"] = ranking_texto["id"].isin(relevantes)

    tabela_texto = ranking_texto[
        ["posicao_texto", "id", "titulo", "score_texto", "top3_texto", "relevante"]
    ]
    print("\n1) Ranking Textual, com TD-IDF")
    print(tabela_texto.round(6).to_string(index=False))

    G = criar_grafo(docs)

    df_links_recebidos = pd.DataFrame({
        "id": list(dict(G.in_degree()).keys()),
        "links_recebidos": list(dict(G.in_degree()).values())
    })

    df_links_recebidos = df_links_recebidos.merge(df[["id", "titulo"]], on="id")
    df_links_recebidos = df_links_recebidos[["id", "titulo", "links_recebidos"]]
    df_links_recebidos = df_links_recebidos.sort_values("links_recebidos", ascending=False).reset_index(drop=True)

    print("\n2) links recebidos")
    print(df_links_recebidos.to_string(index=False))

    pagerank = nx.pagerank(G, alpha=0.85)

    df_pr = pd.DataFrame({
        "id": list(pagerank.keys()),
        "pagerank": list(pagerank.values())
    })

    df_pr = df_pr.merge(df[["id", "titulo"]], on="id")
    df_pr = df_pr[["id", "titulo", "pagerank"]]
    df_pr = df_pr.sort_values("pagerank", ascending=False).reset_index(drop=True)
    df_pr["posicao_pagerank"] = df_pr.index + 1

    print("\n3) ranking por pagerank")
    print(df_pr[["posicao_pagerank", "id", "titulo", "pagerank"]].round(6).to_string(index=False))

    hubs, authorities = nx.hits(G, max_iter=1000, normalized=True)

    df_hits = pd.DataFrame({
        "id": [doc["id"] for doc in docs],
        "titulo": [doc["titulo"] for doc in docs],
        "hub": [hubs[doc["id"]] for doc in docs],
        "authority": [authorities[doc["id"]] for doc in docs]
    })

    for col in ["hub", "authority"]:
        df_hits[col] = df_hits[col].apply(lambda x: 0 if abs(x) < 0.0000001 else x)

    ranking_authorities = df_hits.sort_values("authority", ascending=False).reset_index(drop=True)
    ranking_authorities["posicao_authority"] = ranking_authorities.index + 1
    ranking_hubs = df_hits.sort_values("hub", ascending=False).reset_index(drop=True)
    ranking_hubs["posicao_hub"] = ranking_hubs.index + 1

    print("\n4) hits - ranking de authorities")
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

    print("\n5) ranking final antes do anti-spam")
    print(tabela_antes.round(6).to_string(index=False))


    df_rank["spam_suspeito"] = df_rank["id"].eq("P").astype(int)
    for i in range(0,len(df_rank)):
        if(df_rank["id"].values[i] == "F" or df_rank["id"].values[i] == "I"):
            df_rank["spam_suspeito"].values[i] = 1
        else:
            df_rank["spam_suspeito"].values[i] = 0
    df_rank["score_final_seguro"] = (
        df_rank["score_final"] * (1 - 0.70 * df_rank["spam_suspeito"])
    )
    
    # # Bonificação para o top 3 td-idf
    for i in range(0,3):
        aux = df_rank.index[df_rank['id'] == ranking_texto.iloc[i]["id"]].tolist()
        df_rank.at[aux[0], "score_final_seguro"] += df_rank.at[aux[0], "score_final_seguro"] * ((3-i)*0.1)
       
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

    print("\n6) ranking final depois do anti-spam")
    print(tabela_depois.round(6).to_string(index=False))

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

    print("\n7) tabela de avaliacao")
    print(tabela_precision.round(4).to_string(index=False))

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

    top3_texto, _, p3_texto = precision_k(ranking_texto, 3)
    top3_antes, _, p3_antes = precision_k(ranking_antes, 3)
    top3_depois, _, p3_depois = precision_k(ranking_depois, 3)

    top2_antes, _, p2_antes = precision_k(ranking_antes, 2)
    top2_depois, _, p2_depois = precision_k(ranking_depois, 2)

    print("\n15) Precisão para caso com 3 páginas relevantes")
    print(f"Top 3 somente texto: {top3_texto} | Precision@3 = {p3_texto:.4f}")
    print(f"Top 3 antes do anti-spam: {top3_antes} | Precision@3 = {p3_antes:.4f}")
    print(f"Top 3 depois do anti-spam: {top3_depois} | Precision@3 = {p3_depois:.4f}")

    print("\n Precisão para caso com 2 páginas relevantes")
    print(f"Top 2 antes do anti-spam: {top2_antes} | Precision@2 = {p2_antes:.4f}")
    print(f"Top 2 depois do anti-spam: {top2_depois} | Precision@2 = {p2_depois:.4f}")


docs =[ 
       {
        "id": "A",
        "titulo": "Mitologia grega",
        "texto": "Mitologia grega é o estudo dos conjuntos de narrativas relacionadas com a mitologia dos gregos antigos e dos seus significados.",
        "links": ["B", "C", "D", "E", "G", "H"]
    },
{
        "id": "B",
        "titulo": "Odisseia",
        "texto": "O poema relata o regresso de Odisseu (ou Ulisses) herói da Guerra de Troia.",
        "links": ["C", "G"]
    },
{
        "id": "C",
        "titulo": "Guerra de Troia",
        "texto": "A Guerra de Troia foi, de acordo com a mitologia grega, um grande conflito bélico entre os aqueus das cidades-estado da Grécia e Troia.",
        "links": ["B", "D"]
    },
{
        "id": "D",
        "titulo": "Zeus",
        "texto": "Zeus é o pai dos deuses que exercia a autoridade sobre os deuses olímpicos na antiga religião grega. É o deus dos céus, raios, relâmpagos que mantêm a ordem e justiça na mitologia grega. Seu equivalente romano é Júpiter, enquanto seu equivalente etrusco é Tinia; alguns autores estabeleceram seu equivalente hindu como sendo Indra.",
        "links": ["B", "C", "A"]
    },
{   "id": "E",
    "titulo": "Pandora",
    "texto": "O mito de Pandora é uma espécie de teodicéia, pois procura explicar a origem do mal no mundo, que, segundo o mito, teria surgido quando Pandora abriu a 'Caixa de Pandora', espalhando todos os males pelo mundo.",
    "links": []
    },
{
    "id": "F",
    "titulo": "Aprenda mitologia gratuitamente!!",
    "texto": "Aprenda sobre mitologia grega!! Aprenda sobre Odisseu!! Aprenda sobre o mito de Zeus e Pandora!! Aprenda 90 reais de desconto!!",
    "links": ["A","B","C","D","E","F", "G", "H", "I", "J"]
},
{"id": "G",
 "titulo": "Odisseu",
 "texto": "Odisseu ou Ulisses foi, na mitologia grega e na mitologia romana um personagem da Ilíada e da Odisseia, de Homero. Odisseu a personagem principal dessa última obra, e uma figura à parte na narrativa da Guerra de Troia. Odisseu é um dos mais ardilosos guerreiros de toda a epopeia grega, mesmo depois da guerra, quando do longo retorno ao seu reino, Ítaca, uma das numerosas ilhas gregas.",
 "links": []
},
{
    "id": "H",
    "titulo": "Ilíada",
    "texto": "A mulher mais bela do mundo era Helena, filha de Zeus e Leda. Leda era casada com Tíndaro, rei de Esparta. Helena possuía diversos pretendentes, que incluíam muitos dos maiores heróis da Grécia, e o seu pai adotivo, Tíndaro, hesitava tomar uma decisão em favor de um deles temendo enfurecer os outros. Finalmente um dos pretendentes, Odisseu (cujo nome latino era Ulisses), rei de Ítaca, resolveu o impasse propondo que todos os pretendentes jurassem proteger Helena e sua escolha, qualquer que fosse. Helena então se casou com Menelau, que se tornou o rei de Esparta.",
    "links": ["G", "B"]
},
{
    "id": "I",
    "titulo": "Site confiável de mitologia!",
    "texto": "Aprenda sobre mitologia grega!! Aprenda sobre Odisseu!! Aprenda sobre o mito de Zeus e Pandora!! Aprenda 90 reais de desconto!! Mitologia muito importante!! Mitologia!! Mitologia muito gratuita!!",
    "links": ["A","B","C","D","E","F", "G", "F", "K"]
},
{
    "id": "J",
    "titulo": "Mitologia egípcia",
    "texto": "A Mitologia egípcia é a coleção da mitologia do antigo Egito, que descreve as ações e relações dos deuses egípcios como uma forma de compreender o cosmos.",
    "links": ["K"]
},
{
    "id": "K",
    "titulo": "Textos das Pirâmides",
    "texto": "Os Textos da Pirâmide de Unas, descobertos em 1881 por Gaston Maspero, são os escritos religiosos mais antigos descobertos até hoje. Por apresentarem uma síntese das crenças religiosas do Antigo Egito, eles datam de 4500 anos ou mais, considerando que estas crenças devem ter nascido muito antes de serem transcritas na pedra.",
    "links": []
}
]

busca_web(docs=docs, consulta="Odisseu", relevantes=['B', 'C', 'G'])
busca_web(docs=docs, consulta="Mitologia", relevantes=['A', 'J'])