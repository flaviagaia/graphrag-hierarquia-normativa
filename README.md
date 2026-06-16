# GraphRAG na Hierarquia Normativa

[🇧🇷 Português](#-português) · [🇺🇸 English](#-english)

Python 3.10+ · scikit-learn · 100% offline, sem API key · MIT License

Dados públicos: Lei nº 13.709/2018 (LGPD), arts. 6º e 7º, texto do Planalto/gov.br.

> **Em uma frase:** num RAG jurídico, recuperar um inciso solto ("I - mediante
> consentimento") leva o modelo à resposta errada, porque o sentido está no caput
> que o introduz. Modelar a lei como árvore e subir do inciso até o artigo entrega
> a unidade normativa completa. No experimento: **contexto completo em 6/6 vs 1/6.**

---

## 🇧🇷 Português

### O problema, em concreto

Texto de lei é uma árvore: artigo → caput → incisos → alíneas. O significado de um
inciso depende do caput que o introduz. Veja o Art. 7º da LGPD:

```
Art. 7º  O tratamento de dados pessoais somente poderá ser realizado nas
         seguintes hipóteses:
         I  - mediante o fornecimento de consentimento pelo titular;
         II - para o cumprimento de obrigação legal ...;
         ... (até o inciso X)
```

Pergunta: "é preciso consentimento para tratar dados?". A busca vetorial encontra o
trecho mais parecido, o inciso I, e o entrega sozinho. Lido isolado, ele sugere que
**consentimento é obrigatório**. O caput diz o contrário: consentimento é **uma de
dez** hipóteses. O RAG recuperou o trecho certo e produziria a resposta errada.

### Como funciona (o técnico)

A lei é carregada como um **grafo dirigido** de nós tipados com uma aresta `parent`
(cada inciso aponta para o seu artigo):

```
Node{ id, tipo ∈ {artigo, inciso}, parent, rotulo, texto }
```

Duas recuperações sobre o mesmo índice TF-IDF:

```
retrieve_flat(q):                 # busca vetorial pura
    retorna o nó de maior similaridade (um fragmento solto)

retrieve_graph(q):                # GraphRAG hierárquico
    leaf := retrieve_flat(q)
    raiz := sobe a cadeia parent até o artigo            # expansão por ancestral
    retorna [raiz] + filhos(raiz)  ordenados pelo texto  # caput + incisos irmãos
```

Não é re-ranking (não muda a pontuação). É **reconstrução de contexto pela
estrutura do documento**: a aresta `parent` recupera a unidade normativa inteira
(caput + irmãos), nem o fragmento solto, nem a lei inteira. Complexidade: subir a
hierarquia é O(altura), montar a unidade é O(nº de filhos do artigo).

### Resultado real deste repositório

Seis perguntas cuja resposta mora num inciso (LGPD, arts. 6º e 7º):

| | Busca plana | GraphRAG |
| --- | ----------- | -------- |
| Contexto com o caput (completo) | **1 / 6** | **6 / 6** |
| Nós no contexto | 1 (fragmento) | unidade do artigo (caput + incisos) |

### A armadilha, lado a lado

```
[busca plana]  -> Art. 7º, I: mediante consentimento do titular
                  (lido sozinho: "consentimento é obrigatório"  ✗ enganoso)

[graphrag]     -> Art. 7º (caput): ... nas seguintes hipóteses:
                  I consentimento | II obrigação legal | ... | X proteção do crédito
                  ("consentimento é 1 de 10 hipóteses"          ✓ correto)
```

### Como explicar em 30 segundos

Um inciso fora do artigo é como uma resposta sem a pergunta. "Consentimento" só faz
sentido depois de "o tratamento só pode ocorrer nestas hipóteses:". O grafo guarda
essa relação de pai e filho e, quando acha o filho, traz o pai junto.

### Por que não só aumentar o chunk?

Pegar o artigo inteiro num chunk fixo parece resolver, mas quebra em artigos
gigantes, incisos longos e remissões ("observadas as disposições do Capítulo IV").
A hierarquia recupera a unidade certa e ainda permite navegar pelas remissões, o
que um chunk fixo não faz.

### Execução

```
pip install -r requirements.txt
python demo.py            # busca plana vs GraphRAG, com números reais
pytest tests/ -v          # 6 testes (integridade da hierarquia, completude, armadilha)
```

### Estrutura

```
data/lei_lgpd.json   # arts. 6º e 7º da LGPD como grafo (nó + aresta parent)
graphrag.py          # NormGraph: ancestrais, filhos, busca plana e expansão
demo.py              # a comparação, com a armadilha do consentimento
tests/               # um invariante por lição
```

### Limitações honestas

Excerto de dois artigos, hierarquia de dois níveis (artigo → inciso). A norma real
tem parágrafos, alíneas, itens e remissões entre artigos e leis; um grafo de
produção precisa desses tipos de nó e das arestas de remissão. Recuperação lexical
(TF-IDF); com embeddings o efeito da hierarquia é o mesmo. O objetivo é isolar
**por que** a estrutura importa.

---

## 🇺🇸 English

**In one line:** in a legal RAG, retrieving a loose item ("I - upon consent") leads
the model to the wrong answer, because the meaning lives in the caput that
introduces it. Model the law as a tree and walk from the item up to the article to
return the full normative unit. Result: **complete context in 6/6 vs 1/6.**

### The problem

Legal text is a tree: article → caput → items. An item's meaning depends on its
caput. Vector search retrieves the closest fragment (the item) and serves it loose;
read alone, LGPD Art. 7 item I suggests consent is mandatory, while the caput says
it is one of ten legal bases.

### How it works (technical)

The law loads as a typed directed graph with a `parent` edge. `retrieve_flat`
returns the top TF-IDF node (a fragment); `retrieve_graph` walks the `parent` chain
up to the article and returns `[article] + children` (caput + sibling items). Not
re-ranking; context reconstruction from document structure. Cost: O(height) to walk
up, O(#children) to assemble.

### Real result

Across six item-targeting questions, the caput is in context in **1/6** for flat
search and **6/6** for GraphRAG. The consent trap: flat hands the LLM only "item I",
GraphRAG hands the whole Art. 7 (consent is 1 of 10).

### Explain it in 30 seconds

An item without its article is an answer without the question. The graph keeps the
parent–child link and, when it finds the child, brings the parent along.

### Running

```
pip install -r requirements.txt
python demo.py
pytest tests/ -v          # 6 tests
```

---

## Referências científicas (crédito aos autores)

- Edge et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. Microsoft Research, arXiv:2404.16130.
- Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS.
- Günther et al. (2024). Late Chunking. Jina AI, arXiv:2409.04701.
- LGPD (Lei nº 13.709/2018), arts. 6º e 7º: ato oficial de domínio público (Planalto/gov.br).

Este repositório é uma reimplementação didática e offline dessas ideias.

---

Part of my LinkedIn series on RAG efficiency → [Flávia Gaia](https://www.linkedin.com/in/flavia-gaia/)
