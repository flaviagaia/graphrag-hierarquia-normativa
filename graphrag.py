"""GraphRAG sobre a hierarquia normativa (artigo -> inciso).

A lei é uma árvore: o artigo tem um caput que introduz uma lista de incisos. Um
inciso, sozinho, é incompleto. "I - mediante o fornecimento de consentimento pelo
titular" não diz que consentimento é UMA de dez hipóteses; quem diz isso é o caput
("O tratamento ... somente poderá ser realizado nas seguintes hipóteses:").

A busca vetorial recupera o fragmento mais parecido (o inciso) e o entrega solto.
O GraphRAG usa a aresta de hierarquia (`parent`) para EXPANDIR o resultado com os
ancestrais e os irmãos, montando a unidade normativa completa antes de mandar ao
LLM. Não é re-ranking; é reconstrução de contexto pela estrutura do documento.
"""

from __future__ import annotations

import json
import unicodedata
from dataclasses import dataclass
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def _norm(t: str) -> str:
    t = unicodedata.normalize("NFKD", t.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    return " ".join(t.split())


@dataclass(frozen=True)
class Node:
    id: str
    tipo: str
    parent: str | None
    rotulo: str
    texto: str


class NormGraph:
    def __init__(self, path: Path) -> None:
        data = json.loads(path.read_text(encoding="utf-8"))
        self.nodes: dict[str, Node] = {
            n["id"]: Node(n["id"], n["tipo"], n["parent"], n["rotulo"], n["texto"])
            for n in data["nodes"]
        }
        self._ids = list(self.nodes)
        self._vec = TfidfVectorizer(ngram_range=(1, 2), strip_accents="unicode")
        self._matrix = self._vec.fit_transform(self.nodes[i].texto for i in self._ids)

    # ---- relações de hierarquia ----
    def ancestors(self, node_id: str) -> list[str]:
        """Cadeia de ancestrais, do pai para a raiz."""
        out, cur = [], self.nodes[node_id].parent
        while cur is not None:
            out.append(cur)
            cur = self.nodes[cur].parent
        return out

    def children(self, node_id: str) -> list[str]:
        return [i for i, n in self.nodes.items() if n.parent == node_id]

    # ---- recuperação ----
    def retrieve_flat(self, query: str, top_k: int = 1) -> list[str]:
        """Busca vetorial pura: devolve os top_k nós mais parecidos, soltos."""
        scores = cosine_similarity(self._vec.transform([query]), self._matrix).ravel()
        order = scores.argsort()[::-1][:top_k]
        return [self._ids[i] for i in order]

    def retrieve_graph(self, query: str) -> list[str]:
        """GraphRAG: pega o melhor nó e expande pela hierarquia, montando a
        unidade normativa completa (caput + todos os incisos, na ordem do texto).
        """
        best = self.retrieve_flat(query, top_k=1)[0]
        # raiz da unidade: o artigo ancestral (ou o próprio nó, se já for artigo)
        cadeia = self.ancestors(best)
        raiz = cadeia[-1] if cadeia else best
        unidade = [raiz] + self.children(raiz)
        # ordena na ordem original do documento
        return sorted(unidade, key=self._ids.index)

    # ---- montagem de contexto ----
    def context_text(self, node_ids: list[str]) -> str:
        return "\n".join(f"{self.nodes[i].rotulo}: {self.nodes[i].texto}" for i in node_ids)

    def has_caput(self, node_ids: list[str]) -> bool:
        """O contexto inclui o artigo (caput) que dá sentido aos incisos?"""
        return any(self.nodes[i].tipo == "artigo" for i in node_ids)
