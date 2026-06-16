"""A tese vira invariante: o inciso isolado engana; a hierarquia o completa."""

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from demo import PERGUNTAS  # noqa: E402
from graphrag import NormGraph  # noqa: E402

g = NormGraph(ROOT / "data" / "lei_lgpd.json")
TRAP = "É preciso consentimento para tratar dados pessoais?"


def test_todo_inciso_tem_pai_artigo():
    """Integridade da hierarquia: todo inciso aponta para um artigo existente."""
    for n in g.nodes.values():
        if n.tipo == "inciso":
            assert n.parent in g.nodes
            assert g.nodes[n.parent].tipo == "artigo"


def test_cadeia_de_ancestrais():
    assert g.ancestors("art7_I") == ["art7"]
    assert g.ancestors("art7") == []


def test_busca_plana_perde_o_caput():
    """A busca plana traz o inciso do consentimento solto, sem o caput."""
    flat = g.retrieve_flat(TRAP, top_k=1)
    assert flat == ["art7_I"]
    assert g.has_caput(flat) is False


def test_graphrag_monta_a_unidade_completa():
    """O GraphRAG entrega o caput + todos os 10 incisos do Art. 7º."""
    unit = g.retrieve_graph(TRAP)
    assert "art7" in unit and g.has_caput(unit)
    incisos = [i for i in unit if g.nodes[i].tipo == "inciso"]
    assert len(incisos) == 10  # consentimento é UMA de dez hipóteses


def test_graphrag_e_superset_da_busca_plana():
    flat = set(g.retrieve_flat(TRAP, top_k=1))
    assert flat.issubset(set(g.retrieve_graph(TRAP)))


def test_completude_graphrag_vence():
    """Em todas as perguntas, o GraphRAG entrega o caput; a busca plana quase nunca."""
    flat_ok = sum(g.has_caput(g.retrieve_flat(q, 1)) for q in PERGUNTAS)
    graph_ok = sum(g.has_caput(g.retrieve_graph(q)) for q in PERGUNTAS)
    assert graph_ok == len(PERGUNTAS)
    assert flat_ok < graph_ok
