"""Demo: busca plana vs GraphRAG hierárquico sobre a LGPD (~1s).

    python demo.py
"""

from __future__ import annotations

from pathlib import Path

from graphrag import NormGraph

ROOT = Path(__file__).parent

# Perguntas cuja resposta mora num inciso, mas que só fazem sentido com o caput.
PERGUNTAS = [
    "É preciso consentimento para tratar dados pessoais?",
    "Posso tratar dados para cumprir uma obrigação legal?",
    "Dados podem ser tratados para proteger a vida do titular?",
    "Posso usar dados pessoais para proteção do crédito?",
    "O que exige o princípio da necessidade?",
    "O tratamento precisa observar segurança da informação?",
]


def main() -> None:
    g = NormGraph(ROOT / "data" / "lei_lgpd.json")
    print("=" * 70)
    print("GraphRAG na hierarquia normativa (LGPD, arts. 6º e 7º)")
    print("=" * 70)

    flat_ok = graph_ok = 0
    for q in PERGUNTAS:
        flat = g.retrieve_flat(q, top_k=1)
        graph = g.retrieve_graph(q)
        flat_caput = g.has_caput(flat)
        graph_caput = g.has_caput(graph)
        flat_ok += flat_caput
        graph_ok += graph_caput
        print(f"\nP: {q}")
        print(f"   plano   -> {g.nodes[flat[0]].rotulo:<12} | caput no contexto? "
              f"{'sim' if flat_caput else 'NÃO'} | {len(flat)} nó(s)")
        print(f"   graphrag-> {g.nodes[graph[0]].rotulo} + incisos "
              f"| caput no contexto? {'sim' if graph_caput else 'não'} | {len(graph)} nós")

    n = len(PERGUNTAS)
    print("\n" + "=" * 70)
    print(f"Contexto normativo completo (com o caput): plano {flat_ok}/{n} | "
          f"graphrag {graph_ok}/{n}")

    # O caso-armadilha, em detalhe
    print("\n" + "-" * 70)
    print("ARMADILHA: 'É preciso consentimento para tratar dados pessoais?'")
    print("-" * 70)
    trap = "É preciso consentimento para tratar dados pessoais?"
    flat = g.retrieve_flat(trap, top_k=1)
    print("\n[busca plana] o LLM recebe só:")
    print("  " + g.context_text(flat))
    print("  => lido sozinho, sugere que consentimento é obrigatório. ENGANOSO.")
    print("\n[graphrag] o LLM recebe a unidade inteira:")
    print(g.context_text(g.retrieve_graph(trap)))
    print("\n  => o caput mostra que consentimento é UMA de dez hipóteses. CORRETO.")


if __name__ == "__main__":
    main()
