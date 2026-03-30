"""
Fase 3: Word Embeddings - Vetores e Contexto
Transforma palavras em vetores numéricos usando Word2Vec (Gensim).
Demonstra similaridade por cosseno e operações semânticas vetoriais.

Instalação necessária:
    pip install gensim
"""

import math
import random
from collections import defaultdict

# Tenta importar Gensim; se não disponível, usa implementação manual
try:
    from gensim.models import Word2Vec
    GENSIM_DISPONIVEL = True
except ImportError:
    GENSIM_DISPONIVEL = False
    print("[AVISO] Gensim não encontrado. Usando vetores manuais para demonstração.")
    print("        Para usar Word2Vec real: pip install gensim\n")


# =============================================================================
# CORPUS DE TREINAMENTO
# =============================================================================
SENTENCAS_CORPUS = [
    ["o", "rei", "governa", "o", "reino", "com", "sabedoria"],
    ["a", "rainha", "governa", "o", "reino", "com", "elegância"],
    ["o", "homem", "trabalha", "na", "cidade", "grande"],
    ["a", "mulher", "trabalha", "na", "cidade", "grande"],
    ["o", "rei", "é", "um", "homem", "poderoso"],
    ["a", "rainha", "é", "uma", "mulher", "poderosa"],
    ["o", "príncipe", "é", "filho", "do", "rei"],
    ["a", "princesa", "é", "filha", "da", "rainha"],
    ["o", "gato", "dorme", "no", "sofá", "confortável"],
    ["o", "cachorro", "late", "no", "jardim", "grande"],
    ["o", "gato", "e", "o", "cachorro", "são", "animais"],
    ["paris", "é", "a", "capital", "da", "france"],
    ["lisboa", "é", "a", "capital", "de", "portugal"],
    ["brasil", "tem", "sua", "capital", "em", "brasília"],
    ["o", "médico", "cuida", "dos", "pacientes", "no", "hospital"],
    ["o", "professor", "ensina", "os", "alunos", "na", "escola"],
    ["a", "professora", "ensina", "as", "alunas", "na", "escola"],
    ["o", "engenheiro", "projeta", "pontes", "e", "edifícios"],
    ["a", "engenheira", "projeta", "estruturas", "complexas"],
    ["o", "sol", "aquece", "a", "terra", "durante", "o", "dia"],
    ["a", "lua", "ilumina", "o", "céu", "durante", "a", "noite"],
    ["o", "livro", "contém", "conhecimento", "e", "história"],
    ["a", "biblioteca", "guarda", "muitos", "livros", "importantes"],
    ["computador", "processa", "dados", "e", "informações"],
    ["inteligência", "artificial", "aprende", "com", "dados"],
]


# =============================================================================
# IMPLEMENTAÇÃO MANUAL DE VETORES (fallback sem Gensim)
# =============================================================================
class VetoresSimples:
    """
    Implementação simplificada de word vectors baseada em co-ocorrência.
    Usada apenas quando Gensim não está disponível.
    """

    def __init__(self, tamanho_vetor: int = 10):
        self.tamanho_vetor = tamanho_vetor
        self.vocabulario = {}
        self.vetores = {}

    def treinar(self, sentencas: list, janela: int = 2):
        """Cria vetores baseados em co-ocorrência com semente aleatória determinística."""
        # Coleta vocabulário
        todas_palavras = [p for s in sentencas for p in s]
        vocab_unico = sorted(set(todas_palavras))
        self.vocabulario = {p: i for i, p in enumerate(vocab_unico)}

        # Inicializa vetores com semente determinística por palavra
        for palavra in vocab_unico:
            random.seed(hash(palavra) % (2**32))
            self.vetores[palavra] = [random.gauss(0, 0.1) for _ in range(self.tamanho_vetor)]

        # Ajusta vetores com base em co-ocorrência
        coocorrencia = defaultdict(lambda: defaultdict(int))
        for sentenca in sentencas:
            for i, palavra in enumerate(sentenca):
                inicio = max(0, i - janela)
                fim = min(len(sentenca), i + janela + 1)
                for j in range(inicio, fim):
                    if i != j:
                        coocorrencia[palavra][sentenca[j]] += 1

        # Ajusta dimensões com base na co-ocorrência
        for palavra, vizinhos in coocorrencia.items():
            for vizinho, freq in vizinhos.items():
                if palavra in self.vetores and vizinho in self.vetores:
                    for k in range(self.tamanho_vetor):
                        self.vetores[palavra][k] += freq * 0.01 * self.vetores[vizinho][k]

    def similaridade_cosseno(self, v1: list, v2: list) -> float:
        """Calcula a similaridade de cosseno entre dois vetores."""
        produto = sum(a * b for a, b in zip(v1, v2))
        norma1 = math.sqrt(sum(a ** 2 for a in v1))
        norma2 = math.sqrt(sum(b ** 2 for b in v2))
        if norma1 == 0 or norma2 == 0:
            return 0.0
        return produto / (norma1 * norma2)

    def palavras_similares(self, palavra: str, top_n: int = 5) -> list:
        """Retorna as N palavras mais similares."""
        if palavra not in self.vetores:
            return []
        vetor_alvo = self.vetores[palavra]
        similaridades = []
        for outra, vetor in self.vetores.items():
            if outra != palavra:
                sim = self.similaridade_cosseno(vetor_alvo, vetor)
                similaridades.append((outra, sim))
        return sorted(similaridades, key=lambda x: x[1], reverse=True)[:top_n]

    def analogia(self, pos1: str, neg1: str, pos2: str) -> str:
        """
        Resolve: pos1 - neg1 + pos2 = ?
        Exemplo clássico: Rei - Homem + Mulher = Rainha
        """
        if not all(p in self.vetores for p in [pos1, neg1, pos2]):
            return "Palavra não encontrada no vocabulário."

        resultado = [
            self.vetores[pos1][i] - self.vetores[neg1][i] + self.vetores[pos2][i]
            for i in range(self.tamanho_vetor)
        ]

        excluir = {pos1, neg1, pos2}
        melhor = None
        melhor_sim = -1

        for palavra, vetor in self.vetores.items():
            if palavra not in excluir:
                sim = self.similaridade_cosseno(resultado, vetor)
                if sim > melhor_sim:
                    melhor_sim = sim
                    melhor = palavra

        return melhor


# =============================================================================
# FUNÇÕES DE DEMONSTRAÇÃO
# =============================================================================
def demonstrar_similaridade(modelo, palavras_teste: list):
    """Demonstra similaridade semântica entre palavras."""
    print("\n[2] Similaridade Semântica (quanto maior, mais próximas semanticamente):")
    print("    " + "-" * 50)

    pares = [
        ("rei", "rainha"),
        ("homem", "mulher"),
        ("gato", "cachorro"),
        ("gato", "computador"),
        ("professor", "professora"),
        ("paris", "lisboa"),
    ]

    for p1, p2 in pares:
        try:
            if GENSIM_DISPONIVEL:
                sim = modelo.wv.similarity(p1, p2)
            else:
                v1 = modelo.vetores.get(p1)
                v2 = modelo.vetores.get(p2)
                if v1 and v2:
                    sim = modelo.similaridade_cosseno(v1, v2)
                else:
                    sim = 0.0
            barra = "█" * int(abs(sim) * 20)
            print(f"    {p1:<12} ↔ {p2:<12} {barra} {sim:.4f}")
        except KeyError:
            print(f"    {p1} ou {p2} não está no vocabulário.")


def demonstrar_palavras_similares(modelo, palavras: list, top_n: int = 4):
    """Mostra as palavras mais próximas de cada palavra-alvo."""
    print("\n[3] Palavras Mais Próximas no Espaço Vetorial:")
    print("    " + "-" * 50)

    for palavra in palavras:
        print(f"\n    Próximas de '{palavra}':")
        try:
            if GENSIM_DISPONIVEL:
                similares = modelo.wv.most_similar(palavra, topn=top_n)
            else:
                similares = modelo.palavras_similares(palavra, top_n=top_n)

            for sim_palavra, score in similares:
                barra = "▪" * int(abs(score) * 15)
                print(f"      {sim_palavra:<15} {barra} {score:.4f}")
        except KeyError:
            print(f"      '{palavra}' não encontrada no vocabulário.")


def demonstrar_analogia(modelo):
    """Demonstra operações vetoriais semânticas (o exemplo clássico do slide)."""
    print("\n[4] Operações Vetoriais (Analogias Semânticas):")
    print("    " + "-" * 50)

    analogias = [
        ("rei", "homem", "mulher", "rainha"),
        ("rei", "homem", "mulher", "rainha"),
        ("professor", "homem", "mulher", "professora"),
    ]

    for pos1, neg1, pos2, esperado in analogias:
        print(f"\n    {pos1} - {neg1} + {pos2} = ?  (esperado: '{esperado}')")
        try:
            if GENSIM_DISPONIVEL:
                resultado = modelo.wv.most_similar(
                    positive=[pos1, pos2], negative=[neg1], topn=1
                )
                pred = resultado[0][0]
                score = resultado[0][1]
            else:
                pred = modelo.analogia(pos1, neg1, pos2)
                score = None

            acertou = "✓" if pred == esperado else "≈"
            score_str = f"(score: {score:.4f})" if score else ""
            print(f"    Resultado: '{pred}' {acertou} {score_str}")
        except Exception as e:
            print(f"    Erro: {e}")


def main():
    print("=" * 60)
    print("  FASE 3: Word Embeddings")
    print("  Transformando Palavras em Vetores Numéricos")
    print("=" * 60)

    # --- Treinamento ---
    print("\n[1] Treinando modelo Word2Vec no corpus...")

    if GENSIM_DISPONIVEL:
        modelo = Word2Vec(
            sentences=SENTENCAS_CORPUS,
            vector_size=30,
            window=3,
            min_count=1,
            workers=1,
            epochs=200,
            seed=42,
        )
        vocab = list(modelo.wv.key_to_index.keys())
        print(f"    ✓ Modelo Gensim Word2Vec treinado!")
        print(f"    Vocabulário: {len(vocab)} palavras")
        print(f"    Tamanho do vetor: {modelo.wv.vector_size} dimensões")

        # Mostra um vetor como exemplo
        print(f"\n    Exemplo — vetor da palavra 'rei' (30 dimensões):")
        vetor_rei = modelo.wv['rei']
        print(f"    {[round(v, 3) for v in vetor_rei[:8]]} ...")
    else:
        modelo = VetoresSimples(tamanho_vetor=20)
        modelo.treinar(SENTENCAS_CORPUS, janela=3)
        print(f"    ✓ Vetores manuais criados (demonstração simplificada).")
        print(f"    Vocabulário: {len(modelo.vocabulario)} palavras")

    # --- Demonstrações ---
    demonstrar_similaridade(modelo, [])
    demonstrar_palavras_similares(modelo, ["rei", "gato", "professor"])
    demonstrar_analogia(modelo)

    # --- Explicação Conceitual ---
    print("\n" + "=" * 60)
    print("  CONCEITO: O Espaço Vetorial Semântico")
    print("=" * 60)
    print("""
  Cada palavra é um PONTO em um espaço de N dimensões.
  Palavras com contextos semelhantes ficam PRÓXIMAS.

  A Similaridade de Cosseno mede o ângulo entre dois vetores:
    - cos(θ) = 1.0  →  Palavras idênticas ou sinônimas
    - cos(θ) = 0.0  →  Palavras sem relação
    - cos(θ) = -1.0 →  Palavras opostas (antônimos)

  A "mágica" do Word2Vec: Rei - Homem + Mulher ≈ Rainha
  Isso mostra que o modelo captura RELAÇÕES SEMÂNTICAS
  codificadas como direções no espaço vetorial.

  Limitação: o vetor de uma palavra é FIXO (estático).
  "Banco" (financeiro) e "banco" (para sentar) têm o mesmo vetor!
  Isso será resolvido pelos Transformers com atenção contextual.
""")


if __name__ == "__main__":
    main()
