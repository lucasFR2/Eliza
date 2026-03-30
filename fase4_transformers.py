"""
Fase 4: Transformers e Atenção - A Revolução de 2017
Usa a biblioteca Hugging Face para rodar GPT-2 (geração de texto)
e BERT (análise de sentimento e preenchimento de máscara).

Instalação necessária:
    pip install transformers torch sentencepiece
    ou
    pip install transformers tensorflow sentencepiece
"""

# =============================================================================
# VERIFICAÇÃO DE DEPENDÊNCIAS
# =============================================================================
try:
    from transformers import (
        pipeline,
        GPT2LMHeadModel,
        GPT2Tokenizer,
        AutoTokenizer,
        AutoModelForSequenceClassification,
    )
    HF_DISPONIVEL = True
except ImportError:
    HF_DISPONIVEL = False


def verificar_dependencias():
    """Verifica e orienta sobre instalação das dependências."""
    if not HF_DISPONIVEL:
        print("=" * 60)
        print("  [AVISO] Hugging Face Transformers não encontrado.")
        print("=" * 60)
        print("""
  Para rodar esta fase, instale as dependências:

  Com PyTorch (recomendado):
    pip install transformers torch

  Com TensorFlow:
    pip install transformers tensorflow

  Após instalar, execute este script novamente.

  Na primeira execução, os modelos serão baixados
  automaticamente (~500MB para GPT-2 e BERT).
        """)
        return False
    return True


# =============================================================================
# DEMONSTRAÇÃO: GERAÇÃO DE TEXTO COM GPT-2
# =============================================================================
def demo_geracao_gpt2():
    """Demonstra geração de texto com GPT-2."""
    print("\n" + "=" * 60)
    print("  [1] GERAÇÃO DE TEXTO — GPT-2")
    print("=" * 60)
    print("  Carregando modelo GPT-2... (pode demorar na 1ª vez)")

    gerador = pipeline(
        "text-generation",
        model="gpt2",
        tokenizer="gpt2",
    )

    prompts = [
        "The future of artificial intelligence",
        "Once upon a time in a kingdom",
        "Scientists discovered that",
    ]

    print("\n  Gerando continuações de texto:\n")
    for prompt in prompts:
        resultado = gerador(
            prompt,
            max_new_tokens=40,
            num_return_sequences=1,
            temperature=0.8,
            do_sample=True,
            pad_token_id=50256,
        )
        texto = resultado[0]["generated_text"]
        print(f"  Prompt: '{prompt}'")
        print(f"  Gerado: {texto}")
        print()


# =============================================================================
# DEMONSTRAÇÃO: PREENCHIMENTO DE MÁSCARA COM BERT
# =============================================================================
def demo_fill_mask_bert():
    """Demonstra o preenchimento de máscara com BERT."""
    print("\n" + "=" * 60)
    print("  [2] PREENCHIMENTO DE MÁSCARA — BERT")
    print("=" * 60)
    print("  Carregando modelo BERT... (pode demorar na 1ª vez)")

    preenchedor = pipeline(
        "fill-mask",
        model="bert-base-uncased",
    )

    frases_mascara = [
        "The [MASK] is the capital of France.",
        "The cat sat on the [MASK].",
        "She is a [MASK] at the hospital.",
        "The [MASK] revolves around the sun.",
    ]

    print("\n  Prevendo palavras mascaradas ([MASK]):\n")
    for frase in frases_mascara:
        resultados = preenchedor(frase, top_k=3)
        print(f"  Frase: '{frase}'")
        for r in resultados:
            token = r["token_str"]
            score = r["score"]
            barra = "█" * int(score * 20)
            print(f"    [{token:<12}] {barra} {score:.4f}")
        print()


# =============================================================================
# DEMONSTRAÇÃO: ANÁLISE DE SENTIMENTO
# =============================================================================
def demo_sentimento():
    """Demonstra análise de sentimento com modelo pré-treinado."""
    print("\n" + "=" * 60)
    print("  [3] ANÁLISE DE SENTIMENTO — BERT Fine-tuned")
    print("=" * 60)
    print("  Carregando modelo de sentimento...")

    analisador = pipeline(
        "sentiment-analysis",
        model="distilbert-base-uncased-finetuned-sst-2-english",
    )

    textos = [
        "I absolutely love this product! It's amazing.",
        "This was a terrible experience. I'm very disappointed.",
        "The movie was okay, nothing special.",
        "Best day of my life! Everything went perfectly.",
        "I don't know how I feel about this.",
    ]

    print("\n  Classificando sentimento de frases:\n")
    for texto in textos:
        resultado = analisador(texto)[0]
        label = resultado["label"]
        score = resultado["score"]
        emoji = "😊" if label == "POSITIVE" else "😞"
        barra = "█" * int(score * 20)
        print(f"  {emoji} [{label:<8}] {barra} {score:.4f}")
        print(f"     \"{texto}\"")
        print()


# =============================================================================
# DEMONSTRAÇÃO: MECANISMO DE ATENÇÃO (CONCEITUAL)
# =============================================================================
def demo_atencao_conceitual():
    """
    Demonstra conceitualmente como o mecanismo de atenção funciona,
    sem dependências externas.
    """
    print("\n" + "=" * 60)
    print("  [4] MECANISMO DE AUTO-ATENÇÃO (Self-Attention)")
    print("=" * 60)

    frase = ["O", "banco", "aprovou", "o", "empréstimo", "do", "cliente"]

    print(f"\n  Frase: {' '.join(frase)}")
    print("""
  O Self-Attention calcula, para cada token, QUANTO ele deve
  "prestar atenção" em cada outro token da sequência.

  Exemplo simplificado de pesos de atenção para "banco":
  (valores ilustrativos — em modelos reais são aprendidos)
    """)

    # Pesos de atenção ilustrativos para a palavra "banco" (índice 1)
    pesos_atencao = {
        "O":          0.05,
        "banco":      0.25,
        "aprovou":    0.18,
        "o":          0.05,
        "empréstimo": 0.22,
        "do":         0.07,
        "cliente":    0.18,
    }

    for token, peso in pesos_atencao.items():
        barra = "█" * int(peso * 60)
        destaque = " ← foco principal" if peso >= 0.20 else ""
        print(f"  {token:<14} {barra} {peso:.2f}{destaque}")

    print("""
  Interpretação:
  - "banco" presta mais atenção em si mesmo, "empréstimo"
    e "aprovou" — palavras que definem SEU CONTEXTO.
  - Em "O banco ficou sentado no parque", os pesos seriam
    diferentes, apontando para "parque" e "sentado".
  - Isso é o que torna os embeddings contextuais (≠ Word2Vec)!
    """)


# =============================================================================
# COMPARATIVO FINAL: ERA SIMBÓLICA → TRANSFORMERS
# =============================================================================
def exibir_comparativo():
    """Exibe o comparativo de abordagens do slide 10."""
    print("\n" + "=" * 60)
    print("  COMPARATIVO FINAL DE ABORDAGENS")
    print("=" * 60)

    cabecalho = f"  {'Atributo':<18} {'ELIZA':<22} {'N-Grams':<22} {'Transformers':<20}"
    print(cabecalho)
    print("  " + "-" * 80)

    linhas = [
        ("Lógica",        "Regex / Regras",    "Frequência/Prob.",  "Vetor / Atenção"),
        ("Contexto",      "Nenhum",            "Curto (2-3 pal.)",  "Longo e Dinâmico"),
        ("Flexibilidade", "Rígida",            "Moderada",          "Extrema"),
        ("Semântica",     "Não",               "Não",               "Sim (contextual)"),
        ("Paralelismo",   "Sim (regras)",      "Limitado",          "Total (GPU)"),
        ("Treinamento",   "Manual",            "Corpus",            "Auto-supervisionado"),
    ]

    for atributo, eliza, ngram, transformer in linhas:
        print(f"  {atributo:<18} {eliza:<22} {ngram:<22} {transformer:<20}")

    print("""
  CONCLUSÃO SOBRE RLHF (Reinforcement Learning from Human Feedback):
  
  O GPT-2 e BERT são modelos BASE — eles aprendem a estrutura
  da linguagem, mas sem valores ou utilidade alinhada ao humano.
  
  O RLHF (usado no ChatGPT, Claude, etc.) é a etapa FINAL que:
  1. Coleta preferências humanas sobre respostas do modelo.
  2. Treina um modelo de recompensa com base nessas preferências.
  3. Usa Reinforcement Learning para ajustar o LLM base.
  
  Resultado: modelos que não apenas ENTENDEM linguagem, mas
  que respondem de forma ÚTIL, SEGURA e HONESTA.
  Sem RLHF, o modelo pode gerar textos fluentes mas prejudiciais.
""")


# =============================================================================
# MODO INTERATIVO COM GPT-2
# =============================================================================
def modo_interativo_gpt2():
    """Permite ao usuário interagir com GPT-2 para geração de texto."""
    print("=" * 60)
    print("  MODO INTERATIVO — Geração com GPT-2")
    print("=" * 60)
    print("  Digite um início de frase e pressione Enter.")
    print("  (Digite 'sair' para encerrar)\n")

    gerador = pipeline(
        "text-generation",
        model="gpt2",
        pad_token_id=50256,
    )

    while True:
        try:
            prompt = input("  Você: ").strip()
            if prompt.lower() in {"sair", "exit", "quit"}:
                print("  Encerrando modo interativo.")
                break
            if not prompt:
                continue

            resultado = gerador(
                prompt,
                max_new_tokens=50,
                num_return_sequences=1,
                temperature=0.9,
                do_sample=True,
            )
            print(f"  GPT-2: {resultado[0]['generated_text']}\n")

        except KeyboardInterrupt:
            print("\n  Encerrando.")
            break


# =============================================================================
# MAIN
# =============================================================================
def main():
    print("=" * 60)
    print("  FASE 4: Transformers e Atenção")
    print("  A Revolução de 2017 — 'Attention is All You Need'")
    print("=" * 60)

    print("""
  Arquitetura Transformer (Vaswani et al., 2017):
  
  ┌─────────────────────────────────────────────┐
  │  Input → Tokenização → Embeddings           │
  │          ↓                                  │
  │  [Multi-Head Self-Attention] × N camadas    │
  │          ↓                                  │
  │  Feed-Forward Neural Network                │
  │          ↓                                  │
  │  Output (geração / classificação)           │
  └─────────────────────────────────────────────┘
  
  Vantagem sobre RNNs: processa TODOS os tokens em paralelo,
  permitindo capturar dependências de longa distância.
""")

    # Demonstração conceitual (sem dependências)
    demo_atencao_conceitual()

    # Demonstrações com Hugging Face (se disponível)
    if not verificar_dependencias():
        exibir_comparativo()
        return

    print("\n  Escolha o que deseja demonstrar:")
    print("  [1] Geração de texto com GPT-2")
    print("  [2] Preenchimento de máscara com BERT")
    print("  [3] Análise de sentimento")
    print("  [4] Modo interativo com GPT-2")
    print("  [5] Tudo acima (exceto interativo)")
    print("  [6] Apenas comparativo final")

    escolha = input("\n  Opção: ").strip()

    if escolha == "1":
        demo_geracao_gpt2()
    elif escolha == "2":
        demo_fill_mask_bert()
    elif escolha == "3":
        demo_sentimento()
    elif escolha == "4":
        modo_interativo_gpt2()
    elif escolha == "5":
        demo_geracao_gpt2()
        demo_fill_mask_bert()
        demo_sentimento()
    else:
        pass

    exibir_comparativo()


if __name__ == "__main__":
    main()
