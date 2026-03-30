
import re
import random

REGRAS = [
    (
        r"\b(eu sinto|eu me sinto|estou me sentindo|me sinto)\b(.+)",
        [
            "Por que você diz que se sente {1}?",
            "Há quanto tempo você se sente {1}?",
            "O que te faz sentir {1}?",
        ],
    ),
    (
        r"\b(eu preciso|eu quero|eu desejo)\b(.+)",
        [
            "Por que você precisa de {1}?",
            "O que aconteceria se você conseguisse {1}?",
            "Você realmente acha que vai conseguir {1}?",
        ],
    ),
    (
        r"\b(eu (não )?consigo|eu (não )?posso)\b(.+)",
        [
            "O que te impede de {3}?",
            "Você já tentou {3}?",
            "Como você se sentiria se conseguisse {3}?",
        ],
    ),
    (
        r"\b(minha? (mãe|pai|família|irmão|irmã|filho|filha))\b",
        [
            "Fale mais sobre sua família.",
            "Como é a sua relação com {0}?",
            "Sua família tem influência em como você se sente?",
        ],
    ),
    (
        r"\b(triste|deprimido|ansioso|com medo|preocupado|estressado)\b",
        [
            "Sinto muito ouvir isso. O que está causando esse sentimento?",
            "Há quanto tempo você se sente assim?",
            "Você já conversou com alguém sobre isso?",
        ],
    ),
    (
        r"\b(feliz|alegre|bem|ótimo|maravilhoso)\b",
        [
            "Que bom! O que está te deixando assim?",
            "O que contribui para essa felicidade?",
            "Você se sente assim com frequência?",
        ],
    ),
    (
        r"\b(sim|claro|com certeza|exatamente)\b",
        [
            "Pode me dizer mais sobre isso?",
            "Por quê você acha isso?",
            "Interessante. Continue...",
        ],
    ),
    (
        r"\b(não|nunca|jamais)\b",
        [
            "Por que não?",
            "Você tem certeza disso?",
            "Isso sempre foi assim?",
        ],
    ),
    (
        r"\b(por que|porque|por quê)\b(.+)",
        [
            "Essa é uma boa pergunta. O que você pensa sobre isso?",
            "Por que você acha que {1}?",
            "Você consegue encontrar uma resposta para isso?",
        ],
    ),
    (
        r"\b(sonho|sonhei)\b(.+)",
        [
            "Sonhos podem ser fascinantes. Você se lembra de outros detalhes?",
            "O que esse sonho significa para você?",
            "Como você se sentiu após esse sonho?",
        ],
    ),
    (
        r"\b(tchau|adeus|até logo|sair|exit|quit)\b",
        [
            "Foi um prazer conversar com você. Cuide-se!",
            "Até logo! Lembre-se: falar sobre seus sentimentos é importante.",
            "Adeus! Volte sempre que precisar conversar.",
        ],
    ),
]

RESPOSTAS_GENERICAS = [
    "Pode me contar mais sobre isso?",
    "Isso é muito interessante. Continue...",
    "Por que você traz isso à tona agora?",
    "Como isso te faz sentir?",
    "Entendo. O que mais você gostaria de compartilhar?",
    "Isso tem relação com algo que aconteceu recentemente?",
]

def processar_entrada(texto: str) -> str:
    texto = texto.lower().strip()

    for padrao, respostas in REGRAS:
        match = re.search(padrao, texto, re.IGNORECASE)
        if match:
            resposta = random.choice(respostas)
            try:
                grupos = match.groups()
                for i, grupo in enumerate(grupos):
                    if grupo:
                        resposta = resposta.replace(f"{{{i}}}", grupo.strip())
                        resposta = resposta.replace(f"{{{i+1}}}", grupo.strip())
            except Exception:
                pass
            return resposta

    return random.choice(RESPOSTAS_GENERICAS)

def iniciar_conversa():
    print("=" * 60)
    print("  ELIZA - O Psicólogo de Terminal (Fase 1)")
    print("  Inspirado no programa ELIZA de Joseph Weizenbaum (1966)")
    print("=" * 60)
    print("\nELIZA: Olá! Sou ELIZA. Como você está se sentindo hoje?")
    print("       (Digite 'sair' para encerrar)\n")

    palavras_saida = {"sair", "exit", "quit", "tchau", "adeus", "não quero mais conversar"}

    while True:
        try:
            entrada = input("Você: ").strip()

            if not entrada:
                print("ELIZA: Por favor, me diga algo.\n")
                continue

            resposta = processar_entrada(entrada)
            print(f"ELIZA: {resposta}\n")

            # Verifica se é despedida
            if any(palavra in entrada.lower() for palavra in palavras_saida):
                break

        except KeyboardInterrupt:
            print("\nELIZA: Até logo! Cuide-se.")
            break

def exibir_reflexao():
    print("\n" + "=" * 60)
    print("  REFLEXÃO: Por que o sistema falha fora do roteiro?")
    print("=" * 60)
    reflexao = """
A ELIZA usa um conjunto fixo  de expressões regulares, famoso Regex.
Isso significa que:

  1. RIGIDEZ: Apenas frases que casam com os padrões definidos
     recebem respostas relevantes. Fora do roteiro, o sistema
     cai em respostas genéricas vazias.

  2. SEM CONTEXTO: Cada mensagem é processada de forma isolada.
     O chatbot não lembra o que foi dito anteriormente.

  3. SEM COMPREENSÃO: O sistema não "entende" o significado
     das palavras — apenas reconhece padrões de caracteres.

  4. FRÁGIL A VARIAÇÕES: "Estou triste" pode casar, mas
     "Ando meio pra baixo" não, pois as palavras são diferentes.

Essa é a principal limitação da Era Simbólica do PLN.
"""
    print(reflexao)


if __name__ == "__main__":
    iniciar_conversa()
    exibir_reflexao()
