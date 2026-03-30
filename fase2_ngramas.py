import re
import random
from collections import defaultdict, Counter

CORPUS = """
O gato miou alto na janela. O gato correu pelo jardim rapidamente.
O gato bebeu água da tigela. O cachorro latiu para o gato assustado.
O sol brilhou forte durante o dia todo. O sol nasceu cedo pela manhã.
A criança brincou no parque feliz. A criança comeu o sorvete gelado.
O livro estava sobre a mesa. O livro contava uma história bonita.
A professora ensinou os alunos com paciência. A professora corrigiu as provas.
O computador processou os dados rapidamente. O computador estava ligado.
A cidade estava movimentada naquele dia. A cidade cresceu muito nos últimos anos.
O homem caminhou até a padaria de manhã. O homem bebeu o café com leite.
A mulher leu o jornal com atenção. A mulher ouviu música enquanto trabalhava.
O tempo passou depressa durante as férias. O tempo estava nublado e frio.
O menino jogou bola com os amigos. O menino estudou para a prova difícil.
A menina cantou uma música bonita. A menina desenhou flores coloridas no caderno.
O carro parou no semáforo vermelho. O carro acelerou na estrada aberta.
A flor desabrochou na primavera quente. A flor perfumou o quarto inteiro.
"""

def tokenizar(texto: str) -> list:
    texto = texto.lower()
    tokens = re.findall(r'\b[a-záéíóúâêîôûãõàèìòùç]+\b', texto)
    return tokens

def construir_ngramas(tokens: list, n: int) -> dict:
    modelo = defaultdict(Counter)
    for i in range(len(tokens) - n):
        contexto = tuple(tokens[i:i + n - 1])
        proxima = tokens[i + n - 1]
        modelo[contexto][proxima] += 1
    return dict(modelo)

def calcular_probabilidades(modelo: dict) -> dict:
    probabilidades = {}
    for contexto, contagens in modelo.items():
        total = sum(contagens.values())
        probabilidades[contexto] = {
            palavra: round(freq / total, 4)
            for palavra, freq in contagens.items()
        }
    return probabilidades

def prever_proxima_palavra(modelo_prob: dict, contexto: tuple) -> str:
    if contexto not in modelo_prob:
        return None

    opcoes = list(modelo_prob[contexto].keys())
    pesos = list(modelo_prob[contexto].values())
    return random.choices(opcoes, weights=pesos, k=1)[0]

def gerar_texto(modelo_prob: dict, semente: str, comprimento: int = 15) -> str:
    n = len(list(modelo_prob.keys())[0]) + 1  # Descobre o N
    contexto_tamanho = n - 1

    tokens = semente.lower().split()

    if len(tokens) < contexto_tamanho:
        tokens = tokens * contexto_tamanho
    tokens = tokens[-contexto_tamanho:]

    texto_gerado = list(tokens)

    for _ in range(comprimento):
        contexto = tuple(tokens[-contexto_tamanho:])
        proxima = prever_proxima_palavra(modelo_prob, contexto)
        if proxima is None:
            break
        texto_gerado.append(proxima)
        tokens = tokens[1:] + [proxima]

    return " ".join(texto_gerado)


def exibir_distribuicao(modelo_prob: dict, contexto_str: str):
    palavras = contexto_str.lower().split()
    contexto = tuple(palavras[-1:]) if len(palavras) >= 1 else tuple(palavras)

    print(f"\n  Contexto: '{contexto_str}'")
    print("  " + "-" * 40)

    if contexto not in modelo_prob:
        contexto = tuple(palavras)
    if contexto not in modelo_prob:
        print(f"  Contexto '{contexto_str}' não encontrado no corpus.")
        return

    dist = modelo_prob[contexto]
    dist_ordenada = sorted(dist.items(), key=lambda x: x[1], reverse=True)

    for palavra, prob in dist_ordenada:
        barra = "█" * int(prob * 40)
        print(f"  {palavra:<15} {barra} {prob*100:.1f}%")


def main():
    print("=" * 60)
    print("  FASE 2: Predição por N-Grams")
    print("  Modelo Estatístico de Linguagem")
    print("=" * 60)

    tokens = tokenizar(CORPUS)
    print(f"\n[1] Corpus tokenizado: {len(tokens)} tokens")
    print(f"    Vocabulário único: {len(set(tokens))} palavras")
    print(f"    Amostra: {tokens[:10]}")

    print("\n[2] Construindo modelos N-Gram...")
    modelo_bigrama = construir_ngramas(tokens, n=2)
    modelo_trigrama = construir_ngramas(tokens, n=3)

    prob_bigrama = calcular_probabilidades(modelo_bigrama)
    prob_trigrama = calcular_probabilidades(modelo_trigrama)

    print(f"    Bigramas únicos:   {len(prob_bigrama)}")
    print(f"    Trigramas únicos:  {len(prob_trigrama)}")

    print("\n[3] Distribuição de Probabilidade (exemplo do slide):")
    exibir_distribuicao(prob_bigrama, "o gato")
    exibir_distribuicao(prob_bigrama, "a criança")
    exibir_distribuicao(prob_bigrama, "o sol")

    print("\n[4] Gerando texto com Bigramas (contexto = 1 palavra):")
    for semente in ["o", "a", "sol"]:
        texto = gerar_texto(prob_bigrama, semente, comprimento=10)
        print(f"    Semente '{semente}': {texto}")

    print("\n[5] Gerando texto com Trigramas (contexto = 2 palavras):")
    for semente in ["o gato", "a criança", "o sol"]:
        texto = gerar_texto(prob_trigrama, semente, comprimento=10)
        print(f"    Semente '{semente}': {texto}")

    print("\n" + "=" * 60)
    print("  LIMITAÇÃO: Contexto Curto dos N-Grams")
    print("=" * 60)
    print("""
  O bigrama só enxerga 1 palavra anterior para prever a próxima.
  O trigrama enxerga 2 palavras. Isso causa:

  - INCOERÊNCIA em textos longos: o modelo "esquece" o início.
  - REPETIÇÃO DE LOOPS: pode ficar preso em padrões circulares.
  - SEM SEMÂNTICA: não entende o significado, apenas frequência.

  Exemplo de falha:
  "O gato miou e depois o cachorro..." → o bigrama pode sugerir
  "gato" após "cachorro" porque ambos aparecem após "o".

  Solução histórica: aumentar N (4-grams, 5-grams), mas o custo
  computacional explode e o problema de contexto persiste.
""")

    print("=" * 60)
    print("  MODO INTERATIVO")
    print("=" * 60)
    print("  Digite uma palavra para gerar texto (ou 'sair' para encerrar)")
    print()

    while True:
        try:
            entrada = input("  Semente: ").strip()
            if entrada.lower() in {"sair", "exit", "quit"}:
                print("  Encerrando gerador N-Grams.")
                break
            if not entrada:
                continue

            palavras = entrada.split()
            if len(palavras) >= 2:
                texto = gerar_texto(prob_trigrama, entrada, comprimento=12)
                print(f"  Trigrama → {texto}\n")
            else:
                texto = gerar_texto(prob_bigrama, entrada, comprimento=12)
                print(f"  Bigrama  → {texto}\n")

        except KeyboardInterrupt:
            print("\n  Encerrando.")
            break


if __name__ == "__main__":
    main()
