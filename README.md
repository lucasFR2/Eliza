# Da ELIZA aos Transformers 
### A Evolução do Processamento de Linguagem Natural

---

## 📋 Sobre o Projeto

Este repositório documenta a jornada de aprendizado sobre a evolução do PLN (Processamento de Linguagem Natural), desde os sistemas simbólicos dos anos 1960 até os modernos Transformers, implementando cada abordagem em Python.

---

## 🗂️ Estrutura do Repositório

```
eliza/
│
├── fase1_eliza.py          # Era Simbólica: Chatbot com Regex
├── fase2_ngramas.py        # Revolução Estatística: N-Grams
├── fase3_transformers.py   # Lógica Transformer: GPT-2 e BERT
└── README.md
```

---

## 🚀 Como Executar

### Pré-requisitos

- Python 3.8 ou superior

### Instalação das Dependências

```bash
# Dependências básicas (Fases 1 e 2 não precisam de instalação extra)
pip install gensim                          # Fase 3: Word2Vec
pip install transformers torch              # Fase 4: GPT-2 e BERT
```

### Executando cada fase

```bash
python fase1_eliza.py        # Inicia o chatbot ELIZA no terminal
python fase2_ngramas.py      # Treina e demonstra o gerador N-Grams
python fase3_transformers.py # Demonstra GPT-2, BERT e Self-Attention
```

> ⚠️ **Fase 3:** Na primeira execução, os modelos GPT-2 e BERT serão baixados automaticamente (~500MB).

---

## 📖 As 3 Fases do Aprendizado

### Fase 1 — Era Simbólica: ELIZA 
**Arquivo:** `fase1_eliza.py`

Reimplementação do chatbot ELIZA (Weizenbaum, 1966). O sistema usa **Expressões Regulares (Regex)** para identificar padrões na entrada do usuário e retornar respostas pré-programadas.

**Conceitos:** Regex, Casamento de Padrões, Sistemas Baseados em Regras

**Limitação explorada:** Por que o sistema falha com frases fora do roteiro?
- Sem memória de contexto
- Sem compreensão semântica real
- Frágil a variações de vocabulário

---

### Fase 2 — Revolução Estatística: N-Grams
**Arquivo:** `fase2_ngramas.py`

Abandona as regras fixas e adota **probabilidade baseada em frequência**. O modelo aprende bigramas e trigramas a partir de um corpus e gera texto prevendo a próxima palavra.

**Conceitos:** N-Grams, Bigramas, Trigramas, Probabilidade Condicional

**Exemplo do slide:**

| Contexto | Próxima Palavra | Probabilidade |
|----------|----------------|---------------|
| "O gato" | miou           | 85%           |
| "O gato" | correu         | 10%           |
| "O gato" | latiu          | 5%            |

**Limitação explorada:** A "memória curta" impede coerência em textos longos.

---

### Fase 3 — Lógica Transformer 
**Arquivo:** `fase3_transformers.py`

Usa a biblioteca **Hugging Face** para explorar modelos Transformer modernos: GPT-2 para geração de texto e BERT para análise de sentimento e preenchimento de máscara.

**Conceitos:** Self-Attention, Processamento Paralelo, Embeddings Contextuais, RLHF

**Revolução de 2017:** "Attention is All You Need" (Vaswani et al.)
- Substitui RNNs por processamento paralelo
- Self-Attention captura dependências de longa distância
- Embeddings **contextuais** (a mesma palavra tem vetores diferentes conforme o contexto)

---

## 📊 Comparativo de Abordagens

| Atributo       | ELIZA (Simbólica) | N-Grams (Estatística) | Transformers (Neural) |
|----------------|-------------------|----------------------|----------------------|
| **Lógica**     | Regex / Regras    | Frequência / Prob.   | Vetor / Atenção      |
| **Contexto**   | Nenhum            | Curto (2–3 palavras) | Longo e Dinâmico     |
| **Flexibilidade** | Rígida         | Moderada             | Extrema              |
| **Semântica**  | Não               | Não                  | Sim (contextual)     |
| **Treinamento**| Manual            | Corpus               | Auto-supervisionado  |

---

## 🎯 Conclusão: O Impacto do RLHF

Os modelos Transformer base (GPT-2, BERT) aprendem a estrutura da linguagem, mas sem alinhamento com valores humanos. O **RLHF (Reinforcement Learning from Human Feedback)** é a etapa que transforma um modelo linguístico em um assistente útil:

1. **Coleta** de preferências humanas sobre respostas do modelo
2. **Treinamento** de um modelo de recompensa com base nessas preferências  
3. **Ajuste fino** do LLM via Reinforcement Learning (PPO)

Sem RLHF: modelos fluentes, mas potencialmente prejudiciais ou não-úteis.  
Com RLHF: assistentes **úteis, seguros e honestos** (ChatGPT, Claude, Gemini).

---

## 📚 Referências

- Weizenbaum, J. (1966). ELIZA — a computer program for the study of natural language communication between man and machine.
- Mikolov, T. et al. (2013). Efficient Estimation of Word Representations in Vector Space. *(Word2Vec)*
- Vaswani, A. et al. (2017). Attention is All You Need. *(Transformer)*
- Ouyang, L. et al. (2022). Training language models to follow instructions with human feedback. *(RLHF)*

---

