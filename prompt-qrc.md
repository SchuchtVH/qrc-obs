# Prompt para o Claude Code — Alocação de shots por grupo de medição em QRC

> Cole o conteúdo abaixo (da linha `---` em diante) como primeira mensagem numa sessão limpa do Claude Code,
> dentro de um diretório vazio. Peça para ele entrar em *plan mode* antes de codar se quiser revisar o plano.

---

# Projeto: a distribuição de shots entre grupos de medição reduz o erro de um QRC?

Você vai construir, do zero, um experimento numérico completo e reprodutível. Leia todo este
documento antes de escrever qualquer código, e comece propondo um plano de implementação
(arquivos, ordem, testes). Só depois implemente.

## 0. Pergunta científica

Com **o mesmo reservatório**, **os mesmos pesos de readout** e **o mesmo orçamento total de shots**,
quanto o erro de previsão muda apenas ao redistribuir os shots entre grupos de medição?

Três estratégias principais a comparar:

| Estratégia | Como aloca o orçamento |
|---|---|
| `uniform` | Mesmo número de shots por grupo |
| `shap` | Proporcional à importância SHAP agregada por grupo (calculada no treino) |
| `variance` | Proporcional ao desvio-padrão da contribuição de cada grupo à previsão (incerteza de medição) |

Mais baselines na Seção 8.

**Entregável principal (marco 1):** a curva **erro de previsão × número de execuções (shots)**
no conjunto de teste, com barras de erro sobre múltiplas sementes, para todas as estratégias.

**Hipótese central e falsificável:** `variance` domina `shap` sempre que o ranking dos grupos por
*variância de medição da contribuição* diferir do ranking por *variância de sinal ponderada*.
SHAP mede a segunda; o custo estatístico depende da primeira. O experimento deve **medir esses
dois rankings explicitamente** e mostrar a correlação entre a divergência dos rankings e o gap
de desempenho. Se os rankings coincidirem na tarefa escolhida, isso é um resultado válido e deve
ser reportado como tal — não force um resultado positivo.

## 1. Restrições de implementação (não negociáveis)

- **Python 3.11+, simulação de estado quântico escrita à mão em NumPy.** Nada de PennyLane,
  Qiskit ou Cirq. 4 qubits ⇒ matriz densidade 16×16; use densidade, não vetor de estado
  (há reset de qubit, o estado fica misto).
- Dependências permitidas: `numpy`, `scipy`, `scikit-learn`, `shap`, `pandas`, `pyarrow`,
  `matplotlib`, `pyyaml`, `pytest`, `tqdm`. Nada além disso sem justificar.
- **Determinismo total.** Toda aleatoriedade passa por `numpy.random.Generator` semeado
  explicitamente e propagado por argumento. Nada de `np.random.seed` global. Rodar duas vezes com
  a mesma config tem que dar bit-a-bit o mesmo resultado.
- Configuração num único `configs/base.yaml`; o script de execução aceita overrides por CLI.
- Resultados brutos em Parquet (`results/raw/*.parquet`), figuras em `figures/`, nada de
  resultado hardcoded em notebook.
- Sem `if __name__ == "__main__"` gigante: código em `src/qrcshots/`, scripts finos em `scripts/`.

## 2. Tarefa temporal

Sinal de entrada: `u_t ~ Uniform(-1, 1)`, i.i.d., comprimento total `T = 6000`.

Alvo primário:

```
y_t = u_{t-2} * u_{t-5}
```

Essa tarefa exige memória de curto prazo e uma não-linearidade multiplicativa, e sabemos
exatamente qual informação precisa sobreviver ao reservatório.

Implemente também, atrás de uma flag de config, duas tarefas secundárias para checar generalidade
(rode-as só depois do pipeline fechado na tarefa primária):

- `narma5`: NARMA de ordem 5 padrão, com entrada em `Uniform(0, 0.2)`.
- `product3`: `y_t = u_{t-1} * u_{t-3} * u_{t-7}` (memória mais longa, sinal mais fraco).

**Split temporal, sem embaralhar:** 60% treino / 20% validação / 20% teste, nessa ordem.
Descarte as primeiras `L` amostras (washout, ver Seção 3).

> **Regra dura:** toda decisão — hiperparâmetros do ridge, alocações de shots, escolha de
> estimadores — é tomada usando **apenas treino e validação**. O teste é aberto uma única vez,
> no fim, e o script deve falhar com exceção se qualquer código de decisão tocar no split de teste.
> Implemente isso como uma guarda de verdade (por exemplo, um objeto `Dataset` que só libera o
> split de teste depois de `freeze()` ter sido chamado), não como comentário.

## 3. Reservatório

Esquema tipo Fujii–Nakajima, com reinicialização por janela:

- `n_qubits = 4`.
- Janela de `L = 20` entradas por previsão. Para prever `y_t`, o sistema é inicializado em
  `|0000><0000|` e as entradas `u_{t-L+1}, ..., u_t` são injetadas em sequência.
- **Injeção:** a cada passo, o qubit 0 é resetado e substituído pelo estado puro
  `|psi(u)> = cos(theta/2)|0> + sin(theta/2)|1>`, com `theta = pi*(u+1)/2` mapeando `[-1,1] -> [0,pi]`.
  Operacionalmente: traço parcial sobre o qubit 0 e produto tensorial com `|psi(u)><psi(u)|`.
- **Evolução:** após cada injeção, aplique um unitário fixo `U = exp(-i H tau)`, com `H` um
  Ising transverso aleatório:
  `H = sum_{i<j} J_ij X_i X_j + sum_i h_i Z_i`, `J_ij ~ Uniform(-J, J)`, `h_i ~ Uniform(0.5, 1.5)`.
  `J`, `tau` e a semente do reservatório vêm da config. `H` e `U` são gerados **uma única vez** e
  reusados por todos os experimentos — o reservatório é idêntico entre estratégias, sempre.
- Calcule `U` uma vez por `scipy.linalg.expm` e cacheie.

**Otimização importante:** a matriz densidade final `rho_t` de cada janela depende só da janela
e é **determinística**. Calcule `rho_t` exatamente para todas as janelas de treino/val/teste **uma
única vez**, salve em disco (`cache/rho_<hash-da-config>.npy`, complexo, shape `(N, 16, 16)`), e
reuse. Todo o ruído de shot depois é amostragem a partir dessas `rho_t`. Isso é o que torna o
experimento barato e é o que garante que todas as estratégias veem exatamente a mesma preparação
de estado.

Contabilize o custo de preparação: **cada shot exige reproduzir os `L = 20` passos da janela**.
Registre o custo tanto em shots quanto em `shots * L` (operações de preparação).

## 4. Grupos de medição e amostragem com shots finitos

12 observáveis: `X_i`, `Y_i`, `Z_i` para `i = 0..3`. Três grupos de Paulis que comutam
qubit-a-qubit e são medíveis simultaneamente:

| Grupo | Observáveis | Base de medição |
|---|---|---|
| `X` | `X_0, X_1, X_2, X_3` | Hadamard em todos os qubits |
| `Y` | `Y_0, Y_1, Y_2, Y_3` | `S†` seguido de Hadamard em todos |
| `Z` | `Z_0, Z_1, Z_2, Z_3` | computacional |

**Um shot de um grupo devolve 4 valores `±1` de uma vez** (uma bitstring de 4 bits), com
correlações entre eles. Isso é o cerne do problema: alocamos shots a *grupos*, não a features.

Amostrador (`sample_group(rho, group, n_shots, rng)`):

1. Aplique a rotação de base `R_g` (produto tensorial de rotações de 1 qubit): `rho' = R_g rho R_g†`.
2. `p = real(diag(rho'))`, clip em `>= 0`, renormalize.
3. `counts = rng.multinomial(n_shots, p)` sobre as 16 bitstrings.
4. Converta para as 4 estimativas `<P_i>_hat` (média dos `±1`) e devolva também os counts brutos.

Guarde os counts brutos, não só as médias — a estimativa de covariância intra-grupo depende deles.

Exponha também os valores **exatos** `<P_i> = tr(rho P_i)` e a **covariância de shot único**
`Sigma_g[i,j] = tr(rho P_i P_j) - tr(rho P_i) tr(rho P_j)` (para `i != j` dentro do grupo,
`P_i P_j` é ele mesmo um Pauli, então é um traço direto; para `i = j`, `Sigma_g[i,i] = 1 - <P_i>^2`).
Esses valores exatos são a referência para validar os estimadores empíricos.

## 5. Coleta inicial e treino do readout

- Coleta de treino: **64 shots por grupo por janela** (192 shots/janela). Features = as 12 médias.
- Modelo: `sklearn.linear_model.Ridge` com `alpha` escolhido por busca em grade **na validação**
  (grade log de `1e-6` a `1e2`). Padronize as features com estatísticas do treino apenas.
- **Congele os pesos `w` e o intercepto após o treino.** Todas as estratégias usam exatamente esses
  pesos. Isso isola o efeito da estratégia de medição.
- Registre no relatório o caveat de *errors-in-variables*: pesos treinados com features ruidosas são
  atenuados. Como diagnóstico, treine também um readout com features exatas (shots → ∞) e reporte
  a diferença de norma dos pesos e de erro. Isso não entra na comparação principal, só no apêndice.

Métrica de erro: **NMSE** = `mean((y - y_hat)^2) / var(y)`. Reporte também RMSE.

## 6. Duas formas de decidir o que merece mais shots

### 6.1 SHAP (estratégia de comparação)

Use `shap.LinearExplainer` com o ridge treinado e o background do **treino**. Para cada feature `j`,
`phi_j = mean_over_train(|SHAP_j|)`. Agregue por grupo: `I_g = sum_{j in g} phi_j`.

Anote no código a identidade analítica para modelo linear:
`mean|SHAP_j| = |w_j| * mean|x_j - E[x_j]|`, ou seja, SHAP aqui é essencialmente
`|w_j| * dispersão do sinal na feature j`. **Verifique numericamente** que o SHAP calculado bate
com essa fórmula (teste unitário, tolerância apertada). Isso deixa explícito que SHAP mede
variabilidade *de sinal*, não incerteza *de medição*.

### 6.2 Incerteza da contribuição do grupo

Use um **lote de calibração independente** (subconjunto do treino, disjunto do que treinou o ridge,
ou um conjunto de janelas de treino re-amostradas). Para um grupo `g` com peso `w_g` (subvetor de
`w` restrito às features do grupo, já na escala padronizada correta):

- Contribuição do grupo à previsão: `c_g = w_g · x_g_hat`.
- Com `n` shots, `Var(c_g) = (w_g^T Sigma_g w_g) / n`, onde `Sigma_g` é a covariância de shot único.
- Defina `s_g = sqrt( E_janelas[ w_g^T Sigma_g w_g ] )`.

Estime `s_g` de **duas** formas e compare:

1. **Empírica:** repita `R` vezes (ex. `R = 200`) a medição da *mesma* janela com `n_cal` shots e
   pegue o desvio-padrão de `c_g` entre repetições; escale por `sqrt(n_cal)`. Média sobre janelas
   de calibração.
2. **Exata:** direto de `Sigma_g` calculada de `rho`.

A concordância entre as duas é um teste de sanidade obrigatório.

> **Distinção conceitual a documentar no README:** variar porque a entrada mudou é comportamento do
> sinal (é isso que o SHAP capta); variar ao repetir a *mesma* preparação é incerteza de medição.
> Só a segunda é reduzível gastando shots.

## 7. Da incerteza para a alocação

Para grupos de custo igual, minimizar `Var(sum_g c_g) = sum_g (w_g^T Sigma_g w_g)/n_g` sujeito a
`sum_g n_g = B` dá, por multiplicador de Lagrange:

```
n_g = B * s_g / sum_h s_h
```

**Derive isso explicitamente num docstring** e **verifique numericamente**: para um `B` fixo, faça
um sweep sobre o simplex de alocações e confirme que o mínimo empírico da variância da previsão
cai onde a fórmula prevê (dentro do erro de Monte Carlo). Esse é o teste que impede a implementação
de estar sutilmente errada.

Detalhes de implementação:

- Piso `n_min` shots por grupo (config, default 8) — nenhum grupo pode zerar.
- Arredondamento pelo método dos maiores restos, respeitando `sum_g n_g = B` exatamente.
- A alocação é **global e fixa** (uma por estratégia por orçamento), calculada a partir de
  treino+calibração. Alocação adaptativa por janela é uma extensão opcional (Seção 10).

## 8. Baselines e protocolo de comparação

Além de `uniform`, `shap` e `variance`, implemente:

- `random`: alocação sorteada do simplex (Dirichlet(1,1,1)) — controle contra "qualquer
  desbalanceamento ajuda".
- `oracle`: alocação que minimiza o erro **no teste**, por busca em grade fina no simplex.
  É um limite superior de desempenho, claramente rotulado como não-implementável. Serve para
  medir quanto da margem disponível cada estratégia captura.
- `magnitude`: proporcional a `sum_{j in g} |w_j|` — baseline ingênuo, sem informação de variância.
- `kernel_readout` (**baseline de literatura**): implemente a leitura ótima por kernel de
  [arXiv:2602.14677](https://arxiv.org/abs/2602.14677) — regressão ridge de kernel com kernel de
  Hilbert–Schmidt `K(rho_a, rho_b) = tr(rho_a rho_b)`, que fornece o observável de readout ótimo
  `O* = sum_a alpha_a rho_a` para o reservatório e o dataset dados. Decomponha `O*` na base de
  Pauli, **restrinja ao suporte dos 12 Paulis de peso 1** que conseguimos medir com os três grupos
  (projeção ortogonal na base de Pauli), e reporte:
  (a) o NMSE dessa leitura com shots infinitos vs. o do ridge treinado;
  (b) a fração da norma de `O*` que vive fora do suporte medível — isso quantifica o que a nossa
  restrição de grupos está jogando fora;
  (c) o NMSE com shots finitos, usando a alocação `variance` calculada para os pesos de `O*`.
  Antes de codar, **leia o abstract e o método do paper** (`WebFetch` no arXiv) e escreva um
  resumo de meia página em `docs/baseline_kernel.md`, incluindo onde a nossa formulação difere.
  Se algum detalhe do método não for reconstruível a partir do abstract/página, diga isso
  explicitamente em vez de inventar; implemente a versão que você consegue justificar e documente
  a lacuna.

Protocolo:

- Orçamentos `B ∈ {96, 150, 300, 600, 1200, 2400}` shots por previsão (o piso `n_min`
  precisa caber; ajuste se necessário).
- `n_seeds = 30` realizações independentes da amostragem no teste, por (estratégia, orçamento).
- **Semente pareada:** para um dado par (seed, orçamento), todas as estratégias usam o mesmo
  `rng` de partida por janela, de modo que a comparação seja pareada e a variância entre
  estratégias caia. Documente como isso foi feito.
- Reporte média, desvio-padrão e IC bootstrap 95% do NMSE. Para o par `shap` vs `variance`,
  faça um teste pareado (Wilcoxon signed-rank sobre as seeds) e reporte o **tamanho de efeito**,
  não só o p-valor.

## 9. Contabilidade honesta de custo

A figura principal é **NMSE (eixo y, log) × número de execuções (eixo x, log)**. Mas produza
**duas versões**:

1. **Custo marginal:** só os shots de previsão (`B` por janela de teste).
2. **Custo total amortizado:** `(shots_coleta_inicial + shots_calibração + N_test * B) / N_test`
   no eixo x. Aqui `uniform` tem overhead zero e as outras pagam a calibração.
   Calcule e reporte o **break-even**: quantas previsões são necessárias para que `variance`
   compense seu custo de calibração frente a `uniform`, para cada orçamento.

Reporte também:

- O **piso de erro do reservatório**: NMSE com shots infinitos (features exatas). Se o erro de
  shot já for muito menor que esse piso, nenhuma estratégia de alocação importa — e isso precisa
  aparecer na figura como uma linha horizontal. **Esse é o resultado negativo mais provável;
  não o esconda.**
- Decomposição do erro: `NMSE_total ≈ NMSE_piso + contribuição de shot noise`. Verifique
  numericamente essa aditividade e mostre a fração atribuível a cada parte por orçamento.

## 10. Fase 2 (implementar só depois da fase 1 fechada e testada)

Aprendizado conjunto de pesos e alocação, em `src/qrcshots/joint.py`:

```
repita:
  1. treine w com as features medidas sob a alocação atual
  2. recalcule s_g com os novos w e realoque
  3. colete novas medições sob a nova alocação
  4. avalie na validação; pare quando não melhorar por `patience` iterações
```

Reporte: curva de validação por iteração, se converge, quantas iterações, e o **custo total de
treino** (shots gastos no loop) — comparado ao ganho no teste. Compare contra `kernel_readout`.
Deixe claro se o ganho compensa o custo adicional; se não compensar, diga isso.

## 11. Testes (pytest, obrigatórios)

- `sample_group` é não-viesado: com `n` grande, `<P_i>_hat -> tr(rho P_i)` (tolerância por CLT).
- `Var(<P_i>_hat) ≈ (1 - <P_i>^2)/n`.
- Covariância empírica intra-grupo converge para `Sigma_g` exata.
- Estados de referência: para `|0000>`, `<Z_i> = 1`, `<X_i> = <Y_i> = 0`; para `|++++>`,
  `<X_i> = 1`. Rotações de base corretas para X, Y, Z.
- `U` é unitário; `rho` permanece hermitiana, PSD e de traço 1 ao longo de toda a janela.
- Ridge bate com a solução fechada `(X^T X + alpha I)^{-1} X^T y`.
- SHAP linear bate com `|w_j| * mean|x_j - E[x_j]|`.
- Alocação: soma exatamente `B`, respeita `n_min`, é invariante a reescala de `s`.
- A guarda de test-set levanta exceção se acessada antes de `freeze()`.
- Reprodutibilidade: mesma config duas vezes ⇒ mesmos números.

## 12. Entregáveis

```
README.md                  # pergunta, método, como rodar, resultados principais
configs/base.yaml
src/qrcshots/{reservoir,measure,readout,allocate,shapley,experiment,joint,plots}.py
scripts/{run_phase1.py,run_phase2.py,make_figures.py}
tests/
results/raw/*.parquet
figures/error_vs_shots_marginal.png
figures/error_vs_shots_amortized.png
figures/allocation_simplex.png     # alocações das estratégias sobre o simplex, com o oráculo
figures/ranking_divergence.png     # ranking SHAP vs ranking de variância, por grupo
docs/baseline_kernel.md
RESULTS.md                 # achados, com números, tabelas e as ressalvas
```

`RESULTS.md` deve responder, com números:

1. Com o mesmo orçamento, qual estratégia tem menor erro? Qual o tamanho do efeito?
2. Para atingir um NMSE alvo, quantos shots cada estratégia precisa?
3. SHAP oferece alguma vantagem sobre a decisão direta por incerteza? Em que regime?
4. O ganho sobrevive à contabilidade de custo total (calibração inclusa)? A partir de quantas
   previsões?
5. Quanto do erro é ruído de medição e quanto é limitação do reservatório?
6. Como as três estratégias se comparam ao oráculo e ao `kernel_readout`?

## 13. Como trabalhar

- Comece por um plano e me mostre antes de implementar.
- Construa em ordem: reservatório → amostrador → testes → readout → alocação → experimento →
  figuras → baselines de literatura → fase 2. Rode os testes a cada etapa.
- Faça commits pequenos e descritivos.
- Antes de rodar a grade completa, faça um *smoke run* com `T=600`, `n_seeds=3`, 2 orçamentos,
  e me mostre a figura preliminar.
- **Não ajuste nada para tornar o resultado mais bonito.** Um resultado nulo bem medido — "a
  alocação não importa nesse regime porque o piso do reservatório domina" — é uma resposta válida
  e útil. Se encontrar isso, diga claramente e proponha em que regime (menos shots, reservatório
  melhor, tarefa mais difícil, mais grupos) o efeito apareceria.
- Se algo no enunciado acima estiver ambíguo ou parecer errado, pergunte antes de escolher sozinho.

## Referências

- Yen et al., *Deterministic improvements of quantum measurements with grouping of compatible
  operators*, npj Quantum Information (2023). https://www.nature.com/articles/s41534-023-00683-y
- *Measurement budget allocation*, Phys. Rev. Research 4, 033173 (2022).
  https://doi.org/10.1103/PhysRevResearch.4.033173
- Leitura/observáveis ótimos para QRC via kernel de Hilbert–Schmidt: https://arxiv.org/abs/2602.14677
- `shap.LinearExplainer`: https://shap.readthedocs.io/en/latest/generated/shap.LinearExplainer.html
