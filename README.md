# QRC shot allocation — a distribuição de shots entre grupos de medição reduz o erro?

Experimento numérico reprodutível: com **o mesmo reservatório**, **os mesmos pesos de
readout** e **o mesmo orçamento total de shots**, quanto o erro de previsão muda só
ao redistribuir os shots entre os três grupos de medição comutantes (`X`, `Y`, `Z`,
4 qubits cada)?

Hipótese central: a estratégia `variance` (aloca por incerteza de *medição*) domina
`shap` (aloca por variabilidade de *sinal*) sempre que os rankings de grupo por essas
duas grandezas divergirem. Quando os rankings coincidem, as duas devem empatar — e é
isso, de fato, o que a grade completa mostra (ver `RESULTS.md`).

## Leia primeiro: por que a tarefa principal não é a `primary` do enunciado original

A tarefa primária especificada (`y_t = u_{t-2}·u_{t-5}`) foi diagnosticada como sem
sinal decodificável pelos 12 observáveis de peso 1 deste reservatório (4 qubits, reset
por passo) — testamos ~80 combinações de parâmetros do reservatório, todas no piso
trivial (NMSE ≥ 0.94). `narma5` (tarefa secundária do enunciado), com os *mesmos*
observáveis e protocolo, tem sinal forte (NMSE ≈ 0.16-0.22 com features exatas) e foi
promovida a tarefa principal do marco 1, com aprovação explícita do usuário. A tarefa
primária é reportada como resultado negativo honesto. Detalhes completos, com os
números do diagnóstico, em `RESULTS.md` (seção "Por que narma5...") e em comentários
em `configs/base.yaml`.

## Como rodar

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -e . numpy scipy scikit-learn shap pandas pyarrow matplotlib pyyaml pytest tqdm

pytest -q                                   # 59 testes

python scripts/run_phase1.py                # grade completa (T=6000, 30 seeds, 6 orçamentos)
python scripts/make_figures.py results/raw/phase1_<hash>.parquet
python scripts/run_kernel_readout.py        # baseline de literatura (Seção 8)
python scripts/run_phase2.py --budget-per-window 300   # fase 2 opcional (Seção 10)
```

Overrides de config via CLI: `python scripts/run_phase1.py signal.T=600 experiment.n_seeds=3 "allocation.budgets=[96,300]"`.

Smoke run (recomendado antes de qualquer mudança): `signal.T=600 experiment.n_seeds=3`.

## Estrutura

```
configs/base.yaml                # config única; overrides via CLI (a.b.c=valor)
src/qrcshots/
  config.py       seeds derivadas de seeds.master (SeedSequence.spawn), hash de config p/ cache
  tasks.py        sinal de entrada + 3 tarefas (primary/narma5/product3) + encoding de injeção
  dataset.py      split temporal 60/20/20, janelas, guarda de freeze() do teste
  reservoir.py    matriz densidade 16x16 à mão (NumPy), injeção com reset, Ising transverso, cache de rho_t
  measure.py      grupos de medição X/Y/Z, amostragem com shots finitos, amostragem pareada
  readout.py      coleta de features, padronização, ridge congelado
  shapley.py      SHAP linear + verificação da identidade analítica
  allocate.py     incerteza de contribuição (s_g), fórmula de Lagrange, 6 estratégias de alocação
  kernel_readout.py  baseline de literatura (Hilbert-Schmidt kernel, Seção 8)
  experiment.py   orquestração da fase 1 completa
  joint.py        fase 2: aprendizado conjunto de pesos e alocação
  plots.py        bootstrap CI, Wilcoxon, break-even, decomposição de NMSE, figuras
scripts/          run_phase1.py, run_phase2.py, make_figures.py, run_kernel_readout.py
tests/            59 testes pytest
RESULTS.md        achados com números, respondendo as 6 perguntas da Seção 12
docs/baseline_kernel.md   leitura do paper arXiv:2602.14677 + onde diferimos
```

## Decisões de design não triviais (documentadas em detalhe no código)

- **Determinismo total.** Um único `seeds.master` na config; toda semente usada em
  qualquer parte do código é derivada dele via `numpy.random.SeedSequence.spawn()`,
  indexada por posição (`config.py:SEED_SLOTS`). Nunca `np.random.seed` global.
  Verificado por teste: mesma config → mesmos números, bit-a-bit, mesmo recomputando
  o cache do zero.
- **Amostragem pareada entre estratégias (Seção 8).** `rng.multinomial` amostrado
  independentemente para cada `n_shots` não compartilha prefixo entre alocações
  diferentes do mesmo grupo. Em vez disso, para cada seed sorteamos uma sequência-
  mestre de índices de bitstring de tamanho `max(budgets)` por (janela, grupo) uma
  única vez; toda estratégia e todo orçamento usa um prefixo dessa mesma sequência
  (`measure.draw_shot_indices` + `experiment.evaluate_all_seeds`). Isso pareia não só
  entre estratégias de um mesmo orçamento (exigido pelo enunciado), mas também entre
  orçamentos diferentes.
- **Escala do peso na fórmula de incerteza (Seção 6.2/7).** Os pesos do ridge atuam no
  espaço padronizado (`x_std=(x_raw-mean)/scale`); `Sigma_g` é a covariância das
  estimativas raw. Combiná-los exige converter o peso para a escala raw
  (`w_g/scale_g`, ver `allocate.group_weight_raw_scale`) antes de calcular
  `w_g^T Sigma_g w_g` -- combinar direto daria unidades erradas.
- **Custo de calibração: passe único vs. protocolo de sanidade.** A Seção 6.2 pede
  duas estimativas de `s_g` (empírica por R=200 repetições da mesma janela, e exata)
  para uma checagem de sanidade. Usar o protocolo de R repetições como o custo real
  de calibração da estratégia `variance` infla o custo ~200x e torna o break-even
  (Seção 9, Q4) artificialmente infinito. Implementamos um segundo estimador,
  estatisticamente equivalente, de **passe único** (`allocate.s_g_single_pass_empirical`,
  usa a covariância entre os shots individuais de UM lote, sem repetir a janela) --
  é o que decide a alocação de `variance` e entra na contabilidade de custo; o
  protocolo de R repetições é mantido só como checagem de sanidade contra `s_g` exato.
- **Reescala da injeção para `narma5`.** `u ~ Uniform(0,0.2)` cobre só um arco de
  ~10° na esfera de Bloch via `theta=pi*(u+1)/2`, tornando a dispersão das features
  exatas menor que o ruído de 64 shots -- sem sinal aprendível mesmo com reservatório
  bom. `tasks.py` injeta uma versão reescalada de `u` (`u'=(u-centro)/semi_faixa`,
  cobrindo `theta` em `[0,pi]`) mantendo `y` calculado sobre o `u` original. Ver
  `RESULTS.md`.
- **Errors-in-variables (Seção 5, apêndice).** Pesos treinados com features de 64
  shots são atenuados frente aos treinados com features exatas (norma ~21% menor
  neste experimento). Diagnóstico em `RESULTS.md`, não entra na comparação principal.
- **Oracle eficiente (Seção 8).** Em vez de re-simular o pipeline completo por ponto
  de grade no simplex (inviável para `resolution=41`), `oracle` minimiza a variância
  analítica `sum_g s_g_teste^2/n_g` usando `Sigma_g` exato do próprio teste --
  matematicamente equivalente (verificado: a alocação de `oracle` sempre tem a menor
  variância analítica entre todas as estratégias) e computacionalmente trivial.
  Continua rotulado não-implementável porque usa estatística exata do teste.

## Resultados principais

Ver `RESULTS.md` para os números completos (6 perguntas da Seção 12), tabelas em
`figures/tables/` e as 4 figuras em `figures/`. Resumo:

- Com `narma5`, o piso do reservatório é NMSE≈0.664; em orçamentos baixos (`B≤300`)
  as estratégias informadas (`shap`, `variance`, `magnitude`, quase empatadas com o
  `oracle`) reduzem o NMSE em 1.4-3.6% frente a `uniform`, com ICs bootstrap que não
  se sobrepõem. Em `B≥1200` a diferença desaparece (piso do reservatório domina).
- `shap` e `variance` são estatisticamente indistinguíveis aqui (Wilcoxon p>0.1 em
  todo orçamento) porque, nesta tarefa/reservatório, os rankings de grupo por
  importância SHAP e por incerteza `s_g` **coincidem** (Kendall tau=1.0) -- é o caso
  previsto pela hipótese central quando não há divergência de ranking.
- `random` é consistentemente a pior estratégia em orçamentos baixos/médios --
  confirma que desbalancear na direção errada piora, não ajuda.
- O ganho sobrevive à contabilidade de custo total (calibração inclusa), com
  break-even entre ~556 e ~11.000 previsões dependendo do orçamento por previsão.
- O baseline `kernel_readout` mostra que 97.6% da informação útil do observável
  ótimo vive fora do suporte medível por grupos de Pauli compatíveis -- projetar
  esse observável para o que conseguimos medir é muito pior que ajustar
  diretamente nesse subespaço (como o ridge do marco 1 faz).
- A fase 2 (aprendizado conjunto) dá um ganho marginal (~0.77% de NMSE) a um custo
  ~50x maior que a calibração da fase 1 -- não compensa.
