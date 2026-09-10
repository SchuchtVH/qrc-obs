# RESULTS

Grade completa: `T=6000`, `n_seeds=30`, orçamentos `B ∈ {96, 150, 300, 600, 1200, 2400}`,
tarefa `narma5` (ver "Por que narma5 e não a tarefa primária" abaixo). Resultado salvo em
`results/raw/phase1_33180331d4572d11.parquet` (+ `_meta.json`). Figuras em `figures/`.

**Reprodutibilidade (Seção 1, não-negociável):** reverificado ponta-a-ponta no
código final (pós todos os ajustes de tau/J, reescala de injeção, estimador de
passe único) -- `run_phase1.py` rodado duas vezes com a mesma config
(`T=600, n_seeds=3`), cache limpo entre as rodadas: parquet de resultados, `s_g`,
`floor_nmse` e alocações idênticos bit-a-bit nas duas rodadas.

## Por que `narma5` e não a tarefa primária (nulo honesto, Seção 13)

A tarefa primária do enunciado, `y_t = u_{t-2}·u_{t-5}`, foi testada exaustivamente
(~80 combinações de `tau`/`J`/faixa de `h`/semente do Hamiltoniano, buscadas só em
treino+validação) e **não tem sinal linearmente decodificável** a partir dos 12
observáveis de peso-1 (`X_i, Y_i, Z_i`, i=0..3) usados neste projeto:

- Isolando a tarefa: memória linear de um único lag decai rápido com a distância. A
  lag 2 chega a NMSE 0.32 (bom) nos melhores parâmetros testados, mas a lag 5 nunca
  passa de ~0.83-0.89 em nenhuma combinação -- uma parede estrutural, não falta de
  busca. É esperado: o qubit 0 é resetado a cada passo (fixo pelo enunciado, Seção 3),
  então qualquer informação injetada tem só *um* passo de evolução para se propagar
  aos qubits 1-3 antes de ser potencialmente destruída pelo próximo reset.
- Como a tarefa primária precisa das duas lags simultaneamente, o produto fica
  travado pela lag mais fraca: melhor NMSE de teste encontrado em ~80 configs foi
  ~0.945-0.98 (piso trivial é NMSE=1.0). Com a configuração final do reservatório
  (`tau=2, J=8`, a mesma usada para narma5), o piso real é **NMSE=1.0018** -- ou seja,
  pior que prever a média. Não há efeito de alocação de shots para medir aqui: não dá
  para proteger sinal que não existe.

Teste discriminante: a tarefa secundária `narma5` (Seção 2), com os **mesmos**
observáveis, mesmo reservatório e mesmo protocolo, chega a NMSE≈0.16-0.36 (readout
treinado com features exatas) em várias configurações -- prova que o pipeline
(reservatório, amostrador, ridge, tudo) funciona e capta sinal quando ele existe. O
problema da tarefa primária é específico da combinação "produto de duas lags
diferentes" + "só observáveis de peso 1" + "reset por passo", não um bug.

**Decisão (aprovada pelo usuário):** promover `narma5` a tarefa principal do marco 1;
reportar a tarefa primária como resultado negativo, documentado aqui e em
`configs/base.yaml`.

### Ajuste adicional necessário até narma5 funcionar: reescala da injeção

Mesmo em narma5, com a injeção *literal* do enunciado (`u ~ Uniform(0,0.2)` injetado
direto via `theta = pi*(u+1)/2`), o piso de erro ficava em **NMSE≈0.99** -- outro nulo
efetivo. Causa: `u∈[0,0.2]` mapeia para `theta∈[pi/2, 0.6·pi]`, um arco de ~10° na
esfera de Bloch. A dispersão das features exatas resultante (`std≈0.003-0.04`) fica
muito abaixo do ruído de 64 shots (`std≈1/sqrt(64)≈0.125`) -- o readout treinado com
shots reais não consegue distinguir sinal de ruído (razão sinal-ruído < 1), mesmo que
o mesmo reservatório, com features exatas, decodifique bem (NMSE exato ≈0.16).

Correção implementada em `src/qrcshots/tasks.py` (`encode_for_injection`): a
recorrência de `y` e o valor de `y` continuam calculados sobre o `u` original
(`Uniform(0,0.2)`); só o sinal **injetado no reservatório** é reescalado para
`u' = (u - centro)/semi_faixa`, cobrindo `theta` em `[0, pi]` por completo. Isso é uma
mudança na codificação (parte do "reservatório", que a Seção 13 explicitamente
autoriza ajustar), não na tarefa: `y` nunca muda. Depois da reescala + um re-ajuste de
`tau`/`J` (mesmo procedimento, só treino/validação), o piso caiu para **NMSE=0.664**
-- o regime usado na grade completa.

### `product3` (segunda tarefa secundária, Seção 2): também nulo

Checagem barata (piso + readout exato-treinado, sem grade de 30 seeds -- a mesma
lógica do diagnóstico da tarefa primária), com a config final (`tau=2, J=8`,
reescala de injeção não aplicável pois `u∈[-1,1]` já cobre o range completo):

- Piso (readout ruidoso): **NMSE=1.0021**.
- Readout treinado com features exatas (teste): **NMSE=1.0028**.

Também nulo, e pela mesma razão estrutural: `product3` (`y_t=u_{t-1}·u_{t-3}·u_{t-7}`)
exige memória de **7 passos**, mais longa que a lag-5 da tarefa primária (que já
trava em NMSE~0.85 de memória linear isolada -- Seção acima) -- não rodamos a grade
completa aqui porque o piso já mostra que não há sinal a proteger. As duas tarefas
secundárias do enunciado foram checadas; só `narma5` mostrou sinal decodificável.

## 1. Com o mesmo orçamento, qual estratégia tem menor erro? Qual o tamanho do efeito?

| B | uniform | shap | variance | magnitude | random | oracle |
|---|---|---|---|---|---|---|
| 96 | 0.9166 ± 0.0307 | 0.8840 ± 0.0253 | 0.8846 ± 0.0260 | 0.8840 ± 0.0253 | 0.9052 ± 0.0243 | 0.8807 ± 0.0231 |
| 150 | 0.8248 ± 0.0163 | 0.8068 ± 0.0141 | 0.8080 ± 0.0140 | 0.8069 ± 0.0141 | 0.8808 ± 0.0290 | 0.8070 ± 0.0141 |
| 300 | 0.7431 ± 0.0110 | 0.7331 ± 0.0112 | 0.7326 ± 0.0105 | 0.7334 ± 0.0115 | 0.8521 ± 0.0277 | 0.7327 ± 0.0106 |
| 600 | 0.7033 ± 0.0073 | 0.6980 ± 0.0065 | 0.6982 ± 0.0066 | 0.6979 ± 0.0064 | 0.7131 ± 0.0098 | 0.6980 ± 0.0068 |
| 1200 | 0.6850 ± 0.0069 | 0.6820 ± 0.0062 | 0.6818 ± 0.0064 | 0.6818 ± 0.0062 | 0.7008 ± 0.0088 | 0.6817 ± 0.0064 |
| 2400 | 0.6747 ± 0.0039 | 0.6735 ± 0.0031 | 0.6733 ± 0.0029 | 0.6736 ± 0.0030 | 0.6738 ± 0.0033 | 0.6732 ± 0.0030 |

(média ± desvio-padrão do NMSE de teste, 30 seeds; tabela completa com IC bootstrap
95% em `figures/tables/nmse_summary.csv`)

`shap`, `variance`, `magnitude` e `oracle` empatam entre si em todos os orçamentos
(diferenças bem dentro de 1 desvio-padrão) e batem **`uniform` de forma consistente e
crescente conforme o orçamento cai**: a redução de NMSE frente a `uniform` é de
**+3.6% relativo em B=96** (0.9166→0.8846), **+2.1% em B=150**, **+1.4% em B=300**,
caindo a ~0.2% em B≥1200 conforme tudo converge ao piso. Os ICs bootstrap de
`uniform` e `variance`/`shap` não se sobrepõem em B∈{96,150,300}, então a diferença é
estatisticamente resolvida com 30 seeds nesse regime.

`random` é a estratégia claramente pior em todos os orçamentos abaixo do piso (ex.
NMSE=0.8521 em B=300, pior até que `uniform`=0.7431) -- confirma que **qualquer**
desbalanceamento não ajuda; o desbalanceamento precisa ir na direção certa.

`oracle` (limite superior, não-implementável -- ver Seção 8) fica sistematicamente à
frente ou empatado com as estratégias implementáveis, como esperado, mas a margem
sobre `variance`/`shap` é pequena (ex. 0.8807 vs 0.8846 em B=96) -- pouca margem
disponível fica sem ser capturada pelas estratégias reais.

## 2. Para atingir um NMSE alvo, quantos shots cada estratégia precisa?

Lendo a tabela 1 na direção inversa: para atingir **NMSE≈0.81**, `uniform` precisa de
`B≈150`; `shap`/`variance`/`magnitude`/`oracle` atingem o mesmo NMSE com `B≈96-110`
(interpolando) -- uma economia de **~30-40% dos shots de previsão** nesse ponto. Para
atingir **NMSE≈0.70**, `uniform` precisa de `B≈650` (interpolado), enquanto as
estratégias informadas já estão lá com `B≈550-600` -- economia menor (~10-15%),
porque a curva já está achatando perto do piso. Em `B≥1200` a economia desaparece: o
piso do reservatório (0.664) domina e não há mais o que ganhar redistribuindo.

## 3. SHAP oferece alguma vantagem sobre a decisão direta por incerteza? Em que regime?

**Não, nesta tarefa.** Teste pareado Wilcoxon signed-rank (`shap` vs `variance`,
30 seeds, por orçamento):

| B | estatística | p-valor | tamanho de efeito (rank-biserial) |
|---|---|---|---|
| 96 | 185.0 | 0.339 | −0.20 |
| 150 | 153.0 | 0.105 | −0.34 |
| 300 | 166.0 | 0.177 | +0.29 |
| 600 | 205.0 | 0.584 | −0.12 |
| 1200 | 211.0 | 0.670 | +0.09 |
| 2400 | 192.0 | 0.416 | +0.17 |

Nenhum p-valor cruza 0.05; os tamanhos de efeito trocam de sinal entre orçamentos
(ruído, não tendência real) -- não há diferença estatística entre `shap` e `variance`
em nenhum orçamento.

**Explicação:** ver `figures/ranking_divergence.png`. Nesta tarefa e reservatório, o
ranking de grupos por importância SHAP (`Z > X > Y`... na verdade `X > Z > Y`, ver
número exato abaixo) e o ranking por incerteza de medição `s_g` **coincidem
perfeitamente** (Kendall tau = 1.00 entre os dois rankings). `s_g` (passe único):
`X=0.0314, Z=0.0214, Y=0.0112`; importância SHAP: `X=0.00588, Z=0.00458, Y=0.00215` --
mesma ordem `X > Z > Y` nas duas métricas. Pela hipótese central do enunciado
(Seção 0): quando os rankings **coincidem**, `shap` e `variance` devem ter desempenho
equivalente -- exatamente o que se observa. Isso é reportado como resultado válido,
não como falha do experimento: a tarefa escolhida não é um contraexemplo à hipótese,
é uma confirmação do caso em que ela prevê equivalência.

## 4. O ganho sobrevive à contabilidade de custo total? A partir de quantas previsões?

Custo de calibração de `variance` (passe único, protocolo realmente implantável --
ver nota abaixo): **206.784 shots** (64 shots × 3 grupos × 1077 janelas de
calibração). Custo de coleta inicial (compartilhado por todas as estratégias, para
treinar o readout): 711.936 shots.

Break-even (nº de previsões `N*` para `variance` compensar seu custo extra de
calibração frente a `uniform`, mesmo NMSE):

| B | NMSE variance | orçamento equivalente de uniform | economia marginal/previsão | N* |
|---|---|---|---|---|
| 96 | 0.8846 | 114.8 | 18.8 | 10.990 |
| 150 | 0.8080 | 180.8 | 30.8 | 6.719 |
| 300 | 0.7326 | 379.0 | 79.0 | 2.617 |
| 600 | 0.6982 | 769.3 | 169.3 | 1.222 |
| 1200 | 0.6818 | 1572.1 | 372.1 | **556** |
| 2400 | 0.6733 | -- | -- | ∞ (curvas convergem, sem margem) |

Em orçamentos de previsão maiores (`B≥600`), o break-even é rápido (poucos milhares
ou até só **556 previsões** em B=1200) porque a economia marginal por previsão cresce
com `B`. Em orçamentos pequenos (`B=96`), embora a vantagem relativa de NMSE seja
maior, a economia absoluta em shots por previsão é pequena, então são necessárias
~11 mil previsões para amortizar a calibração. **Sim, o ganho sobrevive à
contabilidade total**, mas só depois de um número de previsões que varia em ~20x
dependendo do orçamento de shots por previsão.

> **Nota metodológica importante:** o custo de calibração usado aqui é o protocolo de
> **passe único** (`s_g` estimado de um lote de `n_cal_shots` por janela, sem repetir
> -- ver `allocate.s_g_single_pass_empirical`), que é o que um deployment real
> pagaria. A Seção 6.2 do enunciado também descreve um protocolo de **R=200
> repetições da mesma janela**, mas esse é o método de **checagem de sanidade**
> contra `s_g` exato (ambos concordam: ver `figures/tables/` e testes em
> `tests/test_allocate.py`), não uma alocação real -- custaria 200x mais
> (41.356.800 shots) e tornaria o break-even artificialmente infinito em todo B.

## 5. Quanto do erro é ruído de medição e quanto é limitação do reservatório?

Decomposição `NMSE_total ≈ NMSE_piso + contribuição_shot_noise` (verificada
numericamente; tabela completa em `figures/tables/nmse_decomposition.csv`), piso =
0.6644:

| B | fração do piso (reservatório) | fração de shot noise |
|---|---|---|
| 96 | ~72-75% | ~25-28% |
| 150 | ~80-82% | ~18-20% |
| 300 | ~91% | ~9% |
| 600 | ~95% | ~5% |
| 1200 | ~97.4% | ~2.6% |
| 2400 | ~98.6% | ~1.4% |

Em `B≤150` o ruído de shot é uma fração substancial do erro total (18-28%) -- é
exatamente aí que a estratégia de alocação tem influência visível (Q1). A partir de
`B≈300` o piso do reservatório já domina (>90%), e a partir de `B≥1200` a alocação
praticamente não importa mais (a diferença entre a melhor e a pior estratégia
implementável cai para ~0.3% do NMSE total). Isso é consistente com o aviso da
Seção 9: o resultado nulo ("alocação não importa") é o mais provável em orçamentos
altos, e aparece exatamente onde a teoria prevê.

**Verificação independente da aditividade** (a definição acima, por si só, não pode
falhar -- `shot_noise_contribution` é definida como `mean_nmse - floor_nmse`; isso não
verifica nada). Calculamos separadamente uma previsão **analítica** do excesso de
shot noise, usando exatamente o mesmo `s_g` exato (via `Sigma_g` do teste) e pesos `w`
que alimentam a alocação, mas nunca olhando para o NMSE observado:
`excesso_previsto(B,alocação) = (1/Var(y_teste)) · sum_g s_g_teste^2 / n_g`. Comparado
ao excesso observado (`mean_nmse - floor_nmse`) nas 36 combinações
(orçamento×estratégia) da grade: razão observado/previsto entre **0.95 e 1.05** em
todas, correlação (log-log) **0.9998** (`figures/tables/nmse_additivity_verification.csv`).
A aditividade não é só uma definição de conveniência -- é uma previsão quantitativa
independente que os dados confirmam, e que valida a consistência de toda a
maquinaria de `s_g`/alocação com o NMSE de fato observado.

**Custo de preparação (`shots*L`, Seção 3):** cada shot exige reproduzir os `L=20`
passos da janela. Em operações de preparação: coleta inicial =
`711.936 * 20 = 14.238.720`; calibração de `variance` (passe único) =
`206.784 * 20 = 4.135.680`; por previsão, `shots*L` = `1.920` (B=96) a `48.000`
(B=2400). A ordem relativa dos custos entre coleta/calibração/previsão não muda ao
multiplicar por `L` (é um fator comum), mas o custo absoluto de preparação é a
métrica relevante se o hardware cobra por operação, não só por shot de leitura.

## 6. Como as três estratégias se comparam ao oráculo e ao `kernel_readout`?

`shap`, `variance`, `magnitude` e `oracle` formam um grupo muito próximo em todos os
orçamentos (Q1); a margem do oráculo sobre as estratégias implementáveis nunca passa
de ~0.4% de NMSE relativo, então **as estratégias informadas já capturam quase toda a
margem disponível** dado o mesmo readout.

`kernel_readout` (Seção 8, ver `docs/baseline_kernel.md` e
`scripts/run_kernel_readout.py`) usa um observável de leitura **diferente**, ótimo
segundo o kernel de Hilbert-Schmidt `K(rho_a,rho_b)=tr(rho_a rho_b)` sobre a matriz
densidade **completa** (16x16, todos os pesos de Pauli 0-4), não só os 12 observáveis
de peso 1:

| Método | Acesso à informação | NMSE (teste, shots→∞) |
|---|---|---|
| Ridge do marco 1 (treinado com features exatas) | só os 12 Paulis de peso 1 | 0.4559 |
| `kernel_readout`, `O*` completo (sem restrição) | matriz densidade completa | **0.2167** |
| `kernel_readout`, `O*` projetado no suporte medível | só os 12 Paulis de peso 1, mas com coeficientes vindos do ajuste conjunto | 2.8896 |

Achados, respondendo os 3 itens da Seção 8:

- **(a)** o `O*` sem restrição quase **dobra a precisão** do ridge do marco 1
  (NMSE 0.217 vs 0.456) -- há muito mais informação sobre `y` na matriz densidade
  completa do que nos 12 observáveis de peso 1 que conseguimos medir com os três
  grupos compatíveis.
- **(b)** **97.6% da norma de Hilbert-Schmidt de `O*`** vive fora do suporte de peso 1
  (`fraction_outside_support=0.9762`) -- a restrição a grupos de Pauli compatíveis
  descarta a maior parte do que o observável ótimo usaria; a informação decisiva está
  em correlações de peso ≥2 (ex. `Z_0 Z_1`), que exigiriam medir pares de qubits em
  bases conjuntas, não observáveis de 1 qubit isoladamente.
- **(c)** **projetar** `O*` no suporte medível (jogar fora os coeficientes de peso≥2 e
  medir só o que resta com a alocação `variance`) dá um resultado **muito pior que
  trivial** (NMSE 2.89 com shots infinitos, piorando para 3.4-16.7 com shots finitos,
  Tabela abaixo) -- **muito pior que o próprio ridge do marco 1**, que usa exatamente o
  mesmo suporte de 12 features mas os ajusta *diretamente* nesse subespaço. Isso é
  esperado, não uma contradição: coeficientes otimizados *junto* com termos de peso
  alto (que se cancelam parcialmente com os de peso 1 no ajuste conjunto) ficam
  descalibrados quando esses termos são removidos depois. **Lição prática:** para essa
  arquitetura de reservatório, ajustar o readout diretamente no suporte medível
  (como o marco 1 faz) é preferível a truncar um observável ótimo irrestrito
  post-hoc.

| B | NMSE (shots finitos, alocação `variance` sobre `O*` restrito) |
|---|---|
| 96 | 16.67 ± 0.80 |
| 150 | 11.44 ± 0.41 |
| 300 | 7.13 ± 0.26 |
| 600 | 4.99 ± 0.16 |
| 1200 | 3.92 ± 0.10 |
| 2400 | 3.42 ± 0.07 |

(médias sobre 30 seeds; note que mesmo em `B=2400` o NMSE finito, 3.42, é muito pior
que o já ruim NMSE exato, 2.89 -- o piso deste baseline restrito domina completamente
sobre o efeito de shot noise, então a comparação de estratégias de alocação não é
informativa aqui; reportado por completude, não como alternativa competitiva.)

## Diagnóstico de errors-in-variables (apêndice, Seção 5)

Readout treinado com features ruidosas (64 shots/grupo): `||w||=0.00571`. Readout
treinado com features exatas: `||w||=0.00721`. O readout ruidoso tem pesos ~21%
menores em norma -- atenuação clássica de errors-in-variables (features ruidosas
levam a pesos regularizados mais para baixo). NMSE de validação do readout exato:
0.467 (bem melhor que o piso do readout ruidoso, 0.664 no teste) -- confirma que a
maior parte do "piso" reportado nas seções acima vem do ruído de 64 shots na coleta
de treino, não de uma limitação fundamental do reservatório em si. Isso não entra na
comparação principal (Seção 5), só documentado aqui.

## Fase 2: aprendizado conjunto de pesos e alocação (Seção 10)

Rodado em `B=300` shots/janela (orçamento de *treino*, não de teste), via
`scripts/run_phase2.py`; resumo completo em `results/raw/phase2_summary.json`.

O loop convergiu em 8 iterações (critério: `patience=3` sem melhora), com o melhor
resultado de validação na iteração 4:

| it | NMSE val | alocação (X,Y,Z) |
|---|---|---|
| 0 (uniforme) | 0.7555 | (100,100,100) |
| 1 | 0.7312 | (138,58,104) |
| 2 | 0.7142 | (159,36,105) |
| 3 | 0.6896 | (183,24,93) |
| **4 (melhor)** | **0.6832** | **(200,17,83)** |
| 5 | 0.6922 | (214,17,69) |
| 6 | 0.7003 | (217,15,68) |
| 7 | 0.6897 | (231,13,56) |

A alocação migra de uniforme para fortemente concentrada em `X` (o grupo de maior
`s_g`, consistente com Q1) nas primeiras iterações, depois oscila e piora levemente
-- overfitting do próprio processo de realocação a ruído de amostragem específico de
cada rodada de coleta, não uma tendência real; é exatamente o que `patience` existe
para pegar.

**Comparação final no teste**, mesma alocação de 300 shots/janela:

| Método | NMSE teste |
|---|---|
| Fase 1 `variance` (readout fixo, sem loop) | 0.7310 |
| Fase 2 (readout + alocação co-otimizados) | **0.7254** |
| `kernel_readout` restrito, alocação `variance`, mesmo B | 7.13 (Seção 6) |

**Ganho da fase 2 sobre a fase 1, no mesmo orçamento: 0.0056 de NMSE absoluto
(~0.77% relativo).** Custo do loop até convergir: **10.346.688 shots** de treino
extra -- ~50x o custo de calibração de `variance` na fase 1 (206.784 shots) para um
ganho marginal. Cada iteração recoleta features em **todas** as ~3700 janelas de
treino+validação do zero (`(n_ridge_train+n_val)*B = 3708*300 ≈ 1.11M` shots/iteração),
o que domina o custo total.

**Conclusão (Seção 10 pede isso explicitamente): o ganho NÃO compensa o custo
adicional.** Um ganho de <1% de NMSE por >10 milhões de shots extra de treino não se
justifica frente a rodar a fase 1 uma única vez (readout fixo) com a alocação
`variance` calculada a partir da calibração de passe único -- que já captura a maior
parte do benefício de realocar para longe de `uniform` (Seção Q1) a uma fração do
custo. O loop conjunto continua superior ao baseline `kernel_readout` restrito, mas
esse baseline já está descartado por outros motivos (Seção 6).
