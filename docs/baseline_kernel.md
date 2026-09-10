# Baseline de literatura: leitura ótima por kernel de Hilbert-Schmidt

## O que o paper diz

**Título:** *Kernel-based optimization of measurement operators for quantum reservoir
computers*. **Autores:** Markus Gross, Hans-Martin Rieser. arXiv:2602.14677.

Abstract (via WebFetch, `abs/2602.14677`):

> "Finding optimal measurement operators is crucial for the performance of quantum
> reservoir computers (QRCs), since they employ a fixed quantum feature map."

Os autores formulam o treino de QRCs (tanto sem memória — "quantum extreme learning
machines" — quanto com memória, o caso deste projeto) como uma regressão ridge de
kernel, derivam uma representação em kernel de Hilbert-Schmidt do observável de
leitura ótimo, e discutem estratégias de implementação via decomposição na base de
Pauli e diagonalização de operador, com experimentos em classificação de imagem e
previsão de séries temporais. Para sistemas com muitos qubits, o método é mais
eficiente que o treino iterativo convencional do readout.

**Lacuna documentada:** o WebFetch da página do arXiv devolveu o abstract completo e
uma descrição qualitativa do método, mas **não** expôs as equações completas (a forma
fechada de `alpha` na regressão ridge de kernel, a definição exata do "history space"
para o caso com memória, ou os detalhes da diagonalização de operador mencionada para
hardware). Não tentamos reconstruir esses detalhes por especulação. O que
implementamos abaixo é a instanciação **específica e completa** já dada no enunciado
deste projeto (Seção 8): kernel de Hilbert-Schmidt `K(rho_a,rho_b)=tr(rho_a rho_b)`,
observável ótimo `O*=sum_a alpha_a rho_a`. Essa é precisamente a forma de kernel ridge
regression padrão (`alpha = (K+lambda I)^{-1} y`) aplicada às matrizes densidade de
janela como "pontos" do espaço de kernel — consistente com o que o abstract descreve
("Hilbert-Schmidt kernel representation of the optimal readout observable"), mesmo
sem termos visto a derivação passo a passo do paper.

## Onde nossa formulação difere do paper

1. **Kernel exato vs. estimado por shots.** Ajustamos `K` e `alpha` usando `rho`
   exato (simulação), não estimativas de `tr(rho_a rho_b)` por medição. O paper trata
   do regime geral; não sabemos se os autores discutem estimação do kernel sob shots
   finitos. Aqui isso não é necessário: o objetivo é comparar (a) o quão bom É o
   observável ótimo com informação perfeita, e (c) o que sobra dele depois de
   restringi-lo ao que nossos três grupos de medição conseguem enxergar.
2. **Sem diagonalização de operador.** O paper menciona diagonalização como
   estratégia de implementação em hardware (para medir `O*` como está, sem
   decompor em Pauli). Não implementamos isso porque o projeto já fixa a medição em
   três grupos de Pauli compatíveis (Seção 4) — a via de comparação natural aqui é
   decompor `O*` na base de Pauli e projetar no suporte medível, exatamente como o
   enunciado pede.
3. **`n_qubits=4`, matriz densidade completa.** Não sabemos a escala dos experimentos
   do paper (imagem/série temporal podem usar mais qubits). Aqui os "pontos" de
   kernel são as ~2500 janelas de treino-ridge (mesmo split de `experiment.py`), cada
   uma uma matriz `16x16`.
4. **Intercepto.** Centramos `y` antes do ajuste de kernel ridge e devolvemos a média
   na predição (`y_hat = tr(O* rho_x) + mean(y_train)`), análogo ao intercepto do
   ridge padrão do projeto. Não sabemos se o paper faz o mesmo; é a escolha mais
   direta para comparar de forma justa com o ridge do marco 1 (que também tem
   intercepto).

## Implementação e resultados

Ver `src/qrcshots/kernel_readout.py` e `scripts/run_kernel_readout.py`. Resultados
numéricos (NMSE exato vs. ridge, fração de norma fora do suporte medível, NMSE com
shots finitos) em `RESULTS.md`, Seção 6.
