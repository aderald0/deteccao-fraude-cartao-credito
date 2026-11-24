# Detecção de Fraude em Cartão de Crédito

Projeto de Machine Learning voltado para a detecção de transações fraudulentas em dados reais altamente desbalanceados, cobrindo desde a preparação dos dados até a explicabilidade das decisões do modelo com SHAP.

## 1. O Problema e o Desafio do Desbalanceamento

A base utilizada (`creditcard.csv`) contém **284.807 transações** sem valores nulos, das quais **284.315 (99,8273%)** são normais e apenas **492 (0,1727%)** são fraudes. As variáveis `V1` a `V28` passaram por transformação PCA para preservação de privacidade, enquanto `Time` e `Amount` mantêm seus valores originais.

Nesse cenário, a **acurácia engana**: um modelo ingênuo que classifica todas as transações como normais atinge **99,83% de acurácia**, mas não detecta nenhuma fraude (0% de recall). Por isso, a avaliação foca na classe de fraude (`Class = 1`) utilizando:

* **Recall:** proporção de fraudes reais que o modelo conseguiu capturar (prioridade para reduzir prejuízo financeiro).
* **Precisão:** proporção de alertas de fraude que eram realmente fraudes (controla o atrito com clientes legítimos).
* **F1-Score e Curva Precision-Recall (PR-AUC):** métricas que equilibram recall e precisão em cenários desbalanceados.

## 2. Preparação dos Dados

* **Carregamento direto via URL:** o dataset é lido diretamente do link oficial do TensorFlow Storage com `pd.read_csv`, sem versionar o arquivo bruto no repositório.
* **Transformação de valor:** aplicação de `np.log1p` na coluna `Amount` (`Log_Amount`) para reduzir a assimetria dos valores.
* **Feature Engineering temporal:** criação da variável `Hour` a partir de `Time` para capturar padrões de horário em ciclos de 24 horas.
* **Divisão estratificada e Pipeline:** separação em treino (227.845 registros; 394 fraudes — 0,173%) e teste (56.962 registros; 98 fraudes — 0,172%) com `stratify=y` e padronização (`StandardScaler`) encapsulada dentro de `Pipeline` para evitar vazamento de dados (*data leakage*).

## 3. Comparação entre os Modelos (Classe Fraude)

| Modelo | Recall | Precisão | F1-Score | ROC-AUC | PR-AUC |
| :--- | :---: | :---: | :---: | :---: | :---: |
| Regressão Logística (Baseline) | 0.6531 | 0.8205 | 0.7273 | 0.9606 | 0.7363 |
| Random Forest (`balanced`) | 0.8061 | 0.8229 | 0.8144 | 0.9758 | 0.8226 |
| **XGBoost (`scale_pos_weight`)** | **0.8469** | **0.8737** | **0.8601** | **0.9746** | **0.8792** |
| XGBoost + Undersampling | 0.9184 | 0.0346 | 0.0666 | — | 0.6855 |
| XGBoost + Oversampling (SMOTE) | 0.8673 | 0.5484 | 0.6719 | — | 0.8611 |

O **XGBoost com `scale_pos_weight`** apresentou o melhor equilíbrio geral (maior F1-Score e maior PR-AUC). Nos testes de reamostragem, o *Undersampling* atingiu o maior recall (91,84%), mas gerou excesso de falsos positivos (precisão de 3,46%), enquanto o *SMOTE* manteve um desempenho competitivo (86,73% de recall e PR-AUC de 0,8611).

## 4. Limiar de Decisão e Explicabilidade (SHAP)

* **Otimização e ajuste de limiar:** o `GridSearchCV` com foco em `recall` selecionou os hiperparâmetros `learning_rate=0.05`, `max_depth=4` e `n_estimators=100` (Recall de **0,8655** na validação cruzada). No conjunto de teste:
  * **Limiar `0.5`:** Recall de **88,78%** (87 de 98 fraudes capturadas), Precisão de **17,86%** e **400 falsos positivos** (11 falsos negativos).
  * **Limiar `0.3`:** Recall de **89,80%** (88 de 98 fraudes capturadas), Precisão de **7,03%** e **1.163 falsos positivos** (10 falsos negativos).
  * **Decisão:** o limiar **`0.5`** mostrou-se mais eficiente do que `0.3` nesse modelo otimizado, pois baixar o corte para `0.3` recuperou apenas 1 fraude adicional ao custo de 763 novos falsos positivos.
* **Análise de importância e SHAP:** as variáveis **`V14`**, **`V10`**, **`V4`** e **`V20`** concentraram o maior peso na identificação de fraudes no XGBoost. Os gráficos *Beeswarm* e *Waterfall* do SHAP permitiram visualizar tanto o impacto global dessas componentes quanto auditar individualmente o que levou uma transação fraudulenta específica a ser classificada como alerta.

## 5. Evoluções em Relação ao Fluxo Base

1. **Engenharia de atributo temporal (`Hour`):** extração do ciclo de 24 horas a partir da coluna `Time`.
2. **Prevenção de Data Leakage com `ImbPipeline`:** comparação prática entre `RandomUnderSampler` e `SMOTE` aplicados estritamente nos folds de treino.
3. **Otimização do XGBoost com `GridSearchCV` + análise de matriz de confusão:** integração do pré-processamento com busca de hiperparâmetros focada em `recall` e comparação quantitativa do trade-off de falsos positivos entre os limiares `0.5` e `0.3`.

## Como Executar

1. Abra o arquivo `deteccao_fraude_cartao.ipynb` no Google Colab.
2. Execute todas as células sequencialmente (`Ambiente de execução > Executar tudo`).
