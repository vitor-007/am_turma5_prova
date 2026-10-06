# Manual Didático e Magistral: 15 Provas Práticas de Machine Learning com Naive Bayes Aplicado a Microdados Educacionais e Financeiros

**Elaborado por:** Professor Doutor e Mestre em Ciência de Dados e Estatística Educacional  
**Destinado a:** Alunos e Pesquisadores de Aprendizado de Máquina, Engenharia de Dados e Estatística Aplicada  
**Algoritmo Central:** Naive Bayes Gaussiano (`GaussianNB` via Scikit-Learn) e Manipulação Tabular com Pandas  

---

## Prefácio Pedagógico e Diretrizes Didáticas ao Estudante

Estimados(as), alunos

Com grande satisfação acadêmica apresento-lhe este **Compêndio Magistral de 15 Provas Práticas de Aprendizado de Máquina**. O propósito desta obra é conduzi-lo(a), passo a passo e com rigor técnico e científico, pela construção e validação de pipelines completas de Ciência de Dados, utilizando as bases de microdados públicos mais relevantes do Brasil — o **ENADE 2023**, o **Censo Escolar da Educação Básica 2025**, o **Censo da Educação Superior 2024 (CENSUP)** e a **Prova Nacional Docente (PND) 2025** —, suplementadas pelas séries temporais e dados de mercado da API financeira **Alpha Vantage**.

O modelo probabilístico selecionado para este conjunto de avaliações é o **Classificador Naive Bayes Gaussiano**, fundamentado no clássico Teorema de Bayes:

$$P(Y \mid X_1, X_2, \dots, X_n) = \frac{P(Y) \prod_{i=1}^{n} P(X_i \mid Y)}{P(X_1, X_2, \dots, X_n)}$$

A suposição "ingênua" (*naive*) do algoritmo pressupõe a independência condicional entre cada par de atributos preditores $X_i$ e $X_j$ dada a classe $Y$. Na variante Gaussiana (`GaussianNB`), a probabilidade contínua dos atributos é modelada sob uma distribuição Normal:

$$P(X_i \mid Y = c) = \frac{1}{\sqrt{2\pi \sigma_c^2}} \exp\left( -\frac{(x_i - \mu_c)^2}{2\sigma_c^2} \right)$$

Para obter êxito e pontuação máxima em cada uma das 15 avaliações práticas, você deverá obrigatoriamente projetar e executar a **Pipeline Universal de 9 Etapas de Engenharia e Aprendizado de Máquina**:

1. **Extração e Carregamento da Fonte:** Leitura apropriada dos arquivos brutos (`.csv`, `.txt` delimitados por ponto-e-vírgula ou requisições JSON via API HTTP) preservando a integridade dos dados.
2. **Tratamento, Limpeza de Anomalias e Amostragem:** Remoção de valores ausentes/nulos, descarte de registros ruidosos ou inviáveis e seleção de uma amostra estatisticamente aceitável e equilibrada para o aprendizado.
3. **Agrupamento e Estruturação Tabular em Pandas:** Utilização de operações avançadas de Pandas (`groupby`, `merge`/`join`, `pivot_table`, filtros condicionais) para estruturar a matriz de dados.
4. **Construção do Target Classificatório:** Formulação da variável dependente alvo ($y$) com base em fundamentação conceitual sólida e regras de negócio do domínio.
5. **Seleção e Extração de Features Úteis:** Isolamento do subconjunto de variáveis preditoras ($X$) relevantes para o modelo.
6. **Codificação de Variáveis Categóricas:** Transformação consciente de atributos nominais/ordinais através de `LabelEncoder` e `OneHotEncoder` (`pd.get_dummies`).
7. **Escalonamento e Divisão Treino/Teste:** Normalização/Padronização com `StandardScaler` ou `MinMaxScaler` e separação em bases de treinamento e validação via `train_test_split(..., stratify=y)`.
8. **Treinamento do Modelo Naive Bayes:** Ajuste e convergência do classificador `GaussianNB` do repositório `sklearn.naive_bayes`.
9. **Comprovação e Avaliação de Métricas:** Validação matemática da performance preditiva empregando `accuracy_score`, `confusion_matrix` e `classification_report`.

A tabela a seguir resume a distribuição pedagógica das 15 avaliações entre os 5 domínios de dados disponibilizados:

| Módulo Pedagógico | Provas Práticas | Fonte de Dados Principal | Foco da Modelagem Preditiva |
| :--- | :--- | :--- | :--- |
| **Módulo I** | Provas 1, 2 e 3 | Microdados do ENADE 2023 | Desempenho acadêmico, situação de trabalho e percepção da formação |
| **Módulo II** | Provas 4, 5 e 6 | Microdados do Censo Escolar 2025 | Dependência administrativa, educação em tempo integral e qualificação docente |
| **Módulo III**| Provas 7, 8 e 9 | Microdados do Censo da Educação Superior 2024 | Retenção e evasão de cursos, modalidade de ensino e ações afirmativas |
| **Módulo IV** | Provas 10, 11 e 12| Microdados da Prova Nacional Docente (PND) 2025 | Aptidão em exames docentes, perfil de atuação e área do conhecimento |
| **Módulo V**  | Provas 13, 14 e 15| Alpha Vantage Stock Market API | Tendência de preços em $t+1$, indicadores técnicos (RSI) e sentimento de notícias |

---

## Módulo I: Microdados do ENADE 2023 — Avaliação da Educação Superior

Este módulo explora as variáveis socioeconômicas, demográficas e pedagógicas coletadas pelo Inep na edição de 2023 do Exame Nacional de Desempenho dos Estudantes (ENADE).

### Prova Prática 1: Predição do Nível de Desempenho Geral do Concluinte no ENADE
* **Objetivo Pedagógico:** Classificar se o estudante concluinte obterá nota geral no ENADE acima da mediana nacional, correlacionando renda familiar, escolaridade dos pais e dedicação aos estudos.
* **Fontes de Dados:** `microdados2023_arq3.txt` (Nota Geral `NT_GER`), `microdados2023_arq14.txt` (Renda `QE_I08`) e `microdados2023_arq29.txt` (Horas de Estudo `QE_I23`).
* **Roteiro Didático de Execução:**
  1. Carregue os arquivos `.txt` referentes às notas e aos questionários socioeconômicos utilizando `pd.read_csv(..., sep=';', decimal='.')`.
  2. Filtre apenas os estudantes presentes com prova válida (`TP_PRES == 555` ou equivalente). Elimine valores nulos e extraia uma amostra aleatória de 10.000 registros para garantir eficiência computacional sem perda de representatividade.
  3. Agrupe e concatene as colunas relevantes mantendo a sincronia dos índices dos concluintes.
  4. Calcule a mediana da Nota Geral (`NT_GER`). Defina o `target_desempenho`: `1` para notas acima da mediana (Alto Desempenho) e `0` para notas menores ou iguais à mediana (Baixo Desempenho).
  5. Extraia as variáveis preditoras `QE_I08` (Renda Familiar) e `QE_I23` (Horas Semanais de Estudo extraclasse).
  6. Aplique a técnica de `OneHotEncoder` com `drop='first'` nas variáveis categóricas socioeconômicas.
  7. Divida a base em 80% para treinamento e 20% para teste, mantendo a proporção das classes com `stratify=y`. Padronize as variáveis numéricas resultantes com `StandardScaler`.
  8. Instancie e treine o classificador `GaussianNB()`.
  9. Comprove a eficácia do modelo calculando a acurácia global, a matriz de confusão e o relatório de classificação contendo precisão, recall e F1-score.

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# 1. Carregamento e Extração
df_notas = pd.read_csv('microdados2023_arq3.txt', sep=';', usecols=['NT_GER', 'TP_PRES'])
df_renda = pd.read_csv('microdados2023_arq14.txt', sep=';', usecols=['QE_I08'])
df_estudo = pd.read_csv('microdados2023_arq29.txt', sep=';', usecols=['QE_I23'])

# 2. Tratamento e Amostragem
df_p1 = pd.concat([df_notas, df_renda, df_estudo], axis=1).dropna()
df_p1 = df_p1[df_p1['TP_PRES'] == 555].sample(n=10000, random_state=42)

# 4. Construção do Target Classificatório
mediana_nota = df_p1['NT_GER'].median()
df_p1['target_desempenho'] = (df_p1['NT_GER'] > mediana_nota).astype(int)

# 5. Seleção de Features
X_raw = df_p1[['QE_I08', 'QE_I23']]
y = df_p1['target_desempenho']

# 6. Encoding de Categóricos
ohe = OneHotEncoder(drop='first', sparse_output=False)
X_encoded = ohe.fit_transform(X_raw)

# 7. Divisão Treino/Teste e Escalonamento
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.20, random_state=42, stratify=y)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. Treinamento com Naive Bayes Gaussiano
model_gnb = GaussianNB()
model_gnb.fit(X_train_scaled, y_train)

# 9. Avaliação e Comprovação de Métricas
y_pred = model_gnb.predict(X_test_scaled)
print("=== RESULTADOS PROVA PRÁTICA 1 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, y_pred):.4f}")
print("\nMatriz de Confusão:\n", confusion_matrix(y_test, y_pred))
print("\nRelatório de Desempenho:\n", classification_report(y_test, y_pred))
```

* **Autoavaliação e Critério do Aluno:** O modelo conseguiu superar a acurácia de um classificador aleatório ($>50\%$)? Como a distribuição de horas de estudo influencia na probabilidade a posteriori do grupo de alto desempenho?

---

### Prova Prática 2: Classificação da Situação de Trabalho do Estudante Graduando (escolhi essa) - Vitor Reina
* **Objetivo Pedagógico:** Prever se o estudante universitário precisa trabalhar durante a graduação (`QE_I10`) combinando o financiamento/bolsa recebido (`QE_I11`) e o turno do curso (`CO_TURNO_GRADUACAO`).
* **Fontes de Dados:** `microdados2023_arq2.txt` (Turno), `microdados2023_arq16.txt` (Trabalho `QE_I10`) e `microdados2023_arq17.txt` (Bolsa/Financiamento `QE_I11`).
* **Roteiro Didático de Execução:**
  1. Efetue a carga dos arquivos mantendo a codificação de caracteres apropriada.
  2. Remova registros com respostas omissas ou códigos de não-resposta. Extraia uma amostra de 8.000 alunos.
  3. Agrupe as variáveis em um DataFrame único do Pandas.
  4. Formule o `target_trabalha`: atribua classe `1` para alunos que trabalham 20 horas semanais ou mais (categorias de trabalho formal/informal expressivas em `QE_I10`) e `0` para alunos que não trabalham ou apenas estagiam.
  5. Isole as colunas preditoras `CO_TURNO_GRADUACAO` e `QE_I11`.
  6. Aplique `LabelEncoder` em `QE_I11` para converter ordinais e `pd.get_dummies` para o turno da graduação.
  7. Divida em 75% treino e 25% teste. Aplique padronização com `StandardScaler`.
  8. Treine o modelo `GaussianNB`.
  9. Verifique se o turno noturno aumenta significativamente a probabilidade condicional de trabalho e apresente a Matriz de Confusão e a Acurácia.

```python
from sklearn.preprocessing import LabelEncoder

# 1, 2 e 3. Carga, Tratamento e Agrupamento
df_turno = pd.read_csv('microdados2023_arq2.txt', sep=';', usecols=['CO_TURNO_GRADUACAO'])
df_trab = pd.read_csv('microdados2023_arq16.txt', sep=';', usecols=['QE_I10'])
df_bolsa = pd.read_csv('microdados2023_arq17.txt', sep=';', usecols=['QE_I11'])

df_p2 = pd.concat([df_turno, df_trab, df_bolsa], axis=1).dropna().sample(n=8000, random_state=42)

# 4. Target Classificatório
# Considera categorias de trabalho efetivo
df_p2['target_trabalha'] = df_p2['QE_I10'].astype(str).str.upper().apply(lambda x: 1 if x in ['C', 'D', 'E'] else 0)

# 5 e 6. Features e Encoding
le = LabelEncoder()
df_p2['QE_I11_encoded'] = le.fit_transform(df_p2['QE_I11'].astype(str))
X_p2 = pd.get_dummies(df_p2[['CO_TURNO_GRADUACAO', 'QE_I11_encoded']], columns=['CO_TURNO_GRADUACAO'])
y_p2 = df_p2['target_trabalha']

# 7. Split e Escalonamento
X_train, X_test, y_train, y_test = train_test_split(X_p2, y_p2, test_size=0.25, random_state=42, stratify=y_p2)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8 e 9. Treinamento e Validação
gnb_p2 = GaussianNB().fit(X_train_scaled, y_train)
y_pred_p2 = gnb_p2.predict(X_test_scaled)

print("=== RESULTADOS PROVA PRÁTICA 2 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, y_pred_p2):.4f}")
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_p2))
```

* **Autoavaliação e Critério do Aluno:** Como a distribuição prévia de probabilidade $P(Y=\text{Trabalha})$ varia entre os estudantes do turno matutino e do noturno?

---

### Prova Prática 3: Predição da Avaliação da Infraestrutura Didático-Pedagógica
* **Objetivo Pedagógico:** Avaliar a percepção do concluinte sobre a infraestrutura da faculdade (`QE_I27` a `QE_I68`) em função da categoria administrativa (Pública/Privada) e modalidade (Presencial/EAD).
* **Fontes de Dados:** `microdados2023_arq1.txt` (Categoria Admin `CO_CATEGAD`, Modalidade `CO_MODALIDADE`) e `microdados2023_arq4.txt` (Itens de Infraestrutura `QE_I35` a `QE_I45`).
* **Roteiro Didático de Execução:**
  1. Realize a leitura dos dados cadastrais do curso e das respostas do questionário de percepção formativa.
  2. Elimine registros contendo valores pendentes. Extraia uma amostra equilibrada de 12.000 alunos.
  3. Calcule a média das respostas de infraestrutura física e de laboratórios (`QE_I35` a `QE_I45`, mensuradas em escala Likert de 1 a 5).
  4. Crie o `target_satisfacao`: rotule como `1` (Satisfeito) se a média for $\ge 4.0$ e `0` (Insatisfeito) se a média for $< 4.0$.
  5. Selecione `CO_CATEGAD` e `CO_MODALIDADE` como variáveis explicativas.
  6. Execute a codificação `OneHotEncoder` sobre as categorias administrativas e a modalidade.
  7. Separe a amostra em 80% treino e 20% teste. Efetue a padronização de escalas.
  8. Treine o classificador `GaussianNB`.
  9. Analise os resultados pela matriz de confusão e pela métrica F1-score.

```python
# 1, 2 e 3. Leitura e Cálculo da Média de Infraestrutura
df_arq1 = pd.read_csv('microdados2023_arq1.txt', sep=';', usecols=['CO_CATEGAD', 'CO_MODALIDADE'])
df_arq4 = pd.read_csv('microdados2023_arq4.txt', sep=';')

infra_cols = [f'QE_I{i}' for i in range(35, 46) if f'QE_I{i}' in df_arq4.columns]
df_arq4['media_infra'] = df_arq4[infra_cols].apply(pd.to_numeric, errors='coerce').mean(axis=1)

df_p3 = pd.concat([df_arq1, df_arq4['media_infra']], axis=1).dropna().sample(n=12000, random_state=42)

# 4. Target Classificatório
df_p3['target_satisfacao'] = (df_p3['media_infra'] >= 4.0).astype(int)

# 5 e 6. Features e Encoding
X_p3 = pd.get_dummies(df_p3[['CO_CATEGAD', 'CO_MODALIDADE']], columns=['CO_CATEGAD', 'CO_MODALIDADE'])
y_p3 = df_p3['target_satisfacao']

# 7, 8 e 9. Pipeline
X_train, X_test, y_train, y_test = train_test_split(X_p3, y_p3, test_size=0.20, random_state=42, stratify=y_p3)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p3 = GaussianNB().fit(X_train_scaled, y_train)
y_pred_p3 = gnb_p3.predict(X_test_scaled)

print("=== RESULTADOS PROVA PRÁTICA 3 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, y_pred_p3):.4f}")
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_p3))
```

* **Autoavaliação e Critério do Aluno:** Como o modelo lidou com o desbalanceamento de classes entre estudantes presenciais e EAD?

---

## Módulo II: Microdados do Censo Escolar 2025 — Educação Básica

Este módulo aborda os dados das escolas brasileiras de Educação Básica, focando na infraestrutura, organização administrativa, oferta de turmas de tempo integral e qualificação docente.

### Prova Prática 4: Classificação da Dependência Administrativa da Escola
* **Objetivo Pedagógico:** Determinar se um estabelecimento de ensino básico é de dependência Pública (Federal/Estadual/Municipal) ou Privada utilizando dados de porte, matrículas e localização.
* **Fontes de Dados:** `Tabela_Escola_2025` e `Tabela_Matricula_2025`.
* **Roteiro Didático de Execução:**
  1. Carregue a `Tabela_Escola_2025` e os totais de matrícula agrupados por escola na `Tabela_Matricula_2025`.
  2. Remova escolas inativas e registros sem informação de dependência (`TP_DEPENDENCIA`). Defina uma amostra de 5.000 estabelecimentos.
  3. Execute o agrupamento via `INNER JOIN` utilizando a chave primária `CO_ENTIDADE`.
  4. Formule o `target_privada`: `1` se `TP_DEPENDENCIA == 4` (Privada) e `0` se `TP_DEPENDENCIA` $\in \{1, 2, 3\}$ (Pública).
  5. Extraia as colunas `QT_MAT_BAS` (Total de Matrículas da Educação Básica), `TP_LOCALIZACAO` (Urbana/Rural) e `IN_TEMPO_INTEGRAL`.
  6. Codifique `TP_LOCALIZACAO` via `OneHotEncoder`.
  7. Divida a base em 70% treino e 30% teste. Normalize as variáveis numéricas com `MinMaxScaler`.
  8. Treine o classificador Gaussiano Naive Bayes.
  9. Apresente a matriz de confusão e valide se o porte da escola e a localização rural auxiliam na separação probabilística entre redes públicas e privadas.

```python
from sklearn.preprocessing import MinMaxScaler

# 1, 2 e 3. Extração, Agrupamento e Junção
df_escola = pd.read_csv('Tabela_Escola_2025.csv', sep=';', low_memory=False)
df_mat = pd.read_csv('Tabela_Matricula_2025.csv', sep=';', low_memory=False)

df_mat_grouped = df_mat.groupby('CO_ENTIDADE')['QT_MAT_BAS'].sum().reset_index()
df_p4 = pd.merge(df_escola, df_mat_grouped, on='CO_ENTIDADE').dropna()
df_p4 = df_p4.sample(n=min(5000, len(df_p4)), random_state=42)

# 4. Target Classificatório
df_p4['target_privada'] = (df_p4['TP_DEPENDENCIA'] == 4).astype(int)

# 5 e 6. Features e Encoding
X_p4 = pd.get_dummies(df_p4[['QT_MAT_BAS', 'TP_LOCALIZACAO']], columns=['TP_LOCALIZACAO'])
y_p4 = df_p4['target_privada']

# 7. Split e Normalização (MinMaxScaler)
X_train, X_test, y_train, y_test = train_test_split(X_p4, y_p4, test_size=0.30, random_state=42, stratify=y_p4)
norm = MinMaxScaler()
X_train_norm = norm.fit_transform(X_train)
X_test_norm = norm.transform(X_test)

# 8 e 9. Treinamento e Avaliação
gnb_p4 = GaussianNB().fit(X_train_norm, y_train)
y_pred_p4 = gnb_p4.predict(X_test_norm)

print("=== RESULTADOS PROVA PRÁTICA 4 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, y_pred_p4):.4f}")
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_p4))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_p4))
```

* **Autoavaliação e Critério do Aluno:** Por que o algoritmo Naive Bayes Gaussiano pode se beneficiar do escalonamento `MinMaxScaler` quando há atributos com amplitudes muito díspares (como total de matrículas vs variáveis dummy)?

---

### Prova Prática 5: Classificação da Oferta de Educação em Tempo Integral
* **Objetivo Pedagógico:** Prever se a escola possui turmas de educação em tempo integral com base na localização geográfica e dependência administrativa.
* **Fontes de Dados:** `Tabela_Escola_2025` e `Tabela_Turma_2025`.
* **Roteiro Didático de Execução:**
  1. Realize a carga das tabelas de escolas e de turmas do Censo Escolar 2025.
  2. Filtre apenas estabelecimentos em atividade funcional (`TP_SITUACAO_FUNCIONAMENTO == 1`).
  3. Agrupe a `Tabela_Turma_2025` por `CO_ENTIDADE` identificando escolas com turmas de tempo integral (`IN_TEMPO_INTEGRAL == 1`).
  4. Crie o `target_integral`: `1` se a escola possui oferta em tempo integral e `0` caso contrário.
  5. Selecione as features `TP_LOCALIZACAO`, `TP_DEPENDENCIA` e `CO_UF`.
  6. Aplique `LabelEncoder` em `CO_UF` e `OneHotEncoder` em `TP_DEPENDENCIA`.
  7. Efetue o split 80/20 com estratificação e aplique `StandardScaler`.
  8. Ajuste o modelo `GaussianNB`.
  9. Reporte a Acurácia e a Curva ROC-AUC.

```python
# 1, 2 e 3. Carga e Agrupamento
df_turma = pd.read_csv('Tabela_Turma_2025.csv', sep=';', low_memory=False)
integral_per_escola = df_turma.groupby('CO_ENTIDADE')['IN_TEMPO_INTEGRAL'].max().reset_index()

df_p5 = pd.merge(df_escola, integral_per_escola, on='CO_ENTIDADE').dropna()
df_p5 = df_p5.sample(n=min(6000, len(df_p5)), random_state=42)

# 4. Target
df_p5['target_integral'] = df_p5['IN_TEMPO_INTEGRAL'].astype(int)

# 5 e 6. Features e Encoding
le_uf = LabelEncoder()
df_p5['CO_UF_enc'] = le_uf.fit_transform(df_p5['CO_UF'].astype(str))
X_p5 = pd.get_dummies(df_p5[['CO_UF_enc', 'TP_LOCALIZACAO', 'TP_DEPENDENCIA']], columns=['TP_LOCALIZACAO', 'TP_DEPENDENCIA'])
y_p5 = df_p5['target_integral']

# 7, 8 e 9. Pipeline
X_train, X_test, y_train, y_test = train_test_split(X_p5, y_p5, test_size=0.20, random_state=42, stratify=y_p5)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p5 = GaussianNB().fit(X_train_scaled, y_train)
y_pred_p5 = gnb_p5.predict(X_test_scaled)

print("=== RESULTADOS PROVA PRÁTICA 5 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, y_pred_p5):.4f}")
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_p5))
```

* **Autoavaliação e Critério do Aluno:** O viés de prevalência regional de escolas integrais afetou as probabilidades *a priori* recalculadas pelo modelo?

---

### Prova Prática 6: Classificação do Perfil de Qualificação do Corpo Docente
* **Objetivo Pedagógico:** Classificar se a escola possui um corpo docente com alta proporção de professores com nível superior completo.
* **Fontes de Dados:** `Tabela_Docente_2025` e `Tabela_Escola_2025`.
* **Roteiro Didático de Execução:**
  1. Carregue a `Tabela_Docente_2025`.
  2. Calcule a proporção de docentes com ensino superior completo por escola (`CO_ENTIDADE`).
  3. Una os dados com a `Tabela_Escola_2025`. Selecione amostra de 5.000 escolas.
  4. Construa o `target_alta_qualificacao`: `1` se o percentual de docentes com superior completo for $> 85\%$ e `0` caso contrário.
  5. Extraia as variáveis explicativas de dependência administrativa e localização.
  6. Execute codificação One-Hot para os atributos categóricos.
  7. Divida em 75% treino e 25% teste. Padronize os dados.
  8. Ajuste o classificador `GaussianNB`.
  9. Apresente as métricas de validação cruzada simples e relatório de precisão e recall.

```python
# 1 e 2. Carga do Censo Docente Escolar
df_doc = pd.read_csv('Tabela_Docente_2025.csv', sep=';', low_memory=False)
doc_qualif = df_doc.groupby('CO_ENTIDADE')['TP_ESCOLARIDADE'].apply(lambda x: (x >= 4).mean()).reset_index()
doc_qualif.columns = ['CO_ENTIDADE', 'pct_superior']

# 3, 4 e 5. Agrupamento e Target
df_p6 = pd.merge(df_escola, doc_qualif, on='CO_ENTIDADE').dropna().sample(n=min(5000, len(df_escola)), random_state=42)
df_p6['target_alta_qualificacao'] = (df_p6['pct_superior'] > 0.85).astype(int)

X_p6 = pd.get_dummies(df_p6[['TP_DEPENDENCIA', 'TP_LOCALIZACAO']], columns=['TP_DEPENDENCIA', 'TP_LOCALIZACAO'])
y_p6 = df_p6['target_alta_qualificacao']

# 7, 8 e 9. Modelagem
X_train, X_test, y_train, y_test = train_test_split(X_p6, y_p6, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p6 = GaussianNB().fit(X_train_scaled, y_train)
print("=== RESULTADOS PROVA PRÁTICA 6 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, gnb_p6.predict(X_test_scaled)):.4f}")
```

---

## Módulo III: Microdados do Censo da Educação Superior 2024 (CENSUP)

Neste módulo, o foco orienta-se aos dados das Instituições de Ensino Superior (IES) e cursos de graduação do Brasil, analisando sustentabilidade acadêmica, retenção, evasão e modalidades de ensino.

### Prova Prática 7: Classificação da Taxa de Retenção e Evasão de Cursos
* **Objetivo Pedagógico:** Identificar cursos de graduação com alta taxa de retenção/evasão acadêmica baseando-se no fluxo entre vagas, ingressantes, matrículas e concluintes.
* **Fontes de Dados:** `microdados_cadastro_cursos_2024.csv` (Módulo Curso do CENSUP 2024).
* **Roteiro Didático de Execução:**
  1. Faça a leitura do cadastro dos cursos do Censo Superior 2024 delimitado por ponto-e-vírgula.
  2. Remova cursos sem alunos vinculados ou sem oferta de vagas.
  3. Realize agrupamentos por Categoria Administrativa da IES (`CO_CATEGAD`) e Área do Curso (`CO_CINE_AREA_GERAL`).
  4. Formule o cálculo da taxa de sucesso de concluintes e o `target_alta_evasao`:
     $$\text{Taxa de Sucesso} = \frac{\text{QT\_CONCLUINTE}}{\text{QT\_INGRESSO} + 1}$$
     Classifique como `1` (Alta Evasão/Retenção) se a Taxa de Sucesso for $< 0.25$, e `0` (Baixa Evasão) se $\ge 0.25$.
  5. Extraia as variáveis `QT_VAGA_TOTAL`, `TP_MODALIDADE_ENSINO` e `TP_TURNO`.
  6. Aplique `OneHotEncoder` em `TP_MODALIDADE_ENSINO` e `TP_TURNO`.
  7. Separe a base em 75% treino e 25% teste. Aplique `StandardScaler`.
  8. Ajuste o classificador `GaussianNB`.
  9. Apresente a precisão na detecção de cursos com alta taxa de retenção e discuta a utilidade prática da matriz de confusão para gestores acadêmicos.

```python
# 1 e 2. Carga dos Cursos do Censo Superior 2024
df_cursos = pd.read_csv('microdados_cadastro_cursos_2024.csv', sep=';', low_memory=False)
df_cursos = df_cursos[df_cursos['QT_INGRESSO'] > 0].dropna()

# 4. Target Classificatório
df_cursos['taxa_sucesso'] = df_cursos['QT_CONCLUINTE'] / (df_cursos['QT_INGRESSO'] + 1)
df_cursos['target_alta_evasao'] = (df_cursos['taxa_sucesso'] < 0.25).astype(int)

# 5 e 6. Features e Encoding
X_p7 = pd.get_dummies(df_cursos[['QT_VAGA_TOTAL', 'TP_MODALIDADE_ENSINO', 'TP_TURNO']], 
                      columns=['TP_MODALIDADE_ENSINO', 'TP_TURNO'])
y_p7 = df_cursos['target_alta_evasao']

# 7, 8 e 9. Split, Escalonamento e Avaliação
X_train, X_test, y_train, y_test = train_test_split(X_p7, y_p7, test_size=0.25, random_state=42, stratify=y_p7)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p7 = GaussianNB().fit(X_train_scaled, y_train)
y_pred_p7 = gnb_p7.predict(X_test_scaled)

print("=== RESULTADOS PROVA PRÁTICA 7 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, y_pred_p7):.4f}")
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_p7))
```

* **Autoavaliação e Critério do Aluno:** Qual a sensibilidade do modelo Naive Bayes em identificar corretamente os cursos que pertencem à classe minoritária de alta evasão?

---

### Prova Prática 8: Classificação da Modalidade de Ensino do Curso
* **Objetivo Pedagógico:** Prever se um curso universitário é presencial ou EAD com base em parâmetros de carga horária, turno e estrutura mantenedora.
* **Fontes de Dados:** `microdados_cadastro_cursos_2024.csv` e `microdados_ed_sup_ies_2024.csv`.
* **Roteiro Didático de Execução:**
  1. Carregue as bases de dados de cursos e mantenedoras da Educação Superior.
  2. Elimine registros ruidosos ou cursos em extinção.
  3. Agrupe as informações pela chave `CO_IES`.
  4. Defina o `target_ead`: `1` para modalidade EAD (`TP_MODALIDADE_ENSINO == 2`) e `0` para presencial (`TP_MODALIDADE_ENSINO == 1`).
  5. Extraia a carga horária total, o número total de vagas e a organização acadêmica.
  6. Converta a organização acadêmica com `LabelEncoder`.
  7. Divida em 80% treino e 20% teste. Padronize com `StandardScaler`.
  8. Ajuste o classificador `GaussianNB`.
  9. Exiba o relatório de classificação detalhado por classe.

```python
# Pipeline da Prova 8
df_p8 = df_cursos[df_cursos['TP_MODALIDADE_ENSINO'].isin([1, 2])].copy()
df_p8['target_ead'] = (df_p8['TP_MODALIDADE_ENSINO'] == 2).astype(int)

X_p8 = pd.get_dummies(df_p8[['QT_CARGA_HORARIA_TOTAL', 'QT_VAGA_TOTAL', 'CO_ORGANIZACAO_ACADEMICA']], 
                      columns=['CO_ORGANIZACAO_ACADEMICA'])
y_p8 = df_p8['target_ead']

X_train, X_test, y_train, y_test = train_test_split(X_p8, y_p8, test_size=0.20, random_state=42, stratify=y_p8)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p8 = GaussianNB().fit(X_train_scaled, y_train)
print("=== RESULTADOS PROVA PRÁTICA 8 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, gnb_p8.predict(X_test_scaled)):.4f}")
```

---

### Prova Prática 9: Classificação do Perfil do Beneficiário de Políticas Afirmativas
* **Objetivo Pedagógico:** Prever se o estudante da Educação Superior utiliza financiamento estudantil ou bolsa de apoio com base em perfil demográfico e escolaridade de origem.
* **Fontes de Dados:** `ANEXO II - MÓDULO ALUNO 2024.pdf` (Microdados do Aluno no CENSUP 2024).
* **Roteiro Didático de Execução:**
  1. Carregue o módulo de dados dos alunos do Censo da Educação Superior 2024.
  2. Filtre e trate registros sem informação válida. Amostra de 10.000 alunos.
  3. Agrupe os registros mantendo a coerência por aluno/curso.
  4. Formule o `target_bolsista`: `1` se o aluno possui bolsa ProUni, FIES ou financiamento estadual/IES, e `0` caso contrário.
  5. Extraia variáveis demográficas: Cor/Raça (`TP_COR_RACA`), Sexo (`TP_SEXO`) e Escola de Origem (`TP_ESCOLA_CONCLUSAO_ENS_MEDIO`).
  6. Aplique `OneHotEncoder` sobre as variáveis nominais.
  7. Separe em 80% treino e 20% teste. Padronize as variáveis.
  8. Instancie e treine o algoritmo `GaussianNB`.
  9. Valide o desempenho pelas métricas de precisão, recall e matriz de confusão.

```python
# Estrutura didática da Prova 9
# Leitura hipotética compatível com o dicionário Módulo Aluno 2024
df_aluno_sup = pd.read_csv('microdados_ed_sup_aluno_2024.csv', sep=';', low_memory=False).dropna().sample(10000, random_state=42)

# Supondo coluna IN_FINANCIAMENTO_ESTUDANTIL no dicionário
df_aluno_sup['target_bolsista'] = (df_aluno_sup['IN_FINANCIAMENTO_ESTUDANTIL'] == 1).astype(int)

X_p9 = pd.get_dummies(df_aluno_sup[['TP_COR_RACA', 'TP_SEXO', 'TP_ESCOLA_CONCLUSAO_ENS_MEDIO']], 
                      columns=['TP_COR_RACA', 'TP_SEXO', 'TP_ESCOLA_CONCLUSAO_ENS_MEDIO'])
y_p9 = df_aluno_sup['target_bolsista']

X_train, X_test, y_train, y_test = train_test_split(X_p9, y_p9, test_size=0.20, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p9 = GaussianNB().fit(X_train_scaled, y_train)
print("=== RESULTADOS PROVA PRÁTICA 9 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, gnb_p9.predict(X_test_scaled)):.4f}")
```

---

## Módulo IV: Microdados da Prova Nacional Docente — PND 2025

A Prova Nacional Docente (PND), instituída no âmbito do Programa Mais Professores, avalia candidatos a professores das redes públicas de educação básica. Este módulo explora o desempenho e perfil socioeconômico de seus participantes.

### Prova Prática 10: Predição da Aptidão no Exame Teórico da PND
* **Objetivo Pedagógico:** Prever a aprovação/aptidão de um candidato na PND 2025 através de seu vetor de acertos e perfil socioeconômico do Questionário Contextual (`QC`).
* **Fontes de Dados:** Microdados da PND 2025 (`DS_VT_ACE_OBJ`) e `QUESTIONARIO_CONTEXTUAL_PND.pdf`.
* **Roteiro Didático de Execução:**
  1. Carregue o arquivo de respostas e vetores de acertos da PND 2025 (`microdados_pnd_2025_arq3.txt`).
  2. Converta a string binária do vetor de acertos `DS_VT_ACE_OBJ` (ex: `"110101..."`) na soma total de acertos do candidato. Selecione uma amostra de 8.000 participantes.
  3. Agrupe o somatório de acertos com os dados do Questionário Contextual.
  4. Formule o `target_apto`: `1` se o candidato acertou $\ge 60\%$ das questões do exame e `0` caso contrário.
  5. Extraia as variáveis do `QC`: faixa etária, cor/raça e renda familiar.
  6. Execute `OneHotEncoder` sobre as variáveis socioeconômicas.
  7. Divida em 80% treino e 20% teste com `stratify=y`. Aplique `StandardScaler`.
  8. Treine o modelo `GaussianNB`.
  9. Comprove a acurácia, exiba o relatório de classificação e avalie se o histórico socioeconômico correlaciona-se com o desempenho no exame docente.

```python
# 1 e 2. Carga e Parsing do Vetor de Acertos da PND 2025
df_pnd = pd.read_csv('microdados_pnd_2025_arq3.txt', sep=';', low_memory=False).dropna()
df_pnd_sample = df_pnd.sample(n=min(8000, len(df_pnd)), random_state=42)

# Função para contar acertos na string do vetor
def calcular_acertos(vetor_str):
    return sum([1 for char in str(vetor_str) if char == '1'])

df_pnd_sample['total_acertos'] = df_pnd_sample['DS_VT_ACE_OBJ'].apply(calcular_acertos)

# 4. Target Classificatório (Considere prova com 40 questões, corte de 24 acertos)
df_pnd_sample['target_apto'] = (df_pnd_sample['total_acertos'] >= 24).astype(int)

# 5 e 6. Features e Encoding
X_p10 = pd.get_dummies(df_pnd_sample[['CO_GRUPO', 'TP_INSCRICAO_PND']], columns=['CO_GRUPO', 'TP_INSCRICAO_PND'])
y_p10 = df_pnd_sample['target_apto']

# 7, 8 e 9. Divisão, Treino e Avaliação
X_train, X_test, y_train, y_test = train_test_split(X_p10, y_p10, test_size=0.20, random_state=42, stratify=y_p10)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p10 = GaussianNB().fit(X_train_scaled, y_train)
y_pred_p10 = gnb_p10.predict(X_test_scaled)

print("=== RESULTADOS PROVA PRÁTICA 10 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, y_pred_p10):.4f}")
print("Relatório de Classificação:\n", classification_report(y_test, y_pred_p10))
```

* **Autoavaliação e Critério do Aluno:** O vetor de acertos convertido em variável contínua preservou a distribuição Gaussiana ideal exigida pelo modelo?

---

### Prova Prática 11: Classificação do Perfil Profissional do Candidato à Docência
* **Objetivo Pedagógico:** Classificar se o participante da PND já possui experiência prévia atuando como professor na Educação Básica ou se é um candidato recém-graduado sem experiência.
* **Fontes de Dados:** Questionário Contextual da PND 2025 (`QUESTIONARIO_CONTEXTUAL_PND.pdf`).
* **Roteiro Didático de Execução:**
  1. Carregue os microdados das respostas ao Questionário Contextual (`QC`).
  2. Remova inscrições sem informação sobre histórico profissional.
  3. Agrupe as declarações por perfil de vínculo.
  4. Crie o `target_experiente`: `1` para candidatos com 2 ou mais anos de docência e `0` para docentes iniciantes sem experiência.
  5. Isole idade e participação em programas de formação continuada.
  6. Aplique `LabelEncoder` sobre atributos ordinais.
  7. Divida em 75% treino e 25% teste. Aplique padronização.
  8. Ajuste o classificador `GaussianNB`.
  9. Exiba a matriz de confusão e meça a acurácia.

```python
# Pipeline didático da Prova 11
df_pnd_qc = pd.read_csv('microdados_pnd_2025_arq1.txt', sep=';', low_memory=False).dropna().sample(6000, random_state=42)

# Supondo coluna de tempo de experiência no QC
df_pnd_qc['target_experiente'] = (df_pnd_qc['TP_INSCRICAO_PND'] == 1).astype(int)

X_p11 = pd.get_dummies(df_pnd_qc[['CO_GRUPO']], columns=['CO_GRUPO'])
y_p11 = df_pnd_qc['target_experiente']

X_train, X_test, y_train, y_test = train_test_split(X_p11, y_p11, test_size=0.25, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p11 = GaussianNB().fit(X_train_scaled, y_train)
print("=== RESULTADOS PROVA PRÁTICA 11 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, gnb_p11.predict(X_test_scaled)):.4f}")
```

---

### Prova Prática 12: Predição da Área de Licenciatura do Candidato (Multiclasse)
* **Objetivo Pedagógico:** Treinar um classificador Naive Bayes Gaussiano Multiclasse para prever a área de conhecimento da licenciatura do candidato (`CO_GRUPO`: ex. Matemática, Biologia, Filosofia) através da sua percepção sobre a prova (`QPP`).
* **Fontes de Dados:** `microdados_pnd_2025_arq1.txt` e Questionário de Percepção de Prova (`QPP`).
* **Roteiro Didático de Execução:**
  1. Carregue os dados da PND contendo `CO_GRUPO` e respostas ao `QPP`.
  2. Filtre os dados mantendo apenas as 3 áreas com maior número de candidatos para um problema multiclasse de 3 rótulos.
  3. Estruture os dados em formato tabular.
  4. Crie o `target_area` com `LabelEncoder` transformando os códigos da área em rótulos `0`, `1` e `2`.
  5. Extraia as respostas dos itens do QPP (grau de dificuldade percebido e extensão da prova).
  6. Codifique os atributos de percepção.
  7. Divida em 80% treino e 20% teste com `stratify=y`.
  8. Ajuste o modelo `GaussianNB` (o algoritmo lida com problemas multiclasse de forma nativa calculando verossimilhanças para cada classe $c \in \{0, 1, 2\}$).
  9. Plote e comente a Matriz de Confusão Multiclasse $3 \times 3$.

```python
# 1, 2 e 3. Carga e Seleção Multiclasse
df_pnd_multi = pd.read_csv('microdados_pnd_2025_arq1.txt', sep=';', low_memory=False).dropna()
top_3_grupos = df_pnd_multi['CO_GRUPO'].value_counts().nlargest(3).index
df_p12 = df_pnd_multi[df_pnd_multi['CO_GRUPO'].isin(top_3_grupos)].sample(n=6000, random_state=42)

# 4. Target Multiclasse
le_target = LabelEncoder()
df_p12['target_area'] = le_target.fit_transform(df_p12['CO_GRUPO'])

# 5 e 6. Features
X_p12 = pd.get_dummies(df_p12[['TP_INSCRICAO_PND']], columns=['TP_INSCRICAO_PND'])
y_p12 = df_p12['target_area']

# 7, 8 e 9. Modelagem Multiclasse
X_train, X_test, y_train, y_test = train_test_split(X_p12, y_p12, test_size=0.20, random_state=42, stratify=y_p12)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p12 = GaussianNB().fit(X_train_scaled, y_train)
y_pred_p12 = gnb_p12.predict(X_test_scaled)

print("=== RESULTADOS PROVA PRÁTICA 12 (MULTICLASSE) ===")
print(f"Acurácia Multiclasse Obtida: {accuracy_score(y_test, y_pred_p12):.4f}")
print("Matriz de Confusão Multiclasse:\n", confusion_matrix(y_test, y_pred_p12))
print("\nRelatório de Classificação:\n", classification_report(y_test, y_pred_p12, target_names=[str(c) for c in le_target.classes_]))
```

* **Autoavaliação e Critério do Aluno:** Como o Teorema de Bayes generalizou o cálculo da classe prevista $Y^* = \arg\max_c P(Y=c) \prod P(X_i \mid Y=c)$ para três classes simultâneas?

---

## Módulo V: Inteligência de Mercado Financeiro — Alpha Vantage Stock Market API

Este módulo final conecta o estudante ao mercado financeiro em tempo real e histórico, processando cotações diárias, indicadores técnicos e métricas de sentimento de mercado obtidas via API HTTP da **Alpha Vantage**.

### Prova Prática 13: Predição da Direção do Preço de Ações no Pregão Seguinte
* **Objetivo Pedagógico:** Prever se o preço de fechamento de uma ação (ex. `IBM`) subirá ou cairá no dia seguinte ($t+1$) com base na cotação e volatilidade do dia atual ($t$).
* **Fontes de Dados:** Endpoint `TIME_SERIES_DAILY` da API Alpha Vantage (`datatype=json`).
* **Roteiro Didático de Execução:**
  1. Realize uma requisição HTTP via Python utilizando a biblioteca `requests` para recuperar dados diários da Alpha Vantage.
  2. Converta a resposta JSON no formato de DataFrame Pandas, formate os nomes das colunas e ordene a série cronologicamente com `.sort_index()`.
  3. Calcule o retorno diário do preço e a amplitude intraday:
     $$\text{return\_today}_t = \frac{\text{close}_t - \text{open}_t}{\text{open}_t}, \quad \text{amplitude}_t = \frac{\text{high}_t - \text{low}_t}{\text{open}_t}$$
  4. Aplique a operação de deslocamento temporal (`shift(-1)`) para construir o `target_direction`:
     $$\text{target\_direction}_t = \begin{cases} 1 & \text{se } \text{close}_{t+1} > \text{close}_t \\ 0 & \text{se } \text{close}_{t+1} \le \text{close}_t \end{cases}$$
  5. Extraia `return_today`, `amplitude` e o volume discretizado.
  6. Categorize o volume em 3 faixas (`['Baixo', 'Medio', 'Alto']`) com `pd.qcut` e aplique `OneHotEncoder`.
  7. Elimine a última linha (que conterá valor nulo no target devido ao `shift`). Divida a série temporal sem reembaralhar os dados históricos (`shuffle=False` na `train_test_split`). Padronize com `StandardScaler`.
  8. Instancie e treine o classificador `GaussianNB`.
  9. Avalie a acurácia em dados não vistos do teste financeiro e apresente a matriz de confusão.

```python
import requests

# 1 e 2. Requisição à API Alpha Vantage e Construção do DataFrame
url_av = 'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol=IBM&apikey=demo'
res = requests.get(url_av).json()

time_series = res.get('Time Series (Daily)', {})
df_stock = pd.DataFrame.from_dict(time_series, orient='index')
df_stock.columns = ['open', 'high', 'low', 'close', 'volume']
df_stock = df_stock.astype(float).sort_index()

# 3. Engenharia de Features
df_stock['return_today'] = (df_stock['close'] - df_stock['open']) / df_stock['open']
df_stock['amplitude'] = (df_stock['high'] - df_stock['low']) / df_stock['open']
df_stock['volume_cat'] = pd.qcut(df_stock['volume'], q=3, labels=['Baixo', 'Medio', 'Alto'])

# 4. Target Classificatório com Shift Temporal (t+1)
df_stock['target_direction'] = (df_stock['close'].shift(-1) > df_stock['close']).astype(int)
df_stock = df_stock.dropna()

# 5 e 6. Features e Encoding
X_stock = pd.get_dummies(df_stock[['return_today', 'amplitude', 'volume_cat']], columns=['volume_cat'])
y_stock = df_stock['target_direction']

# 7. Split Sequencial (Sem Shuffle Temporal) e Escalonamento
X_train, X_test, y_train, y_test = train_test_split(X_stock, y_stock, test_size=0.20, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8 e 9. Treinamento e Avaliação
gnb_stock = GaussianNB().fit(X_train_scaled, y_train)
y_pred_stock = gnb_stock.predict(X_test_scaled)

print("=== RESULTADOS PROVA PRÁTICA 13 ===")
print(f"Acurácia Financeira (IBM): {accuracy_score(y_test, y_pred_stock):.4f}")
print("Matriz de Confusão:\n", confusion_matrix(y_test, y_pred_stock))
print("\nRelatório de Desempenho:\n", classification_report(y_test, y_pred_stock))
```

* **Autoavaliação e Critério do Aluno:** Por que a premissa de `shuffle=False` é imprescindível em séries temporais financeiras para evitar vazamento de dados futuros (*data leakage*)?

---

### Prova Prática 14: Classificação da Zona Extrema do Indicador Técnico RSI
* **Objetivo Pedagógico:** Classificar se o Índice de Força Relativa (RSI) semanal de um ativo indicará zona de Sobrecompra ($>70$) ou Sobrevenda ($<30$) no período seguinte.
* **Fontes de Dados:** Endpoints `RSI` e `TIME_SERIES_WEEKLY` da API Alpha Vantage.
* **Roteiro Didático de Execução:**
  1. Efetue requisições à API solicitando o indicador técnico `RSI` (`interval=weekly`, `time_period=14`) e cotações semanais.
  2. Converta as respostas em DataFrames alinhados pelo índice de data.
  3. Agrupe e una as séries temporais.
  4. Formule o `target_rsi_zone`: `2` se $\text{RSI}_{t+1} > 70$ (Sobrecomprado), `0` se $\text{RSI}_{t+1} < 30$ (Sobrevendido) e `1` se $30 \le \text{RSI}_{t+1} \le 70$ (Neutro).
  5. Utilize como preditores o retorno semanal e a variação da Média Móvel Simples (SMA).
  6. Aplique `LabelEncoder` no Target Multiclasse.
  7. Divida a base temporal em 80% treino e 20% teste. Padronize com `StandardScaler`.
  8. Ajuste o modelo `GaussianNB`.
  9. Exiba a Matriz de Confusão Multiclasse $3 \times 3$ e avalie a acurácia nas zonas extremas.

```python
# Pipeline da Prova 14
url_rsi = 'https://www.alphavantage.co/query?function=RSI&symbol=IBM&interval=weekly&time_period=14&series_type=open&apikey=demo'
res_rsi = requests.get(url_rsi).json()

rsi_data = res_rsi.get('Technical Analysis: RSI', {})
df_rsi = pd.DataFrame.from_dict(rsi_data, orient='index').astype(float).sort_index()

# Simulando target de zonas do RSI deslocado
df_rsi['rsi_next'] = df_rsi['RSI'].shift(-1)
def rotular_rsi(valor):
    if valor > 70: return 2
    elif valor < 30: return 0
    else: return 1

df_rsi['target_rsi_zone'] = df_rsi['rsi_next'].apply(rotular_rsi)
df_rsi = df_rsi.dropna()

X_p14 = df_rsi[['RSI']]
y_p14 = df_rsi['target_rsi_zone']

X_train, X_test, y_train, y_test = train_test_split(X_p14, y_p14, test_size=0.20, shuffle=False)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

gnb_p14 = GaussianNB().fit(X_train_scaled, y_train)
print("=== RESULTADOS PROVA PRÁTICA 14 ===")
print(f"Acurácia Obtida: {accuracy_score(y_test, gnb_p14.predict(X_test_scaled)):.4f}")
```

---

### Prova Prática 15: Predição do Sentimento de Notícias de Mercado Financeiro
* **Objetivo Pedagógico:** Prever se o sentimento geral de um lote de notícias financeiras será otimista (*Bullish*) ou pessimista (*Bearish*) utilizando o score de inteligência de mercado da Alpha Vantage.
* **Fontes de Dados:** Endpoint `NEWS_SENTIMENT` da Alpha Vantage (`function=NEWS_SENTIMENT&tickers=AAPL`).
* **Roteiro Didático de Execução:**
  1. Faça a requisição HTTP ao endpoint `NEWS_SENTIMENT` para um ticker ativo (ex: `AAPL` ou `IBM`).
  2. Parseie a lista de artigos e sentimentos em JSON para um DataFrame do Pandas.
  3. Agrupe as métricas por artigo e tópico setorial.
  4. Crie o `target_sentiment`: `1` se o rótulo da notícia for `Bullish` ou `Somewhat-Bullish` e `0` se for `Bearish` ou `Somewhat-Bearish`.
  5. Extraia o valor numérico contínuo do `overall_sentiment_score` e o número de tópicos citados.
  6. Mapeie os atributos numéricos.
  7. Divida a amostra de notícias em 80% treino e 20% teste.
  8. Instancie e treine o classificador `GaussianNB`.
  9. Apresente o relatório de classificação com precisão, recall e F1-score.

```python
# 1 e 2. Requisição ao Endpoint de Inteligência de Notícias
url_news = 'https://www.alphavantage.co/query?function=NEWS_SENTIMENT&tickers=AAPL&apikey=demo'
res_news = requests.get(url_news).json()

feed = res_news.get('feed', [])
df_news = pd.DataFrame(feed)

if not df_news.empty and 'overall_sentiment_label' in df_news.columns:
    # 4. Target
    df_news['target_sentiment'] = df_news['overall_sentiment_label'].apply(
        lambda x: 1 if 'Bullish' in str(x) else 0
    )
    
    # 5. Features
    df_news['overall_sentiment_score'] = pd.to_numeric(df_news['overall_sentiment_score'], errors='coerce')
    df_news = df_news.dropna(subset=['overall_sentiment_score', 'target_sentiment'])
    
    X_p15 = df_news[['overall_sentiment_score']]
    y_p15 = df_news['target_sentiment']
    
    # 7, 8 e 9. Modelagem e Validação
    X_train, X_test, y_train, y_test = train_test_split(X_p15, y_p15, test_size=0.20, random_state=42)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    gnb_p15 = GaussianNB().fit(X_train_scaled, y_train)
    y_pred_p15 = gnb_p15.predict(X_test_scaled)
    
    print("=== RESULTADOS PROVA PRÁTICA 15 ===")
    print(f"Acurácia no Sentimento de Notícias: {accuracy_score(y_test, y_pred_p15):.4f}")
    print("Relatório de Classificação:\n", classification_report(y_test, y_pred_p15))
else:
    print("Demonstração: Endpoint de Notícias processado sem registros adicionais na chave Demo.")
```

---

## Matriz Geral de Acompanhamento das 15 Provas Práticas

A matriz síntese a seguir permite ao estudante acompanhar a conclusão de cada etapa da pipeline em cada uma das 15 avaliações:

| Prova | Base de Dados | Variável Alvo ($y$) | Atributos Preditores ($X$) | Transformação / Encoding | Métrica de Avaliação Principal |
| :---: | :--- | :--- | :--- | :--- | :--- |
| **1** | ENADE 2023 | Desempenho Geral ($>\text{mediana}$) | Renda Familiar, Horas de Estudo | `OneHotEncoder`, `StandardScaler` | Acurácia, Matriz de Confusão |
| **2** | ENADE 2023 | Situação de Trabalho (Sim/Não) | Turno do Curso, Financiamento | `LabelEncoder`, `pd.get_dummies` | Precision, Recall, F1-Score |
| **3** | ENADE 2023 | Satisfação Infraestrutura ($\ge 4.0$) | Categoria Admin, Modalidade IES | `OneHotEncoder`, `StandardScaler` | Acurácia, Report Completo |
| **4** | Censo Escolar 2025 | Dependência (Pública/Privada) | Matrículas Básicas, Localização | `pd.get_dummies`, `MinMaxScaler` | Acurácia, F1-Score |
| **5** | Censo Escolar 2025 | Oferta de Tempo Integral (Sim/Não)| Unidade da Federação, Dependência | `LabelEncoder`, `OneHotEncoder` | Curva ROC-AUC, Acurácia |
| **6** | Censo Escolar 2025 | Qualificação Docente ($>85\%$) | Dependência Admin, Localização | `OneHotEncoder`, `StandardScaler` | Acurácia, Precision |
| **7** | Censo Superior 2024| Alta Evasão no Curso ($<25\%$) | Vagas Totais, Modalidade, Turno | `pd.get_dummies`, `StandardScaler` | Acurácia, Matriz de Confusão |
| **8** | Censo Superior 2024| Modalidade (EAD/Presencial) | Carga Horária, Organização Acad. | `LabelEncoder`, `StandardScaler` | Acurácia, Report Completo |
| **9** | Censo Superior 2024| Bolsa / Ação Afirmativa (Sim/Não)| Cor/Raça, Sexo, Escola Média | `OneHotEncoder`, `StandardScaler` | Precision, Recall |
| **10**| PND 2025 | Aptidão no Exame Teórico ($\ge 60\%$)| Inscrição, Grupo do Conhecimento | `OneHotEncoder`, `StandardScaler` | Acurácia, F1-Score |
| **11**| PND 2025 | Docente com Experiência (Sim/Não) | Grupo da Licenciatura, Inscrição | `pd.get_dummies`, `StandardScaler` | Acurácia, Confusion Matrix |
| **12**| PND 2025 | Área da Licenciatura (Multiclasse) | Percepção de Prova e Inscrição | `LabelEncoder` Target, `StandardScaler` | Acurácia Multiclasse $3 \times 3$ |
| **13**| Alpha Vantage API | Direção do Preço em $t+1$ (Alta/Baixa) | Retorno Diário, Amplitude, Volume | `pd.qcut`, `OneHotEncoder` | Acurácia Sem Shuffle Temporal |
| **14**| Alpha Vantage API | Zona do RSI ($>70$, $<30$, $30-70$) | Nível Atual do RSI Semanal | `LabelEncoder` Multiclasse | Acurácia Multiclasse |
| **15**| Alpha Vantage API | Sentimento Notícia (Bullish/Bearish)| Overall Sentiment Score | `StandardScaler` | Acurácia, Precision, Recall |

---

## Conclusão e Considerações Finais do Professor

Prezado(a) aluno(a),

Ao completar a execução destas **15 Provas Práticas**, você terá percorrido todo o espectro prático da Ciência de Dados e do Aprendizado de Máquina Supervisionado. Você transformou microdados brutos em matrizes tratadas, contornou limitações de anomalias e desbalanceamento, aplicou técnicas essenciais de codificação de variáveis categóricas (`LabelEncoder` e `OneHotEncoder`), padronizou amplitudes contínuas e validou o modelo **Naive Bayes Gaussiano** em cenários reais da educação e do mercado financeiro brasileiro e internacional.

Lembre-se sempre de que o conhecimento teórico ganha vida através da prática criteriosa e metodológica. Recomendo que você guarde este compêndio como um guia permanente de consulta para seus projetos acadêmicos e profissionais futuros.

Desejo-lhe um excelente aprendizado e grande sucesso em sua trajetória científica!
