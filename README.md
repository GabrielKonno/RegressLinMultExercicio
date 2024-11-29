# Modelo de Previsão de Preços de Carros

## Visão Geral
Esse projeto é uma implementação de modelo de regressão linear múltipla para previsão de preços de carros usando scikit-learn. Um exercício simples realizado para praticar o uso da biblioteca.

## Funcionalidades Principais

### Preparação dos Dados
```python
# Carregamento e encoding
df = pd.read_csv('CarPrice_Assignment.csv')
le = LabelEncoder()
# Normalização
scaler = MinMaxScaler()
df_norm = pd.DataFrame(scaler.fit_transform(df))
```

### Features Utilizadas
- Características do carro: nome, tipo de combustível, número de portas
- Especificações técnicas: potência, rpm, taxa de compressão
- Dimensões: peso, altura, largura
- Motor: localização, tipo, tamanho

### Seleção de Variáveis
Processo iterativo de remoção de variáveis através de regressão múltipla (MRLS), resultando em 13 features finais:
```python
features = ['CarName', 'peakrpm', 'horsepower', 'compressionratio', 
            'stroke', 'enginesize', 'curbweight', 'carheight', 
            'carwidth', 'enginelocation', 'drivewheel', 'carbody', 
            'doornumber']
```

### Modelagem
```python
# Split treino/teste
x_train, x_test, y_train, y_test = train_test_split(x, y, 
                                                    test_size=0.2, 
                                                    random_state=42)
# Treinamento
lr = LinearRegression()
lr.fit(x_train, y_train)
```

## Métricas de Avaliação
- R² (Coeficiente de Determinação)
- MAE (Erro Médio Absoluto)
- MSE (Erro Quadrático Médio)
- RMSE (Raiz do Erro Quadrático Médio)

## Requisitos
```bash
pip install pandas numpy seaborn matplotlib scikit-learn statsmodels
```

## Estrutura
- Arquivo principal: `regressãolinearmúltipla_scikitlearn.py`
- Dataset: `CarPrice_Assignment.csv`

## Como Usar
1. Clone o repositório
2. Instale as dependências
3. Execute o script principal:
```bash
python regressãolinearmúltipla_scikitlearn.py
```

## Melhorias Possíveis
- Feature engineering
- Tratamento de outliers
- Experimentar outros algoritmos
- Cross-validation
- Interface de usuário

## Visualizações
- Histogramas de distribuição de preços
- Análises exploratórias com seaborn
