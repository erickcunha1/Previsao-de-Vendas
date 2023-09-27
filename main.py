import pandas as pd 
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

tabela = pd.read_csv('advertising.csv')

sns.heatmap(tabela.corr(), cmap='YlGnBu', annot=True)

x = tabela[['TV', 'Radio', 'Jornal']]
y = tabela['Vendas']

x_treino, x_teste, y_treino, y_teste = train_test_split(x, y, test_size=0.3)

modelo_regressao_linear = LinearRegression()
modelo_arvoredecisao = RandomForestRegressor()

modelo_regressao_linear.fit(x_treino, y_treino)
modelo_arvoredecisao.fit(x_treino, y_treino)



previsao_arvore = modelo_arvoredecisao.predict(x_teste)
previsao_regressao = modelo_regressao_linear.predict(x_teste)

print(r2_score(y_teste, previsao_arvore))
print(r2_score(y_teste, previsao_regressao))

tabela_aux = pd.DataFrame()
tabela_aux['y_teste'] = y_teste
tabela_aux['Previsao Regressao Linear'] = previsao_regressao
tabela_aux['Previsao Arvore Decisao'] = previsao_arvore

plt.figure(figsize=(15,5))
sns.lineplot(data=tabela_aux)
plt.show()

tabela_nova = pd.read_csv('novos.csv')

previsao = modelo_arvoredecisao.predict(tabela_nova)
print(previsao)
