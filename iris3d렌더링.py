import plotly.express as px

df = px.data.iris()
fig = px.scatter_3d(df, x='petal_length', y='petal_width', z='sepal_length', color = 'species')
fig.show(renderer = "browser")

