import pandas as pd
import zipfile
import wget
import os
import patoolib
from tabulate import tabulate

# descargar datos
url = 'https://mymldatasets.s3.eu-de.cloud-object-storage.appdomain.cloud/ml-1m.zip'
wget.download(url)


# extraer datos
patoolib.extract_archive("ml-1m.zip")


# mostrar archivos
print(os.listdir('ml-1m'))


# cargar los datos de los tres archivos en un dataframe
unames = ['user_id', 'gender', 'age', 'occupation', 'zip']
users = pd.read_table('ml-1m/users.dat', sep='::', header=None, names=unames)
users.head()

rnames = ['user_id', 'movie_id', 'rating', 'timestamp']
ratings = pd.read_table('ml-1m/ratings.dat', sep='::', header=None, names=rnames)
ratings.head()

mnames = ['movie_id', 'title', 'genres']
movies = pd.read_table('ml-1m/movies.dat', sep='::', header=None, names=mnames)
movies.head()

# guardar los usuarios en un csv
users.to_csv('users.csv', index=False)

# mezclar los dataframe para tener uno solo con merge y printarlo con tabulate
data = pd.merge(pd.merge(ratings, users), movies)
data.head()
print(tabulate(data, headers='keys', tablefmt='grid'))


# mostrar información sobre el dataframe
print(data.shape)

print(data.info)

# agrupar datos por título
data_title = data.groupby('title')
print(data_title.size())

# agrupar por la puntuación media usando pivot_table
mean_ratings = data.pivot_table('rating', index='title', columns='gender', aggfunc='mean')
print(mean_ratings.sample(5))

# índices que tengas al menos 250 entradas
ratings_by_title = data.groupby('title').size()
active_titles = ratings_by_title.index[ratings_by_title >= 250]
print(active_titles)

# mostrar 5 películas con ese rating de 250 entradas
mean_ratings_250 = mean_ratings.loc[active_titles]
print(mean_ratings_250.sample(5))

# usar sort_values para filtrar las películas con mejor opinión entre las mujeres
top_female_ratings = mean_ratings_250.sort_values(by='F', ascending=False)
print(top_female_ratings.head())


# mostrar datos NaN
df = pd.DataFrame({
    "weight": {"alice":68, "charles": 112},
    "height": {"bob": 168, "charles": 182}
})

print(df)

# reemplazar NaN por el valor medio de la columna
print(df.fillna(df.mean()))