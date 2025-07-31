# 🎬 Análisis Exploratorio de Datos con Pandas – MovieLens 1M

Este repositorio contiene un script en Python para realizar un análisis exploratorio de datos utilizando el popular dataset [MovieLens 1M](https://grouplens.org/datasets/movielens/1m/). Está pensado como una práctica para aprender a usar **pandas** y otras bibliotecas útiles para la manipulación y análisis de datos.

---

🔍 Análisis incluido

    📥 Descarga automática del dataset desde una URL.

    🗃️ Extracción de archivos comprimidos .zip usando patool.

    📑 Lectura de archivos .dat con pandas.read_table().

    🔗 Fusión de múltiples DataFrames usando pd.merge().

    📊 Estadísticas agrupadas por:

        Género (pivot_table)

        Título de película (groupby)

    📌 Filtrado de películas con al menos 250 valoraciones.

    🏆 Ranking de películas mejor valoradas por mujeres.

    📉 Manejo de valores faltantes (NaN) y reemplazo con medias (fillna(df.mean())).

    📤 Exportación de datos a .csv (users.csv).

    📋 Visualización de resultados tabulados con tabulate.

🧠 Ideal para aprender

    ✅ Fundamentos prácticos de pandas para análisis de datos.

    ✅ Cómo limpiar, unir y transformar datasets reales.

    ✅ Uso de funciones clave de pandas como:

        groupby()

        pivot_table()

        merge()

    ✅ Cómo automatizar la descarga y extracción de datos desde internet.

    ✅ Exportar resultados en formatos útiles como CSV.

    ✅ Cómo mostrar datos de forma legible en consola con tabulate.

### Instalación

Instala las dependencias con:

```bash
pip install -r requirements.txt

🧑‍💻 Autor

Desarrollado por Gus como parte de su aprendizaje en Python e IA.
