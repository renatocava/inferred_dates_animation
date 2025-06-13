## Visualización de movimientos inferidos de linajes virales respiratorios

Este repositorio contiene los scripts en Python utilizados para visualizar los movimientos inferidos de linajes de virus respiratorios a nivel departamental en Perú y a nivel internacional entre países. Las visualizaciones fueron generadas con las bibliotecas Cartopy, GeoPandas, Matplotlib, Basemap y Shapely.

* Gráfica 1 (Perú):

    * Se usaron shapefiles del INEI para los límites departamentales.

    * Los nodos representan departamentos y su tamaño es proporcional al número de transiciones salientes inferidas.

    * Las flechas indican las rutas origen-destino, con grosor y transparencia proporcionales al número de eventos.

    * Se añadieron círculos grises aleatorios (~10,000 hab.) como representación de la densidad poblacional, basados en datos del MINSA.

* Gráfica 2 (Global):

    * Se utilizaron mapas base de Natural Earth Data.

    * Las flechas geodésicas representan transiciones inferidas entre países.
  
    * El grosor de cada flecha indica la cantidad de transiciones inferidas en esa ruta.
  
    * El color de cada flecha representa el linaje viral correspondiente.