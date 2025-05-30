import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
from datetime import datetime, timedelta
import calendar
from matplotlib.collections import LineCollection
from matplotlib.patches import FancyArrowPatch
import cartopy.crs as ccrs
from unidecode import unidecode
import matplotlib.dates as mdates
import matplotlib.cm as cm

def decimal_to_datetime(dec_year):
    """
    Convierte un año en formato decimal (e.g. 2020.2158) a datetime.
    """


    year_int = int(dec_year)
    rem = dec_year - year_int
    days = 366 if calendar.isleap(year_int) else 365
    return datetime(year_int, 1, 1) + timedelta(days=rem * days)

def norm_dep(nombre: str) -> str:
    """
    • Quita tildes: Junín → Junin
    • Pasa a mayúsculas: Junin → JUNIN
    • Colapsa espacios: Madre   De  Dios → MADRE DE DIOS
    """
    return " ".join(unidecode(nombre).upper().split())

def bezier(p1, p2, offset_scale=0.0):
    """Crea una curva de Bézier entre p1 y p2 con desplazamiento perpendicular opcional."""
    import numpy as np

    # Convertir a array
    p1, p2 = np.array(p1), np.array(p2)

    # Punto medio
    mid = (p1 + p2) / 2

    # Vector perpendicular normalizado
    v = p2 - p1
    perp = np.array([-v[1], v[0]])  # rotación 90°
    perp = perp / np.linalg.norm(perp)

    # Aplicar desplazamiento
    control = mid + offset_scale * perp

    # Curva Bézier (cuadrática)
    t = np.linspace(0, 1, 100)[:, None]
    curve = (1 - t)**2 * p1 + 2 * (1 - t) * t * control + t**2 * p2
    return curve

# def curved_segment(start, end, n=50, amp=0.3):
#     """
#     Devuelve una curva Bezier simple (array Nx2) entre start y end
#     curvada con un seno de amplitud 'amp'.
#     """
#
#     x = np.linspace(start[0], end[0], n)
#     y = np.linspace(start[1], end[1], n) + amp * np.sin(np.pi * np.linspace(0, 1, n))
#     return np.column_stack((x, y))

# def precompute_beziers(df, coords, n=50, amp=0.3):
#     """
#     Crea un dict {(origin, dest): array Nx2} para todas las combinaciones
#     presentes en df y existentes en coords.
#     """
#     beziers = {}
#     for _, r in df.iterrows():
#         key = (r["ParentRegion"], r["ChildRegion"])
#         if key not in beziers and all(k in coords for k in key):
#             beziers[key] = curved_segment(coords[key[0]], coords[key[1]],
#                                           n=n, amp=amp)
#     return beziers

def prepare_coords(shapefile, proj="EPSG:32718"):
    """
    Lee el shapefile de departamentos y devuelve:
        gdf    → GeoDataFrame en WGS-84
        coords → dict  { 'Lima': (lon, lat), ... }  con centroides proyectados.
    """


    gdf = gpd.read_file(shapefile).to_crs("EPSG:4326")
    gdf_utm = gdf.to_crs(proj)                 # proyección métrica
    cent = gdf_utm.centroid.to_crs("EPSG:4326")
    coords = {dep: (p.x, p.y) for dep, p in zip(gdf["NOMBDEP"], cent)}
    return gdf, coords

def make_base_map(gdf, coords, extent=(-84, -68.2, -20.2, 0.2),
                  figsize=(8, 10), dpi=100):
    """
    Crea figura y eje cartográfico con el contorno de Perú.
    Devuelve fig, ax.
    """

    # pastel = cm.get_cmap("Pastel2", 25)
    # dep_list = sorted(coords.keys())  # 25 nombres normalizados
    # dep_color = {dep: pastel(i) for i, dep in enumerate(dep_list)}



    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.add_geometries(gdf.geometry, ccrs.PlateCarree(),
                      facecolor="lightgray", edgecolor="black", lw=1, zorder=1)
    # for dep, geom in zip(gdf["NOMBDEP"], gdf.geometry):
    #     ax.add_geometries([geom], ccrs.PlateCarree(),
    #                       facecolor=dep_color.get(norm_dep(dep), "#dddddd"),
    #                       edgecolor="black", lw=0.7, zorder=1)
    ax.set_extent(extent)


    # ───────► ETIQUETAS ◄───────
    for dep, (lon, lat) in coords.items():
        # pequeño ajuste vertical según latitud
        y_off = 0.15 if lat > -10 else -0.15
        ax.text(lon, lat + y_off, dep.title(),  # .title() → “La Libertad”
                transform=ccrs.PlateCarree(),
                fontsize=6, ha="center", va="center",
                # bbox=dict(facecolor="white", alpha=.7, pad=1, edgecolor="none"),
                zorder=5)
    ax.set_title("Dispersal of virus lineages over time\nPeru")
    return fig, ax

def build_artists(ax):
    """
    Prepara los artistas dinámicos:
    puntos  → scatter vacío (se rellena en update)
    lineas  → lista de LineCollections
    flechas → lista de FancyArrowPatch
    Devuelve (puntos, lineas, flechas)
    """

    # punto dummy invisible
    puntos = ax.scatter([], [], s=[],
                        # c=np.empty((0,)),
                        # cmap="viridis",
                        facecolors=(0.886, 0.290, 0.200, 0.8),     # relleno
                        edgecolors="white" ,       # borde,
                        linewidth=0.6,
                        transform=ccrs.PlateCarree(), zorder=2, alpha=0.75)

    # texto dinámico de la fecha (esquina superior-izquierda del mapa)
    date_text = ax.text(0.08, 0.22, "", transform=ax.transAxes,
                        fontsize=18,
                        # fontweight="bold",
                        ha="left", va="bottom",
                        # bbox=dict(facecolor="white", alpha=.5, pad=1, lw=0),
                        zorder=1)

    lineas, flechas = [], []
    return puntos, lineas, flechas, date_text

def init_animation(puntos, lineas, flechas, date_text):
    """Función init() para FuncAnimation."""
    return [puntos, date_text] + lineas + flechas

def update_animation(frame, fechas, df, puntos, lineas, flechas,
                     beziers, coords, sm, ax, date_text):
    """
    Actualiza un fotograma de la animación.

    Parameters
    ----------
    frame   : int
        Índice del fotograma actual.
    fechas  : array-like
        Fechas únicas ordenadas (dtype=datetime64).
    df      : pandas.DataFrame
        DataFrame completo ya filtrado/normalizado.
    puntos  : PathCollection
        Scatter de destinos a actualizar.
    lineas  : list
        Lista mutable que contendrá LineCollections.
    flechas : list
        Lista mutable que contendrá FancyArrowPatch.
    beziers : dict
        {(origen, destino): array Nx2} precalculado.
    coords  : dict
        {'DEPTO': (lon, lat)}.
    sm      : ScalarMappable
        Para mapear fechas-num a RGBA.
    ax      : GeoAxes
        Eje cartográfico donde dibujar.
    """

    WINDOW_DAYS = 10  # ← tamaño de la ventana
    delta = timedelta(days=WINDOW_DAYS)

    # ---------------- fecha actual -----------------
    fecha = fechas[frame]  # Timestamp

    date_text.set_text(fecha.strftime("%Y-%m-%d"))  # ← NUEVO
    # fecha_num = mdates.date2num(fecha)  # float (días)

    # sub = df[df["ParentDate"] <= fecha]

    # Solo eventos cuya ParentDate esté entre (fecha-10 días) y la fecha actual
    sub = df[(df["ParentDate"] >= fecha - delta) &
             (df["ParentDate"] <= fecha)]

    # ---------------- puntos (destinos) -------------
    lonlat = (sub[["ParentRegion"]]
              .join(pd.Series(coords, name="coord"), on="ParentRegion")
              ["coord"].dropna().tolist())

    if lonlat:
        lons, lats = zip(*lonlat)
        puntos.set_offsets(np.column_stack((lons, lats)))

        recuento = sub["ParentRegion"].value_counts() # ← Nº de eventos ORIGEN
        sizes = np.sqrt(recuento).reindex(sub["ParentRegion"]).fillna(1).to_numpy() * 6 # hasta la fecha actual (mismo orden que lon/lat)
        puntos.set_sizes(sizes) # escala final

        # puntos.set_array(mdates.date2num(sub["ParentDate"]))
        # puntos.set_cmap("viridis")
        # puntos.set_facecolor((1, 0, 0, 0.7))  # RGBA (rojo, 70 % opacidad)
        # color de relleno (RGBA): azul 80 % opacidad
        # puntos.set_facecolor((0.10, 0.55, 0.95, 0.8))
        # borde blanco de medio punto
        # puntos.set_edgecolor("white")
        # puntos.set_linewidth(0.6)
    else:
        puntos.set_offsets(np.empty((0, 2)))
        puntos.set_sizes([])

    # ---------------- limpiar trazos previos --------
    for art in lineas + flechas:
        art.remove()
    lineas.clear()
    flechas.clear()



    for _, r in sub.iterrows():
        key = (r["ParentRegion"], r["ChildRegion"])
        if key not in beziers:
            continue

        seg = beziers[key]

        # ← NUEVO: color para ESTE arco según ParentDate
        parent_num = mdates.date2num(r["ParentDate"])
        color = sm.to_rgba(parent_num)

        lc = LineCollection([seg], colors=[color],
                            lw=1, alpha=0.8, zorder=3,
                            transform=ccrs.PlateCarree())
        ax.add_collection(lc)
        lineas.append(lc)

        idx = int(0.7 * (len(seg) - 2))
        fa = FancyArrowPatch(seg[idx], seg[idx + 1],
                             mutation_scale=8, color=color,
                             transform=ccrs.PlateCarree(), zorder=4)
        ax.add_patch(fa)
        flechas.append(fa)

    return [puntos, date_text] + lineas + flechas