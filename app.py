"""Salud financiera de las 10.000 empresas más grandes de Colombia · ACP y K-Means.

Ejecutar con:  streamlit run app.py
"""

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import analisis as an

st.set_page_config(
    page_title="Salud financiera de las empresas colombianas",
    page_icon="🐟",
    layout="wide",
)

# --------------------------------------------------------------------------
# Identidad visual
# --------------------------------------------------------------------------
TINTA = "#0b1f24"
TINTA_2 = "#4a5b5f"
TINTA_3 = "#86979a"
SUPERFICIE = "#fbfcfc"
REJILLA = "#e2e8e9"
MAR = "#0e7c86"
MAR_OSCURO = "#0a5560"
CORAL = "#e0533f"
ARENA = "#eef2f2"
# Orden fijo de colores para los grupos; el símbolo del marcador es la segunda
# codificación, para que la identidad del grupo no dependa solo del color.
COLORES_GRUPO = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#e34948", "#eda100", "#e87ba4", "#52514e"]
SIMBOLOS_GRUPO = ["circle", "square", "diamond", "triangle-up", "x", "cross", "triangle-down", "star"]
DIVERGENTE = [[0.0, "#1c5cab"], [0.25, "#6da7ec"], [0.5, ARENA], [0.75, "#ec8a89"], [1.0, "#b8302f"]]
SECUENCIAL = [[0.0, "#f3f9fa"], [0.35, "#9fd3d8"], [0.7, "#2f9aa3"], [1.0, "#0a5560"]]

st.markdown(
    """
    <style>
    .block-container {padding-top: 2.2rem; max-width: 1250px;}
    h1, h2, h3 {letter-spacing: -0.01em;}
    .etiqueta {color:#4a5b5f; font-size:0.78rem; text-transform:uppercase;
               letter-spacing:0.08em; margin-bottom:0.2rem;}
    .nota {background:#eef2f2; border-left:3px solid #0e7c86; padding:0.75rem 1rem;
           border-radius:4px; color:#22302f; font-size:0.95rem; margin:0.4rem 0 1rem 0;}
    .aviso {background:#fff4e8; border-left:3px solid #d07a1f; padding:0.75rem 1rem;
            border-radius:4px; color:#22302f; font-size:0.95rem; margin:0.4rem 0 1rem 0;}
    div[data-testid="stMetricValue"] {font-size:1.6rem;}
    </style>
    """,
    unsafe_allow_html=True,
)


def nota(texto, clase="nota"):
    st.markdown(f'<div class="{clase}">{texto}</div>', unsafe_allow_html=True)


def num(x, dec=2):
    """Número con coma decimal y punto de miles."""
    if pd.isna(x):
        return "—"
    return f"{x:,.{dec}f}".replace(",", "§").replace(".", ",").replace("§", ".")


def pct(x, dec=1):
    return "—" if pd.isna(x) else f"{num(100 * x, dec)} %"


def mm(x, dec=1):
    """Miles de pesos → miles de millones de pesos."""
    return num(x / 1e6, dec)


def valor(variable, x):
    """Indicador en su unidad natural."""
    formato = an.VARIABLES[variable]["formato"]
    if formato == "pct":
        return pct(x, 1)
    if formato == "veces":
        return f"{num(x, 2)} veces"
    return f"{mm(10 ** x, 1)} mil M"


def estilo(fig, alto=420, leyenda=True):
    fig.update_layout(
        height=alto,
        paper_bgcolor=SUPERFICIE,
        plot_bgcolor=SUPERFICIE,
        font=dict(family="Inter, Segoe UI, sans-serif", color=TINTA_2, size=13),
        title_font=dict(color=TINTA, size=16),
        margin=dict(l=10, r=10, t=60, b=10),
        showlegend=leyenda,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TINTA_2)),
        hoverlabel=dict(bgcolor="white", font_color=TINTA, bordercolor=REJILLA),
        separators=",.",
    )
    fig.update_xaxes(gridcolor=REJILLA, zerolinecolor="#c8d2d3", linecolor="#c8d2d3",
                     title_font=dict(color=TINTA_2))
    fig.update_yaxes(gridcolor=REJILLA, zerolinecolor="#c8d2d3", linecolor="#c8d2d3",
                     title_font=dict(color=TINTA_2))
    return fig


def mapa_calor(tabla, titulo, escala, rango=None, porcentaje=False, dec=2, alto=420):
    z = tabla.values.astype(float)
    texto = np.vectorize(lambda v: pct(v, 0) if porcentaje else num(v, dec))(z)
    fig = go.Figure(go.Heatmap(
        z=z, x=list(tabla.columns), y=list(tabla.index), colorscale=escala,
        zmin=None if rango is None else rango[0], zmax=None if rango is None else rango[1],
        text=texto, hovertemplate="%{y} · %{x}: %{text}<extra></extra>", xgap=2, ygap=2,
        colorbar=dict(thickness=12, outlinewidth=0),
    ))
    # Texto de cada celda con tinta clara u oscura según la intensidad del fondo.
    bajo, alto_r = (np.nanmin(z), np.nanmax(z)) if rango is None else rango
    for i, fila in enumerate(tabla.index):
        for j, col in enumerate(tabla.columns):
            t = (z[i, j] - bajo) / (alto_r - bajo) if alto_r > bajo else 0.5
            oscuro = t > 0.62 if escala is SECUENCIAL else abs(t - 0.5) > 0.36
            fig.add_annotation(x=col, y=fila, text=texto[i, j], showarrow=False,
                               font=dict(size=12, color="white" if oscuro else TINTA))
    fig.update_layout(title=titulo)
    fig.update_yaxes(autorange="reversed", showgrid=False)
    fig.update_xaxes(showgrid=False, side="bottom")
    return estilo(fig, alto=alto, leyenda=False)


# --------------------------------------------------------------------------
# Parámetros (barra lateral, visibles en todas las páginas)
# --------------------------------------------------------------------------
PAGINAS = [
    "1 · Contexto",
    "2 · Exploración",
    "3 · Preparación",
    "4 · Componentes principales",
    "5 · Segmentación K-Means",
    "6 · Perfil de los grupos",
    "7 · Pesca y alimentos",
    "8 · Conclusiones",
]
POR_DEFECTO = {"variables": list(an.VARIABLES), "winsor": 1.0, "componentes": 3, "k": 5, "semilla": 42}
# Nombres de los grupos en la configuración por defecto (ver sección 6).
NOMBRES_POR_DEFECTO = {
    "A": "Solventes de rentabilidad moderada",
    "B": "Grandes y apalancadas",
    "C": "Rentables y poco endeudadas",
    "D": "Bajo margen y alto volumen",
    "E": "En pérdidas",
}

# Los controles de componentes y K viven en páginas distintas: se reasignan en cada
# corrida para que Streamlit no los borre al cambiar de página.
for _clave in ("comp", "k_elegido"):
    if _clave in st.session_state:
        st.session_state[_clave] = st.session_state[_clave]

with st.sidebar:
    st.markdown("### Salud financiera de las empresas colombianas")
    st.caption("ACP y K-Means · 10.000 empresas más grandes, cierre de 2025")
    pagina = st.radio("Sección", PAGINAS, label_visibility="collapsed")
    st.divider()
    st.markdown('<div class="etiqueta">Parámetros del análisis</div>', unsafe_allow_html=True)
    seleccion = st.multiselect(
        "Indicadores que entran al modelo",
        options=list(an.VARIABLES),
        default=POR_DEFECTO["variables"],
        format_func=an.nombre,
        key="vars_sel",
        help="Mínimo cuatro. Todo el análisis se recalcula con los que elijas.",
    )
    winsor = st.slider(
        "Recorte de extremos (% por cola)", 0.0, 5.0, POR_DEFECTO["winsor"], 0.5,
        help="Winsorización: los valores por debajo del percentil p y por encima del 100 − p "
             "se reemplazan por esos percentiles.",
        key="winsor_sel",
    )
    semilla = st.number_input("Semilla de K-Means", 0, 9999, POR_DEFECTO["semilla"], 1, key="semilla_sel")
    if st.button("Volver a los valores por defecto", width="stretch"):
        for clave in ["comp", "k_elegido", "vars_sel", "winsor_sel", "semilla_sel"]:
            st.session_state.pop(clave, None)
        st.rerun()

if len(seleccion) < 4:
    st.warning("Selecciona al menos cuatro indicadores en la barra lateral para correr el análisis.")
    st.stop()


# --------------------------------------------------------------------------
# Cálculos (en caché: se repiten solo si cambian los parámetros)
# --------------------------------------------------------------------------
@st.cache_data(show_spinner="Preparando los datos…")
def correr_preparacion(variables, nivel):
    return an.preparar(nivel_winsor=nivel / 100, seleccion=variables)


@st.cache_data(show_spinner=False)
def correr_acp(z):
    return an.ajustar_acp(z)


@st.cache_data(show_spinner="Comparando valores de K…")
def correr_evaluacion(matriz, semilla):
    return an.evaluar_k(matriz, 2, 8, semilla)


@st.cache_data(show_spinner="Agrupando las empresas…")
def correr_kmeans(matriz, k, semilla):
    return an.ajustar_kmeans(matriz, k, semilla)


@st.cache_data(show_spinner="Midiendo la estabilidad…")
def correr_estabilidad(matriz, k):
    return an.estabilidad(matriz, k)


prep = correr_preparacion(tuple(seleccion), winsor)
acp = correr_acp(prep["z"])
varianza = acp["varianza"]
sugerencia = an.sugerir_componentes(varianza)
n_max = len(varianza)

if "comp" not in st.session_state or st.session_state["comp"] > n_max:
    st.session_state["comp"] = POR_DEFECTO["componentes"] if list(seleccion) == POR_DEFECTO["variables"] \
        else max(2, min(sugerencia["kaiser"], n_max))
if "k_elegido" not in st.session_state:
    st.session_state["k_elegido"] = POR_DEFECTO["k"]

n_comp = st.session_state["comp"]
k = st.session_state["k_elegido"]
matriz = acp["puntajes"].iloc[:, :n_comp]
km = correr_kmeans(matriz, k, semilla)
grupo = km["grupo"]
letras = sorted(grupo.unique())
color_grupo = {g: COLORES_GRUPO[i] for i, g in enumerate(letras)}
simbolo_grupo = {g: SIMBOLOS_GRUPO[i] for i, g in enumerate(letras)}
ident = prep["identificacion"]
variables = prep["variables"]
z = prep["z"]
datos = prep["datos"]
crudo = prep["crudo"]
apartadas = prep["apartadas"]

es_por_defecto = (
    list(seleccion) == POR_DEFECTO["variables"] and winsor == POR_DEFECTO["winsor"]
    and n_comp == POR_DEFECTO["componentes"] and k == POR_DEFECTO["k"] and semilla == POR_DEFECTO["semilla"]
)


def etiqueta(g, corta=False):
    """«Grupo B» o, en la configuración por defecto, «B · Grandes y apalancadas»."""
    if es_por_defecto and g in NOMBRES_POR_DEFECTO:
        return f"{g} · {NOMBRES_POR_DEFECTO[g]}" if not corta else f"Grupo {g}"
    return f"Grupo {g}"


def descriptores(g, n=2):
    """Indicadores que más separan al grupo del promedio, en palabras."""
    perfil = an.perfiles_z(z, grupo)[g].sort_values()
    altos = [f"{an.nombre(v).lower()} alto" for v in perfil.index[::-1][:n] if perfil[v] > 0.3]
    bajos = [f"{an.nombre(v).lower()} bajo" for v in perfil.index[:n] if perfil[v] < -0.3]
    return (", ".join(altos + bajos) or "cerca del promedio en todos los indicadores").replace("roa", "ROA").replace("roe", "ROE")


def tabla_medianas(base_variables, agrupador, orden=None):
    """Medianas por grupo en unidades legibles, con el número de empresas."""
    med = base_variables.groupby(agrupador).median()
    if orden is not None:
        med = med.reindex([o for o in orden if o in med.index])
    salida = pd.DataFrame(index=med.index)
    salida["Empresas"] = base_variables.groupby(agrupador).size().reindex(med.index)
    for v in med.columns:
        salida[an.nombre(v) if v != "tamano" else "Ingresos (miles de millones)"] = [
            valor(v, x) if v != "tamano" else mm(10 ** x, 1) for x in med[v]]
    return salida


# ==========================================================================
# 1. CONTEXTO
# ==========================================================================
if pagina == PAGINAS[0]:
    st.markdown('<div class="etiqueta">Taller individual · ACP y K-Means</div>', unsafe_allow_html=True)
    st.title("¿Qué tan sanas están las empresas más grandes de Colombia?")
    st.markdown(
        "Tengo una empresa de mariscos y, como cualquier dueña de negocio, me pregunto si mis números "
        "son buenos o malos. La respuesta depende de con quién me compare. Un margen del 3 % es pobre "
        "para una inmobiliaria y normal para una distribuidora de alimentos. Este trabajo usa los estados "
        "financieros de las 10.000 empresas más grandes del país para encontrar, con datos y sin reglas "
        "fijadas de antemano, los **perfiles de salud financiera** que existen en Colombia, y ubicar en "
        "ellos a las empresas de pesca, procesamiento de pescado y comercio de alimentos."
    )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Empresas en la base", num(len(crudo), 0))
    c2.metric("Empresas en el modelo", num(len(datos), 0))
    c3.metric("Indicadores", f"{len(seleccion)}")
    c4.metric("Ingresos totales", f"{num(crudo['INGRESOS_OPERACIONALES'].sum() / 1e9, 1)} billones")

    st.subheader("Preguntas que orientan el análisis")
    st.markdown(
        "1. **¿Cuántas ideas distintas hay detrás de los indicadores financieros?** Margen, ROA y ROE "
        "miden rentabilidad desde ángulos parecidos. ¿Cuentan una sola historia o varias?\n"
        "2. **¿Existen grupos de empresas con una salud financiera parecida?** Y si existen, ¿qué los define?\n"
        "3. **¿El perfil financiero depende del sector?** En particular, ¿dónde caen las empresas de la "
        "cadena del pescado y los alimentos, y dónde quedaría una empresa como la mía?"
    )

    st.subheader("Por qué ACP y K-Means")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown(
            "**Análisis de Componentes Principales.** Los indicadores comparten ingredientes: la utilidad "
            "está en el numerador de tres de ellos y los activos en el denominador de otros tres. Por "
            "construcción están correlacionados. El ACP junta los que se mueven a la par en ejes que no "
            "se repiten entre sí, dice cuánta información conserva cada eje y evita que la rentabilidad "
            "cuente tres veces cuando se midan distancias entre empresas."
        )
    with col_b:
        st.markdown(
            "**K-Means.** Nadie ha clasificado a estas empresas como «sanas» o «en riesgo»: no hay una "
            "etiqueta que predecir, así que el problema es no supervisado. K-Means agrupa a las empresas "
            "que quedan cerca en el espacio de componentes y entrega centroides que se leen como perfiles "
            "típicos. El sector y la región no entran al modelo; se usan al final para contrastar."
        )

    st.subheader("Fuente y cobertura")
    st.markdown(
        f"""
- **Base:** informe *10.000 empresas más grandes del país 2026* de la Superintendencia de Sociedades,
  publicado el 30 de junio de 2026 ([comunicado](https://www.supersociedades.gov.co/noticias-supersociedades/-/asset_publisher/atwl/content/las-10.000-empresas-m%C3%A1s-grandes-de-colombia-registraron-ingresos-por-1.852-9-billones-y-utilidades-por-138-1-billones-al-cierre-de-2025),
  [tablero](https://app.powerbi.com/view?r=eyJrIjoiM2FmYWUxMGQtM2M2ZC00NDhjLWJiMGEtYjJlNTBlZWQ3ZWViIiwidCI6IjZlZTk0YzM0LWJiZDYtNDY0Ny1hNDgzLTBlMTk2YTRkZTBmZiIsImMiOjR9)).
- **Corte:** estados financieros al 31 de diciembre de 2025. Los datos se tomaron del tablero público el
  {an.FECHA_CONSULTA}; la copia está en `data/`.
- **Qué es cada fila:** una empresa, con su NIT, razón social, región, departamento, ciudad, actividad CIIU,
  macrosector y cinco cuentas en miles de pesos: ingresos operacionales, utilidad neta, activos, pasivos y patrimonio.
- **Cobertura:** las 10.000 sociedades con mayores ingresos operacionales que reportan a la Supersociedades
  y a otras superintendencias (Financiera, Salud, Transporte, Servicios Públicos, Vigilancia).
  La más pequeña vendió {mm(crudo['INGRESOS_OPERACIONALES'].min(), 1)} mil millones de pesos.
- **Verificación:** el archivo suma ingresos por {num(crudo['INGRESOS_OPERACIONALES'].sum() / 1e9, 1)} billones,
  utilidades por {num(crudo['GANANCIA_PERDIDA'].sum() / 1e9, 1)} billones y activos por
  {num(crudo['TOTAL_ACTIVOS'].sum() / 1e9, 1)} billones. El comunicado, en la versión consultada el
  {an.FECHA_CONSULTA}, reporta 1.853,8; 138,6 y 2.560,4 billones: la única diferencia está en las utilidades,
  de 0,05 %. Su primera versión, la que recogió la prensa, hablaba de 1.852,9 y 138,1 billones; frente a
  esas cifras el archivo difiere en 0,05 % y 0,4 %.
"""
    )

    st.subheader("Indicadores del modelo")
    tabla_vars = pd.DataFrame([
        {"Indicador": v["nombre"], "Dimensión": v["dimension"], "Cálculo": v["formula"], "Lectura": v["lectura"]}
        for v in an.VARIABLES.values()
    ])
    st.dataframe(tabla_vars, hide_index=True, width="stretch")
    nota(
        "Los seis se calculan con las cinco cuentas de la base. Son los indicadores básicos de un "
        "diagnóstico financiero: tres de rentabilidad, uno de estructura, uno de eficiencia y uno de tamaño."
    )

    st.subheader("Limitaciones de la fuente")
    st.markdown(
        "- **Solo empresas grandes.** La base empieza en ventas de unos 21 mil millones de pesos al año. "
        "Una empresa pequeña o mediana no está representada; compararse sirve como referencia, no como diagnóstico.\n"
        "- **Cinco cuentas, un año.** No hay utilidad operacional, flujo de caja, activos corrientes ni deuda "
        "financiera, así que no se puede medir liquidez ni separar deuda con bancos de deuda con proveedores.\n"
        "- **Estados individuales.** Las casas matrices registran utilidades de sus filiales con pocos ingresos "
        "propios, lo que infla su margen neto.\n"
        "- **Un CIIU por empresa.** La actividad es la principal declarada; una comercializadora de pescado "
        "puede estar registrada como procesadora o como mayorista de alimentos.\n"
        "- **Es una fotografía de 2025.** Un mal año no es lo mismo que una empresa enferma."
    )

# ==========================================================================
# 2. EXPLORACIÓN
# ==========================================================================
elif pagina == PAGINAS[1]:
    st.title("Exploración de la base")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Filas", num(len(crudo), 0))
    c2.metric("Columnas originales", "14")
    c3.metric("Celdas faltantes", f"{int(crudo[list(an.CUENTAS)].isna().sum().sum())}")
    c4.metric("NIT duplicados", f"{int(crudo['NIT'].duplicated().sum())}")

    st.subheader("Estructura del archivo")
    originales = ["RANKING", "NIT", "RAZON_SOCIAL", "SUPERVISOR", "REGION", "DEPARTAMENTO", "CIUDAD", "CIIU",
                  "MACROSECTOR"] + list(an.CUENTAS)
    estructura = pd.DataFrame({
        "Tipo": ["número" if pd.api.types.is_numeric_dtype(crudo[c]) else "texto" for c in originales],
        "Faltantes": [int(crudo[c].isna().sum()) for c in originales],
        "Valores distintos": [int(crudo[c].nunique()) for c in originales],
        "Papel": ["identificador"] * 3 + ["categoría"] * 6 + ["cuenta (miles de pesos)"] * 5,
    }, index=originales)
    st.dataframe(estructura, width="stretch")
    with st.expander("Ver las primeras filas"):
        st.dataframe(crudo[originales].head(20), width="stretch", hide_index=True)

    st.subheader("Estadísticas descriptivas de las cinco cuentas")
    desc = (crudo[list(an.CUENTAS)].describe().T[["min", "25%", "50%", "75%", "max", "mean"]] / 1e6)
    desc.index = [an.CUENTAS[c] for c in desc.index]
    desc.columns = ["Mínimo", "Percentil 25", "Mediana", "Percentil 75", "Máximo", "Promedio"]
    st.dataframe(desc.style.format(lambda v: num(v, 1)), width="stretch")
    st.caption("Cifras en miles de millones de pesos.")
    negativos = int((crudo["TOTAL_PATRIMONIO"] <= 0).sum())
    perdidas = int((crudo["GANANCIA_PERDIDA"] < 0).sum())
    nota(
        f"La base es muy desigual: la empresa mediana vende {mm(crudo['INGRESOS_OPERACIONALES'].median(), 1)} "
        f"mil millones y la más grande {num(crudo['INGRESOS_OPERACIONALES'].max() / 1e9, 1)} billones, casi "
        f"dos mil veces más. Por eso el promedio queda muy por encima de la mediana y el tamaño se mide en "
        f"logaritmos. Además, <b>{num(perdidas, 0)}</b> empresas cerraron el año con pérdidas y "
        f"<b>{negativos}</b> tienen patrimonio negativo: deben más de lo que tienen.",
        "aviso",
    )

    st.subheader("Distribución de los indicadores")
    var_hist = st.selectbox("Indicador", list(variables.columns), format_func=an.nombre)
    col_a, col_b = st.columns(2)
    with col_a:
        fig = px.histogram(variables, x=var_hist, nbins=60, color_discrete_sequence=[MAR])
        fig.update_traces(marker_line_color=SUPERFICIE, marker_line_width=1)
        fig.update_layout(title=f"{an.nombre(var_hist)} · sin recortar",
                          xaxis_title=an.nombre(var_hist), yaxis_title="Empresas")
        st.plotly_chart(estilo(fig, 360, False), width="stretch")
    with col_b:
        fig = px.histogram(prep["recortadas"], x=var_hist, nbins=60, color_discrete_sequence=[MAR_OSCURO])
        fig.update_traces(marker_line_color=SUPERFICIE, marker_line_width=1)
        fig.update_layout(title=f"{an.nombre(var_hist)} · con recorte del {num(winsor, 1)} % por cola",
                          xaxis_title=an.nombre(var_hist), yaxis_title="Empresas")
        st.plotly_chart(estilo(fig, 360, False), width="stretch")
    resumen = variables.describe(percentiles=[0.01, 0.5, 0.99]).T[["min", "1%", "50%", "99%", "max"]]
    resumen.index = [an.nombre(v) for v in resumen.index]
    resumen.columns = ["Mínimo", "Percentil 1", "Mediana", "Percentil 99", "Máximo"]
    st.dataframe(resumen.style.format(lambda v: num(v, 3)), width="stretch")
    nota(
        "Sin recortar, unas pocas empresas dominan el gráfico: hay ROE de más de 200 veces el patrimonio "
        "y rotaciones de más de 80 veces. Son empresas con un patrimonio o unos activos diminutos frente a "
        "su operación. La gran mayoría de los casos vive en un rango mucho más estrecho, que es el que se ve a la derecha."
    )

    st.subheader("Correlación entre indicadores")
    corr = prep["recortadas"].corr()
    corr.index = [an.nombre(v) for v in corr.index]
    corr.columns = [an.nombre(v) for v in corr.columns]
    st.plotly_chart(mapa_calor(corr, "Correlación de Pearson (indicadores recortados)", DIVERGENTE, (-1, 1), alto=480),
                    width="stretch")
    pares = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool)).stack().dropna()
    fuertes = pares.reindex(pares.abs().sort_values(ascending=False).index).head(4)
    nota(
        "Pares con mayor correlación absoluta: " + "; ".join(
            f"{a} – {b} ({num(v, 2)})" for (a, b), v in fuertes.items()) + "." + (
            " Los tres indicadores de rentabilidad se mueven juntos y el endeudamiento va en contra del margen "
            "y del ROA. El tamaño casi no se relaciona con nada: vender más no hace a una empresa más rentable "
            "ni menos endeudada." if list(seleccion) == POR_DEFECTO["variables"] else "")
    )

    st.subheader("Empresas por macrosector y por región")
    col_a, col_b = st.columns(2)
    for columna, campo, titulo in [(col_a, "MACROSECTOR", "Empresas por macrosector"),
                                   (col_b, "REGION", "Empresas por región")]:
        conteo = crudo[campo].value_counts().sort_values()
        fig = go.Figure(go.Bar(x=conteo.values, y=conteo.index, orientation="h", marker_color=MAR,
                               hovertemplate="%{y}: %{x} empresas<extra></extra>"))
        fig.update_layout(title=titulo, xaxis_title="Empresas", yaxis_title="")
        columna.plotly_chart(estilo(fig, 380, False), width="stretch")

# ==========================================================================
# 3. PREPARACIÓN
# ==========================================================================
elif pagina == PAGINAS[2]:
    st.title("Preparación de los datos")
    st.markdown("Cada revisión se hace en código (`analisis.py`) y queda registrada en esta bitácora.")

    st.subheader("Bitácora de calidad")
    st.dataframe(prep["bitacora"], hide_index=True, width="stretch")
    c1, c2, c3 = st.columns(3)
    c1.metric("Empresas en la base", num(len(crudo), 0))
    c2.metric("Apartadas (patrimonio ≤ 0)", num(len(apartadas), 0))
    c3.metric("Empresas en el modelo", num(len(datos), 0))
    nota(
        f"La base llega limpia: sin vacíos, sin duplicados y con la ecuación contable cuadrada. La única "
        f"decisión de fondo es qué hacer con las {len(apartadas)} empresas de patrimonio negativo. Se apartan "
        f"porque su ROE engaña, pero no se olvidan: son el caso más grave de deterioro financiero y se "
        f"describen en la sección 6 como un grupo definido por regla."
    )

    st.subheader("Construcción de los indicadores")
    st.markdown(
        f"Los indicadores se calculan con las cinco cuentas. Quedan "
        f"**{int(variables.isna().sum().sum())} valores faltantes** e "
        f"**{int(np.isinf(variables.values).sum())} infinitos**: no hace falta imputar."
    )
    with st.expander("Ver la tabla de indicadores por empresa"):
        st.dataframe(pd.concat([ident[["Puesto", "Empresa", "Macrosector"]], variables], axis=1),
                     width="stretch", hide_index=True)

    st.subheader("Tratamiento de valores extremos")
    if winsor > 0:
        limites = prep["limites"].copy()
        limites.index = [an.nombre(v) for v in limites.index]
        st.dataframe(limites.style.format({"Límite inferior": lambda v: num(v, 3),
                                           "Límite superior": lambda v: num(v, 3)}), width="stretch")
        nota(
            f"Con un recorte del {num(winsor, 1)} % por cola, los valores más extremos de cada indicador se "
            "reemplazan por el percentil límite; la empresa no sale de la base. K-Means trabaja con medias y "
            "distancias: un solo ROE de 255 movería un centroide entero y la estandarización aplastaría al "
            "resto de empresas contra el cero. Cambia el nivel en la barra lateral para ver el efecto."
        )
    else:
        nota("Sin recorte: los valores extremos entran tal cual. Revisa cómo cambian la varianza explicada "
             "y los grupos: suelen aparecer grupos de muy pocas empresas atípicas.", "aviso")

    st.subheader("Estandarización")
    st.markdown(
        "Cada indicador se transforma a $z = (x - \\bar{x}) / s$, con media 0 y desviación 1. Sin este paso "
        "la rotación (que llega a casi 20 veces) pesaría mucho más en las distancias que el margen "
        "(que rara vez pasa de 0,5), solo por la unidad en que se mide."
    )
    largo = z.melt(var_name="Indicador", value_name="z")
    largo["Indicador"] = largo["Indicador"].map(an.nombre)
    fig = px.box(largo, x="z", y="Indicador", color_discrete_sequence=[MAR], points=False)
    fig.update_layout(title="Indicadores estandarizados: todos en la misma escala", xaxis_title="Valor z",
                      yaxis_title="")
    st.plotly_chart(estilo(fig, 380, False), width="stretch")
    resumen_z = z.agg(["mean", "std"]).T
    resumen_z.index = [an.nombre(v) for v in resumen_z.index]
    resumen_z.columns = ["Media", "Desviación estándar"]
    st.dataframe(resumen_z.style.format(lambda v: num(v + 0.0, 2)), width="stretch")

# ==========================================================================
# 4. ACP
# ==========================================================================
elif pagina == PAGINAS[3]:
    st.title("Análisis de Componentes Principales")

    st.subheader("Varianza explicada")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure()
        fig.add_bar(x=varianza["Componente"], y=varianza["Varianza explicada"] * 100, name="Individual",
                    marker_color=MAR, hovertemplate="%{x}: %{y:.1f} %<extra>Individual</extra>")
        fig.add_scatter(x=varianza["Componente"], y=varianza["Varianza acumulada"] * 100, name="Acumulada",
                        mode="lines+markers", line=dict(color=MAR_OSCURO, width=2), marker=dict(size=8),
                        hovertemplate="%{x}: %{y:.1f} %<extra>Acumulada</extra>")
        fig.add_hline(y=80, line_dash="dot", line_color=TINTA_3,
                      annotation_text="80 %", annotation_position="top left")
        fig.update_layout(title="Varianza explicada por componente (%)", yaxis_title="% de la varianza total",
                          xaxis_title="", legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
        st.plotly_chart(estilo(fig, 400), width="stretch")
    with col_b:
        fig = go.Figure(go.Scatter(x=varianza["Componente"], y=varianza["Valor propio"], mode="lines+markers",
                                   line=dict(color=MAR, width=2), marker=dict(size=9),
                                   hovertemplate="%{x}: valor propio %{y:.2f}<extra></extra>"))
        fig.add_hline(y=1, line_dash="dot", line_color=CORAL, annotation_text="Criterio de Kaiser (λ = 1)",
                      annotation_position="top right")
        fig.update_layout(title="Gráfico de sedimentación", yaxis_title="Valor propio (λ)", xaxis_title="")
        st.plotly_chart(estilo(fig, 400, False), width="stretch")

    st.dataframe(varianza.style.format({"Valor propio": lambda v: num(v, 3),
                                        "Varianza explicada": lambda v: pct(v, 1),
                                        "Varianza acumulada": lambda v: pct(v, 1)}),
                 hide_index=True, width="stretch")

    st.subheader("¿Cuántos componentes conservar?")
    c1, c2 = st.columns(2)
    c1.metric("Criterio de Kaiser (λ > 1)", f"{sugerencia['kaiser']} componentes")
    c1.caption(f"Conservan el {pct(varianza['Varianza acumulada'].iloc[sugerencia['kaiser'] - 1])} de la varianza")
    c2.metric("Criterio del 80 % acumulado", f"{sugerencia['acumulada']} componentes")
    c2.caption(f"Conservan el {pct(varianza['Varianza acumulada'].iloc[sugerencia['acumulada'] - 1])} de la varianza")
    st.slider("Componentes que pasan a K-Means", 2, n_max, key="comp",
              help="La segmentación se calcula sobre los puntajes de estos componentes.")
    if es_por_defecto:
        nota(
            f"Tres componentes tienen valor propio mayor que 1 y juntos conservan el "
            f"{pct(varianza['Varianza acumulada'].iloc[2])} de la variación de los seis indicadores. Llegar al "
            f"80 % pide un cuarto, pero su valor propio ({num(varianza['Valor propio'].iloc[3], 2)}) dice que "
            f"resume menos que un indicador original, y en el gráfico de sedimentación la curva ya viene "
            f"aplanándose. Se conservan <b>tres</b>: cada uno tiene una lectura financiera propia y, en la "
            f"sección 5, dan grupos mejor separados que cuatro (silueta de 0,31 frente a 0,28). El precio es "
            f"dejar fuera cerca de una cuarta parte de la información."
        )
    else:
        nota(f"Ahora usas {n_comp} componentes, que resumen el "
             f"{pct(varianza['Varianza acumulada'].iloc[n_comp - 1])} de la varianza. Compara con los dos criterios.")

    st.subheader("Cargas: qué indicador define cada componente")
    st.markdown(
        "Cada celda es la **correlación** entre un indicador y un componente (carga × √λ). "
        "Valores cerca de ±1 indican que ese indicador define el eje."
    )
    corr_cp = acp["correlaciones"].iloc[:, :max(n_comp, 3)].copy()
    corr_cp.index = [an.nombre(v) for v in corr_cp.index]
    st.plotly_chart(mapa_calor(corr_cp, "Correlación entre indicadores y componentes", DIVERGENTE, (-1, 1), alto=420),
                    width="stretch")

    for i in range(n_comp):
        cp = f"CP{i + 1}"
        serie = acp["correlaciones"][cp].sort_values()
        pos = [f"{an.nombre(v)} ({num(serie[v], 2)})" for v in serie.index[::-1][:3] if serie[v] > 0.3]
        neg = [f"{an.nombre(v)} ({num(serie[v], 2)})" for v in serie.index[:3] if serie[v] < -0.3]
        st.markdown(
            f"**{cp}** · {pct(varianza['Varianza explicada'].iloc[i])} de la varianza. "
            f"Polo positivo: {', '.join(pos) or '—'}. Polo negativo: {', '.join(neg) or '—'}."
        )
    if es_por_defecto:
        nota(
            "<b>Lectura.</b> <b>CP1 · rentabilidad.</b> ROA, ROE y margen neto suben juntos y el "
            "endeudamiento baja: es el eje que separa a las empresas que ganan plata y deben poco de las que "
            "pierden y deben mucho. <b>CP2 · modelo de volumen con deuda.</b> Suben la rotación de activos, "
            "el endeudamiento y el ROE, y baja el margen: describe a los negocios que ganan poco por cada "
            "peso vendido pero venden muchas veces sus activos y se financian con terceros, frente a los que "
            "viven de un margen alto. <b>CP3 · tamaño.</b> Lo define casi solo el volumen de ventas, con la "
            "rotación en contra. Que el tamaño forme su propio eje confirma que ser grande y ser rentable "
            "son cosas distintas."
        )

    st.subheader("Las empresas en el espacio de componentes")
    col_x, col_y, col_s = st.columns(3)
    opciones_cp = list(varianza["Componente"])
    eje_x = col_x.selectbox("Eje horizontal", opciones_cp, 0)
    eje_y = col_y.selectbox("Eje vertical", opciones_cp, 1)
    sector_dest = col_s.selectbox("Destacar un macrosector", ["Ninguno"] + sorted(ident["Macrosector"].unique()))
    puntos = pd.concat([ident, acp["puntajes"]], axis=1)
    resaltado = puntos["Macrosector"].eq(sector_dest)
    fig = go.Figure()
    fig.add_scattergl(x=puntos.loc[~resaltado, eje_x], y=puntos.loc[~resaltado, eje_y], mode="markers",
                      marker=dict(size=5, color="#8fc9cf" if sector_dest == "Ninguno" else "#d3dadb", opacity=0.6),
                      name="Resto" if sector_dest != "Ninguno" else "Empresas",
                      customdata=puntos.loc[~resaltado, ["Empresa", "Macrosector"]],
                      hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>")
    if sector_dest != "Ninguno":
        fig.add_scattergl(x=puntos.loc[resaltado, eje_x], y=puntos.loc[resaltado, eje_y], mode="markers",
                          marker=dict(size=6, color=MAR_OSCURO, opacity=0.85),
                          name=sector_dest, customdata=puntos.loc[resaltado, ["Empresa", "Macrosector"]],
                          hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}<extra></extra>")
    # Vectores de carga (biplot), escalados para que se vean sobre la nube.
    escala = 0.8 * min(puntos[eje_x].abs().max(), puntos[eje_y].abs().max())
    for v in acp["correlaciones"].index:
        dx, dy = acp["correlaciones"].loc[v, eje_x] * escala, acp["correlaciones"].loc[v, eje_y] * escala
        fig.add_annotation(x=dx, y=dy, ax=0, ay=0, xref="x", yref="y", axref="x", ayref="y", showarrow=True,
                           arrowhead=2, arrowwidth=1.5, arrowcolor=TINTA, text="")
        fig.add_annotation(x=dx * 1.12, y=dy * 1.12, text=an.nombre(v), showarrow=False,
                           font=dict(size=11, color=TINTA), bgcolor="rgba(251,252,252,0.85)")
    i_x, i_y = opciones_cp.index(eje_x), opciones_cp.index(eje_y)
    resumida = varianza["Varianza explicada"].iloc[i_x] + (varianza["Varianza explicada"].iloc[i_y] if i_y != i_x else 0)
    fig.update_layout(title=f"{eje_x} frente a {eje_y} · este plano resume el {pct(resumida)} de la varianza",
                      xaxis_title=f"{eje_x} ({pct(varianza['Varianza explicada'].iloc[i_x])})",
                      yaxis_title=f"{eje_y} ({pct(varianza['Varianza explicada'].iloc[i_y])})",
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    st.plotly_chart(estilo(fig, 600), width="stretch")
    nota(
        "Cada punto es una empresa. Las flechas muestran la correlación de cada indicador con los dos ejes "
        "(biplot): una empresa ubicada en la dirección de una flecha tiene un valor alto en ese indicador. "
        "Este plano no muestra toda la información: lo que queda en los demás componentes no se ve aquí."
    )

# ==========================================================================
# 5. K-MEANS
# ==========================================================================
elif pagina == PAGINAS[4]:
    st.title("Segmentación con K-Means")
    evaluacion = correr_evaluacion(matriz, semilla)
    st.markdown(
        f"K-Means corre sobre los puntajes de los **{n_comp} primeros componentes** "
        f"({pct(varianza['Varianza acumulada'].iloc[n_comp - 1])} de la varianza) de {num(len(matriz), 0)} "
        f"empresas, con 10 inicios aleatorios y semilla {semilla}. Los componentes no están correlacionados "
        f"entre sí, así que ninguna idea cuenta dos veces en las distancias."
    )

    st.subheader("Comparación de valores de K")
    col_a, col_b = st.columns(2)
    with col_a:
        fig = go.Figure(go.Scatter(x=evaluacion["K"], y=evaluacion["Inercia"], mode="lines+markers",
                                   line=dict(color=MAR, width=2), marker=dict(size=9),
                                   hovertemplate="K = %{x}: inercia %{y:,.0f}<extra></extra>"))
        fig.update_layout(title="Método del codo", xaxis_title="Número de grupos (K)",
                          yaxis_title="Inercia (suma de distancias²)")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(estilo(fig, 380, False), width="stretch")
    with col_b:
        mejor = evaluacion.loc[evaluacion["Silueta"].idxmax()]
        fig = go.Figure(go.Scatter(x=evaluacion["K"], y=evaluacion["Silueta"], mode="lines+markers",
                                   line=dict(color=MAR_OSCURO, width=2), marker=dict(size=9),
                                   hovertemplate="K = %{x}: silueta %{y:.3f}<extra></extra>"))
        fig.add_annotation(x=mejor["K"], y=mejor["Silueta"], text=f"máximo: K = {int(mejor['K'])}",
                           showarrow=True, arrowhead=2, ay=-35, font=dict(color=TINTA))
        fig.update_layout(title="Coeficiente de silueta promedio", xaxis_title="Número de grupos (K)",
                          yaxis_title="Silueta promedio")
        fig.update_xaxes(dtick=1)
        st.plotly_chart(estilo(fig, 380, False), width="stretch")

    tabla_k = evaluacion.copy()
    tabla_k["Caída de inercia"] = -tabla_k["Inercia"].pct_change()
    st.dataframe(tabla_k.style.format({"Inercia": lambda v: num(v, 0), "Silueta": lambda v: num(v, 3),
                                       "Grupo más pequeño": lambda v: num(v, 0),
                                       "Caída de inercia": lambda v: pct(v, 1)}, na_rep="—"),
                 hide_index=True, width="stretch")

    st.slider("Número de grupos (K)", 2, 8, key="k_elegido")
    estab_media, estab_min = correr_estabilidad(matriz, k)
    negativas = int((km["silueta"] < 0).sum())
    c1, c2, c3 = st.columns(3)
    c1.metric("Silueta promedio", num(evaluacion.set_index("K").loc[k, "Silueta"], 3))
    c2.metric("Estabilidad (ARI medio, 10 semillas)", num(estab_media, 2))
    c3.metric("Empresas con silueta negativa", f"{num(negativas, 0)} ({pct(negativas / len(grupo))})")
    if es_por_defecto:
        nota(
            "<b>Elección de K = 5.</b> La silueta tiene su máximo en K = 5 (0,307), por encima de K = 2 "
            "(0,291), que solo separa rentables de no rentables. En el codo, la inercia cae alrededor del "
            "17 % con cada grupo hasta K = 5 y de ahí en adelante cae menos (13 % al pasar a 6). La solución "
            "también es estable: al repetir K-Means con otras diez semillas, el índice de Rand ajustado "
            "promedio es 0,89, mientras que con K = 3, 4 o 6 queda entre 0,50 y 0,69, es decir, los grupos "
            "cambian según dónde arranque el algoritmo. Una silueta de 0,31 indica una estructura "
            "<b>débil a moderada</b>: los grupos existen, pero sus fronteras se tocan."
        )
    else:
        nota("La silueta mide qué tan cerca está cada empresa de su grupo frente al grupo vecino (de −1 a 1). "
             "El ARI compara la asignación con la de otras semillas: 1 = idéntica.")

    st.subheader("Los grupos en el espacio de componentes")
    col_x, col_y = st.columns(2)
    opciones = list(matriz.columns)
    eje_x = col_x.selectbox("Eje horizontal", opciones, 0, key="km_x")
    eje_y = col_y.selectbox("Eje vertical", opciones, 1, key="km_y")
    puntos = pd.concat([ident, matriz, grupo], axis=1)
    fig = go.Figure()
    for g in letras:
        sub = puntos[puntos["Grupo"] == g]
        fig.add_scattergl(x=sub[eje_x], y=sub[eje_y], mode="markers", name=f"{etiqueta(g)} ({num(len(sub), 0)})",
                          marker=dict(size=5, color=color_grupo[g], symbol=simbolo_grupo[g], opacity=0.6),
                          customdata=sub[["Empresa", "Macrosector"]],
                          hovertemplate="<b>%{customdata[0]}</b><br>%{customdata[1]}"
                                        f"<extra>Grupo {g}</extra>")
    centros = km["centroides"]
    fig.add_scatter(x=centros[eje_x], y=centros[eje_y], mode="markers+text", name="Centroides",
                    marker=dict(size=16, color=TINTA, symbol="x-thin", line=dict(width=3, color=TINTA)),
                    text=[f"  {g}" for g in centros.index], textposition="middle right",
                    textfont=dict(size=15, color=TINTA), hovertemplate="Centroide %{text}<extra></extra>")
    i_x, i_y = opciones.index(eje_x), opciones.index(eje_y)
    resumida = varianza["Varianza explicada"].iloc[i_x] + (varianza["Varianza explicada"].iloc[i_y] if i_y != i_x else 0)
    fig.update_layout(title=f"Grupos sobre {eje_x} y {eje_y} · el plano resume el {pct(resumida)} de la varianza",
                      xaxis_title=eje_x, yaxis_title=eje_y)
    st.plotly_chart(estilo(fig, 600), width="stretch")
    nota("Las letras solo nombran a los grupos (A es el más numeroso); no indican orden ni calidad. "
         "Los grupos se forman con todos los componentes seleccionados, así que en un plano de dos ejes "
         "pueden verse superpuestos aunque estén separados en una tercera dimensión. Prueba otros ejes.")

    st.subheader("Silueta de cada empresa")
    fig = go.Figure()
    inicio = 0
    for g in letras:
        valores = km["silueta"][grupo == g].sort_values(ascending=False).values
        posiciones = np.arange(inicio, inicio + len(valores))
        fig.add_scatter(x=valores, y=posiciones, mode="lines", fill="tozerox", name=f"Grupo {g}",
                        line=dict(color=color_grupo[g], width=1),
                        hovertemplate="silueta %{x:.2f}" + f"<extra>Grupo {g}</extra>")
        inicio += len(valores) + 150
    fig.add_vline(x=km["silueta"].mean(), line_dash="dot", line_color=TINTA_2,
                  annotation_text=f"promedio {num(km['silueta'].mean(), 2)}")
    fig.update_layout(title="Coeficiente de silueta por empresa, ordenado dentro de cada grupo",
                      xaxis_title="Silueta", yaxis=dict(showticklabels=False, title="", autorange="reversed"),
                      legend=dict(orientation="h", y=1.02, x=1, xanchor="right", yanchor="bottom"))
    st.plotly_chart(estilo(fig, 520), width="stretch")
    resumen_sil = pd.DataFrame({
        "Empresas": grupo.value_counts().sort_index(),
        "Silueta media": km["silueta"].groupby(grupo).mean(),
        "Silueta negativa": (km["silueta"] < 0).groupby(grupo).sum(),
    })
    resumen_sil["% con silueta negativa"] = resumen_sil["Silueta negativa"] / resumen_sil["Empresas"]
    resumen_sil.index = [etiqueta(g) for g in resumen_sil.index]
    st.dataframe(resumen_sil.style.format({"Empresas": lambda v: num(v, 0), "Silueta media": lambda v: num(v, 3),
                                           "Silueta negativa": lambda v: num(v, 0),
                                           "% con silueta negativa": lambda v: pct(v, 1)}), width="stretch")
    if es_por_defecto:
        nota(
            "El grupo A, el más numeroso, es también el más compacto y no tiene ninguna empresa con silueta "
            "negativa. Los menos definidos son el C y el D: cerca de una de cada cuatro empresas del C y una "
            "de cada tres del D quedan más cerca de otro grupo que del suyo. Tiene sentido: «rentable» y "
            "«de alta rotación» son cuestión de grado, y muchas empresas están a medio camino entre esos "
            "perfiles y el grupo central."
        )
    dudosas = pd.concat([ident[["Empresa", "Macrosector"]], grupo, km["silueta"]], axis=1)
    dudosas = dudosas[dudosas["Silueta"] < 0].sort_values("Silueta").head(200)
    with st.expander("Empresas más difíciles de asignar (las 200 de menor silueta)"):
        st.dataframe(dudosas.style.format({"Silueta": lambda v: num(v, 3)}), hide_index=True, width="stretch")
        st.caption("Una silueta negativa indica que la empresa está, en promedio, más cerca de otro grupo "
                   "que del suyo: es un caso de frontera.")

# ==========================================================================
# 6. PERFILES
# ==========================================================================
elif pagina == PAGINAS[5]:
    st.title("Perfil de los grupos")
    st.markdown("Los grupos se describen con los **indicadores originales**, no con los componentes.")

    st.subheader("¿En qué se aleja cada grupo del promedio?")
    pz = an.perfiles_z(z, grupo)
    pz.index = [an.nombre(v) for v in pz.index]
    pz.columns = [f"Grupo {g}" for g in pz.columns]
    st.plotly_chart(mapa_calor(pz, "Media estandarizada de cada indicador por grupo (0 = promedio de la base)",
                               DIVERGENTE, (-2, 2), alto=420), width="stretch")
    nota("Azul: el grupo está por debajo del promedio en ese indicador; rojo: por encima. Un valor de 1 "
         "equivale a una desviación estándar.")

    st.subheader("Valores típicos (mediana de cada grupo)")
    legible = tabla_medianas(variables, grupo)
    total = tabla_medianas(variables, pd.Series("Todo el modelo", index=variables.index))
    legible.index = [etiqueta(g) for g in legible.index]
    legible = pd.concat([legible, total])
    legible.insert(1, "% con pérdidas", list((datos["GANANCIA_PERDIDA"] < 0).groupby(grupo).mean().map(pct))
                   + [pct((datos["GANANCIA_PERDIDA"] < 0).mean())])
    legible.insert(1, "% de los ingresos", list((datos["INGRESOS_OPERACIONALES"].groupby(grupo).sum()
                                                 / datos["INGRESOS_OPERACIONALES"].sum()).map(pct)) + ["100,0 %"])
    st.dataframe(legible, width="stretch")

    st.subheader("Descripción automática")
    for g in letras:
        n_g = int((grupo == g).sum())
        st.markdown(f"- **Grupo {g}** ({num(n_g, 0)} empresas, {pct(n_g / len(grupo))}): {descriptores(g)}.")

    if es_por_defecto:
        st.subheader("Interpretación")
        st.markdown(
            "- **A · Solventes de rentabilidad moderada (5.085 empresas, 52 %).** La empresa colombiana "
            "típica: financia con deuda menos de la mitad de sus activos (47 %), deja un margen neto de "
            "3,6 % y rinde 10 % sobre el patrimonio. Son las más pequeñas de la base (mediana de 38 mil "
            "millones en ventas). Aquí abundan constructoras, distribuidoras de materiales, transportadoras "
            "y empresas de vigilancia.\n"
            "- **B · Grandes y apalancadas (2.358, 24 %).** Venden casi cuatro veces la mediana de la base (196 "
            "mil millones) y deben dos de cada tres pesos de sus activos (67 %). Su margen es bajo (2,5 %), pero "
            "la deuda eleva el ROE a 13 %. Son una cuarta parte de las empresas y concentran el 69 % de los "
            "ingresos: Ecopetrol, Terpel, D1, EPM y Éxito están aquí.\n"
            "- **C · Rentables y poco endeudadas (901, 9 %).** Margen neto de 23 %, ROA de 21 % y deuda de "
            "solo 27 % de los activos. Venden menos de una vez sus activos al año: ganan por margen, no por "
            "volumen. Reúne oleoductos, cerveceras, cementeras, inmobiliarias y casas matrices (Cenit, "
            "Bavaria, Grupo Argos). Ninguna cerró con pérdidas.\n"
            "- **D · Bajo margen y alto volumen (831, 9 %).** Venden seis veces sus activos al año con un "
            "margen de 1,8 %. Tienen pocos activos (mediana de 8,5 mil millones) y deuda alta (67 %); la base "
            "no permite saber cuánta es con proveedores y cuánta con bancos. El resultado es un ROE de 31 %, tan alto "
            "como el del grupo C pero conseguido de otra manera. Son estaciones de servicio, mayoristas de "
            "alimentos, supermercados y empresas de servicios temporales.\n"
            "- **E · En pérdidas (571, 6 %).** Todas perdieron plata en 2025: margen de −5,9 %, ROE de −28 % "
            "y el endeudamiento más alto (76 %). Hay nombres grandes (Jerónimo Martins, Colombia "
            "Telecomunicaciones, MercadoLibre) junto a floricultoras y comercializadoras."
        )
        nota(
            "Los grupos C y D muestran por qué mirar solo el ROE engaña. Los dos rinden cerca de 31 % sobre "
            "el patrimonio, pero uno lo logra con margen y poca deuda, y el otro con rotación y mucha deuda. "
            "El riesgo de cada camino es distinto: a D, una caída pequeña del margen lo deja en pérdidas."
        )

    st.subheader("Fuera del modelo: empresas con patrimonio negativo")
    en_perdida = (apartadas["GANANCIA_PERDIDA"] < 0).mean()
    deuda = (apartadas["TOTAL_PASIVOS"] / apartadas["TOTAL_ACTIVOS"]).median()
    c1, c2, c3 = st.columns(3)
    c1.metric("Empresas", num(len(apartadas), 0))
    c2.metric("Cerraron con pérdidas", pct(en_perdida))
    c3.metric("Pasivos / activos (mediana)", pct(deuda))
    st.markdown(
        "No entraron a K-Means porque su ROE no se puede leer, pero son el caso extremo: sus deudas superan "
        "sus activos. Se agrupan por regla, no por algoritmo."
    )
    with st.expander("Ver estas empresas"):
        tabla_ap = apartadas[["RANKING", "RAZON_SOCIAL", "MACROSECTOR", "ACTIVIDAD"]].copy()
        tabla_ap.columns = ["Puesto", "Empresa", "Macrosector", "Actividad"]
        for cuenta, titulo in [("INGRESOS_OPERACIONALES", "Ingresos"), ("GANANCIA_PERDIDA", "Utilidad"),
                               ("TOTAL_PATRIMONIO", "Patrimonio")]:
            tabla_ap[f"{titulo} (miles de millones)"] = (apartadas[cuenta] / 1e6).round(1)
        st.dataframe(tabla_ap, hide_index=True, width="stretch", height=320)

    st.subheader("¿Los grupos repiten los sectores o las regiones?")
    criterio = st.radio("Comparar con", ["Macrosector", "Región"], horizontal=True)
    comp = an.composicion(ident[criterio], grupo)
    comp.columns = [f"Grupo {g}" for g in comp.columns]
    st.plotly_chart(mapa_calor(comp, f"Porcentaje de las empresas de cada {criterio.lower()} que cae en cada grupo",
                               SECUENCIAL, (0, 1), porcentaje=True, alto=420), width="stretch")
    if es_por_defecto:
        nota(
            "Ningún macrosector ni región tiene un grupo propio: en todos, el grupo A es el más numeroso "
            "(entre 38 % y 72 % de las empresas). Las diferencias están en los demás grupos. El comercio "
            "tiene más empresas de alta rotación (14 % frente a 9 % del total), la minería y los "
            "hidrocarburos se van a los extremos (20 % rentables, 18 % en pérdidas) y los servicios aportan "
            "muchas rentables (16 %). Por región las diferencias son menores, y las más marcadas están en "
            "regiones con pocas empresas: la salud financiera depende más del modelo de negocio que del lugar."
        )

    st.subheader("Empresas de cada grupo")
    g_sel = st.selectbox("Grupo", letras, format_func=etiqueta)
    tabla = pd.concat([ident[["Puesto", "Empresa", "Macrosector", "Actividad", "Departamento"]],
                       grupo, km["silueta"]], axis=1)
    tabla["Ingresos (miles de millones)"] = datos["INGRESOS_OPERACIONALES"] / 1e6
    tabla = tabla[tabla["Grupo"] == g_sel].sort_values("Puesto")
    st.dataframe(tabla.drop(columns="Grupo").style.format({"Silueta": lambda v: num(v, 2),
                                                           "Ingresos (miles de millones)": lambda v: num(v, 1)}),
                 hide_index=True, width="stretch", height=360)

    st.subheader("Buscar una empresa")
    opciones = ident["Empresa"].tolist()
    por_defecto = next((i for i, e in enumerate(opciones) if "GRAN LANGOSTINO" in e), 0)
    elegida = st.selectbox("Empresa", range(len(opciones)), index=por_defecto, format_func=lambda i: opciones[i])
    c1, c2, c3 = st.columns(3)
    c1.metric("Grupo", etiqueta(grupo.iloc[elegida]))
    c2.metric("Actividad", ident["CIIU"].iloc[elegida], help=ident["Actividad"].iloc[elegida])
    c3.metric("Silueta", num(km["silueta"].iloc[elegida], 2))
    comparacion = pd.DataFrame({
        "Empresa": [valor(v, variables[v].iloc[elegida]) for v in variables.columns],
        f"Mediana del grupo {grupo.iloc[elegida]}": [valor(v, variables.loc[grupo == grupo.iloc[elegida], v].median())
                                                     for v in variables.columns],
        "Mediana de la base": [valor(v, variables[v].median()) for v in variables.columns],
    }, index=[an.nombre(v) if v != "tamano" else "Ingresos" for v in variables.columns])
    st.dataframe(comparacion, width="stretch")

# ==========================================================================
# 7. PESCA Y ALIMENTOS
# ==========================================================================
elif pagina == PAGINAS[6]:
    st.title("Pesca y alimentos: ¿dónde quedaría una empresa de mariscos?")
    st.markdown(
        "Esta sección aterriza los resultados en la cadena que le interesa a una empresa de mariscos: "
        "quien pesca o cultiva, quien procesa, quien distribuye y quien vende al consumidor."
    )
    orden = list(an.CADENA_ALIMENTOS)
    en_cadena = ident["Eslabón"].notna()

    st.subheader("Actividades de la cadena")
    codigos = pd.DataFrame({
        "Eslabón": orden,
        "Códigos CIIU": [", ".join(an.CADENA_ALIMENTOS[e]) for e in orden],
        "Empresas en el modelo": [int((ident["Eslabón"] == e).sum()) for e in orden],
    })
    st.dataframe(codigos, hide_index=True, width="stretch")
    pesca = ident["Eslabón"].isin(orden[:2])
    nota(
        f"Solo {int(pesca.sum())} empresas de pesca, acuicultura y procesamiento de pescado están entre las "
        f"10.000 más grandes. Es un sector de empresas pequeñas, así que las conclusiones sobre él se apoyan "
        f"en muy pocos casos. El comercio minorista de carnes y pescados comparte código CIIU (G4723): "
        f"la mayoría de esas empresas vende pollo, cerdo o res.",
        "aviso",
    )

    st.subheader("Indicadores típicos de cada eslabón")
    medianas = tabla_medianas(variables[en_cadena], ident.loc[en_cadena, "Eslabón"], orden)
    total = tabla_medianas(variables, pd.Series("Todas las empresas del modelo", index=variables.index))
    st.dataframe(pd.concat([medianas, total]), width="stretch")

    st.subheader("¿En qué grupo cae cada eslabón?")
    comp = an.composicion(ident.loc[en_cadena, "Eslabón"], grupo[en_cadena]).reindex(orden)
    comp.loc["Todas las empresas del modelo"] = grupo.value_counts(normalize=True).sort_index()
    comp = comp.fillna(0)
    comp.columns = [f"Grupo {g}" for g in comp.columns]
    st.plotly_chart(mapa_calor(comp, "Porcentaje de las empresas de cada eslabón que cae en cada grupo",
                               SECUENCIAL, (0, 1), porcentaje=True, alto=420), width="stretch")
    if es_por_defecto:
        nota(
            "<b>Lectura.</b> La cadena cambia de perfil según el eslabón. <b>Quien pesca y procesa</b> se "
            "reparte entre el grupo A (10 de 20 empresas) y el B (8 de 20), y las otras dos cerraron en "
            "pérdidas. Su margen mediano es de 1,6 %, rotan cerca de dos veces sus activos y deben el 62 %. "
            "Ninguna está entre las rentables ni entre las de alta rotación; una explicación razonable es "
            "que la planta, la flota, los cuartos fríos y el inventario congelado amarran activos. "
            "<b>Quien distribuye y vende</b> sí llega al grupo D: 29 % de los mayoristas de alimentos, 31 % "
            "de los supermercados y 42 % de los minoristas de carnes y pescados, frente a 9 % en toda la "
            "base. Salvo los restaurantes (11 %), casi nadie en la cadena está en el grupo C: en alimentos "
            "se gana rotando, no por margen."
        )

    st.subheader("Empresas de pesca, acuicultura y procesamiento de pescado")
    tabla = pd.concat([ident.loc[pesca, ["Puesto", "Empresa", "Departamento", "Eslabón"]], grupo[pesca]], axis=1)
    tabla["Ingresos (miles de millones)"] = (datos.loc[pesca, "INGRESOS_OPERACIONALES"] / 1e6).round(1)
    for v in variables.columns:
        if v != "tamano":
            tabla[an.nombre(v)] = [valor(v, x) for x in variables.loc[pesca, v]]
    st.dataframe(tabla.sort_values("Puesto"), hide_index=True, width="stretch")

    st.subheader("Ubica tu empresa")
    st.markdown(
        "Escribe las cuatro cifras de tu último cierre, en **millones de pesos**. La aplicación calcula los "
        "indicadores, les aplica el mismo recorte, la misma estandarización y los mismos componentes del "
        "modelo, y asigna el grupo cuyo centroide queda más cerca. Las cifras no se guardan. "
        "Los valores iniciales son un ejemplo inventado."
    )
    c1, c2, c3, c4 = st.columns(4)
    ingresos = c1.number_input("Ingresos operacionales", min_value=1.0, value=6000.0, step=100.0)
    utilidad = c2.number_input("Utilidad neta (negativa si hubo pérdida)", value=150.0, step=10.0)
    activos = c3.number_input("Activos totales", min_value=1.0, value=3500.0, step=100.0)
    pasivos = c4.number_input("Pasivos totales", min_value=0.0, value=2100.0, step=100.0)

    if pasivos >= activos:
        nota(
            "Con estas cifras el patrimonio es cero o negativo. La empresa quedaría fuera de los grupos, "
            f"junto a las {len(apartadas)} que se apartaron por la misma razón: sus deudas superan sus activos.",
            "aviso",
        )
    else:
        # El modelo trabaja en miles de pesos.
        res = an.ubicar_empresa({"ingresos": ingresos * 1000, "utilidad": utilidad * 1000,
                                 "activos": activos * 1000, "pasivos": pasivos * 1000}, prep, acp, km, n_comp)
        g_emp = res["grupo"]
        segunda = res["distancias"].index[1]
        c1, c2, c3 = st.columns(3)
        c1.metric("Grupo más cercano", etiqueta(g_emp))
        c2.metric("Distancia a su centroide", num(res["distancias"].iloc[0], 2))
        c3.metric("Segundo más cercano", f"{etiqueta(segunda, corta=True)} ({num(res['distancias'].iloc[1], 2)})")

        referencia = variables[pesca].median()
        comparacion = pd.DataFrame({
            "Tu empresa": [valor(v, res["indicadores"][v]) for v in variables.columns],
            f"Mediana del grupo {g_emp}": [valor(v, variables.loc[grupo == g_emp, v].median()) for v in variables.columns],
            "Mediana pesca y procesamiento": [valor(v, referencia[v]) for v in variables.columns],
            "Mediana de la base": [valor(v, variables[v].median()) for v in variables.columns],
        }, index=[an.nombre(v) if v != "tamano" else "Ingresos" for v in variables.columns])
        st.dataframe(comparacion, width="stretch")

        recortados = [an.nombre(v) for v in variables.columns
                      if not np.isclose(res["indicadores"][v], res["recortada"][v])]
        if "tamano" in variables.columns and ingresos * 1000 < datos["INGRESOS_OPERACIONALES"].min():
            nota(
                f"Tu empresa vende menos que la más pequeña de la base "
                f"({mm(datos['INGRESOS_OPERACIONALES'].min(), 1)} mil millones). El modelo la trata como si "
                f"tuviera el tamaño mínimo. La comparación de rentabilidad, deuda y rotación sigue siendo "
                f"válida como referencia, pero las empresas de la base operan a otra escala.",
                "aviso",
            )
        elif recortados:
            nota("Estos indicadores superan los límites del recorte y se ajustaron al límite antes de "
                 f"ubicar la empresa: {', '.join(recortados)}.", "aviso")

        fig = go.Figure()
        ejes = list(matriz.columns[:2])
        puntos = pd.concat([matriz, grupo], axis=1)
        for g in letras:
            sub = puntos[puntos["Grupo"] == g]
            fig.add_scattergl(x=sub[ejes[0]], y=sub[ejes[1]], mode="markers", name=etiqueta(g),
                              marker=dict(size=4, color=color_grupo[g], opacity=0.35), hoverinfo="skip")
        sub = pd.concat([matriz[pesca], ident.loc[pesca, "Empresa"]], axis=1)
        fig.add_scatter(x=sub[ejes[0]], y=sub[ejes[1]], mode="markers", name="Pesca y procesamiento",
                        marker=dict(size=10, color="white", line=dict(color=TINTA, width=2)),
                        customdata=sub[["Empresa"]], hovertemplate="<b>%{customdata[0]}</b><extra></extra>")
        fig.add_scatter(x=[res["puntaje"][ejes[0]]], y=[res["puntaje"][ejes[1]]], mode="markers+text",
                        name="Tu empresa", text=["  Tu empresa"], textposition="middle right",
                        textfont=dict(size=14, color=TINTA),
                        marker=dict(size=20, color=CORAL, symbol="star", line=dict(color="white", width=1.5)),
                        hovertemplate="Tu empresa<extra></extra>")
        fig.update_layout(title=f"Tu empresa entre las {num(len(matriz), 0)} del modelo ({ejes[0]} y {ejes[1]})",
                          xaxis_title=ejes[0], yaxis_title=ejes[1])
        st.plotly_chart(estilo(fig, 540), width="stretch")

        st.markdown("**Las empresas de la base con el perfil más parecido**")
        cercanas = pd.concat([ident.loc[res["vecinos"], ["Empresa", "Actividad", "Departamento"]],
                              grupo[res["vecinos"]]], axis=1)
        for v in variables.columns:
            if v != "tamano":
                cercanas[an.nombre(v)] = [valor(v, x) for x in variables.loc[res["vecinos"], v]]
        st.dataframe(cercanas, hide_index=True, width="stretch")
        st.caption("Parecido medido por distancia en el espacio de componentes. Sirve para saber con quién "
                   "compararse, no como diagnóstico: el modelo solo conoce cinco cuentas de un año.")

# ==========================================================================
# 8. CONCLUSIONES
# ==========================================================================
else:
    st.title("Conclusiones")
    if not es_por_defecto:
        evaluacion = correr_evaluacion(matriz, semilla)
        nota("Estás viendo una configuración distinta a la del informe. El texto de abajo describe la "
             "configuración por defecto (6 indicadores, recorte del 1 %, 3 componentes, K = 5, semilla 42); "
             "el resumen de tu corrida aparece primero.", "aviso")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Indicadores", f"{len(seleccion)}")
        c2.metric("Varianza conservada", pct(varianza["Varianza acumulada"].iloc[n_comp - 1]))
        c3.metric("K", f"{k}")
        c4.metric("Silueta", num(evaluacion.set_index("K").loc[k, "Silueta"], 3))
        for g in letras:
            st.markdown(f"- **Grupo {g}** ({num(int((grupo == g).sum()), 0)} empresas): {descriptores(g)}.")
        st.divider()

    st.subheader("1. ¿Cuántas ideas hay detrás de los indicadores?")
    st.markdown(
        "Tres. Los seis indicadores se resumen en tres componentes con valor propio mayor que 1, que conservan "
        "el **75,5 %** de la variación: **rentabilidad** (CP1, 36,9 %), donde ROA, ROE y margen suben juntos "
        "y la deuda baja; **modelo de volumen con deuda** (CP2, 21,2 %), que opone rotación y endeudamiento "
        "al margen; y **tamaño** (CP3, 17,5 %). Margen, ROA y ROE cuentan en buena parte la misma historia, "
        "y el tamaño es independiente de las demás: entre las empresas grandes, vender más no hace a una "
        "empresa más rentable ni más sólida."
    )
    st.subheader("2. ¿Hay grupos con una salud financiera parecida?")
    st.markdown(
        "Sí, cinco, aunque con fronteras difusas. K = 5 maximiza la silueta (0,31) y es estable entre semillas "
        "(ARI de 0,89). Los perfiles se reconocen: **solventes de rentabilidad moderada** (52 % de las "
        "empresas), **grandes y apalancadas** (24 %, con el 69 % de los ingresos), **rentables y poco "
        "endeudadas** (9 %), **bajo margen y alto volumen** (9 %) y **en pérdidas** (6 %). A ellos se suman "
        "254 empresas con patrimonio negativo, apartadas por regla. La silueta dice que es un continuo más "
        "que islas separadas: el 7 % de las empresas queda más cerca de otro grupo que del suyo, sobre todo "
        "en los grupos C y D. El hallazgo más útil es que hay dos caminos hacia un ROE alto: el del margen "
        "con poca deuda y el de la rotación con mucha deuda."
    )
    st.subheader("3. ¿El perfil depende del sector? ¿Dónde queda una empresa de mariscos?")
    st.markdown(
        "El sector inclina la balanza, pero no decide. En todos los macrosectores y regiones cerca de la mitad "
        "de las empresas está en el grupo central. En la cadena de alimentos sí hay un patrón: quien pesca y "
        "procesa cae en los grupos A y B, con un margen mediano de 1,6 % y una rotación cercana a dos veces; "
        "quien distribuye y vende al consumidor entra entre tres y cinco veces más que el promedio en el "
        "grupo de alta rotación. "
        "Para una empresa de mariscos, la pregunta que deja el análisis no es «¿mi margen es bajo?», porque "
        "en alimentos casi todos lo son, sino «¿mis activos rotan lo suficiente para compensarlo, y cuánta "
        "deuda estoy usando para sostener esa rotación?». La sección 7 permite hacer esa comparación con "
        "cifras propias."
    )
    st.subheader("Limitaciones")
    st.markdown(
        "- **Solo empresas grandes.** La base empieza en ventas de unos 21 mil millones de pesos. Una "
        "empresa pequeña se puede comparar, pero el modelo no aprendió de empresas de su tamaño.\n"
        "- **Pocas empresas de pesca.** Hay 20 de pesca, acuicultura y procesamiento de pescado en el modelo. "
        "Lo que se dice de ese eslabón son observaciones, no estadísticas firmes.\n"
        "- **Cinco cuentas y un año.** Sin flujo de caja, utilidad operacional ni activos corrientes no se "
        "puede medir liquidez, que es lo que más aprieta a una empresa de perecederos. Y un mal año no "
        "equivale a mala salud.\n"
        "- **Se pierde información.** Los tres componentes dejan fuera cerca de una cuarta parte de la variación.\n"
        "- **K-Means prefiere grupos redondos y de tamaño parecido.** Con una silueta moderada, otra técnica "
        "(jerárquico, mezclas gaussianas) trazaría fronteras algo distintas.\n"
        "- **Los grupos describen, no califican.** «En pérdidas» no significa «en quiebra»: una empresa "
        "puede perder un año porque está invirtiendo para crecer, y la base no permite distinguirlo."
    )
    st.subheader("Uso posible")
    st.markdown(
        "Sirve como punto de referencia para un diagnóstico financiero: antes de decir que un indicador es "
        "bueno o malo, ubicar a la empresa en su perfil y compararla con las que funcionan como ella. También "
        "sirve a quien otorga crédito o busca proveedores para distinguir una deuda alta que es parte del "
        "modelo de negocio (grupo D) de una deuda alta que acompaña pérdidas (grupo E)."
    )

    st.subheader("Descargar resultados")
    salida = pd.concat([ident, datos[list(an.CUENTAS)].rename(columns=an.CUENTAS), variables, matriz,
                        grupo, km["silueta"]], axis=1)
    st.download_button("Descargar empresas con su grupo (CSV)", salida.to_csv(index=False).encode("utf-8-sig"),
                       file_name="empresas_grupos.csv", mime="text/csv")
    st.caption(f"{num(len(salida), 0)} empresas · cuentas en miles de pesos, indicadores, puntajes de "
               "componentes, grupo y silueta.")
