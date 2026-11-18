"""Funciones del análisis: carga, preparación, ACP y K-Means.

La aplicación (app.py) llama a estas funciones cada vez que cambia un parámetro:
ninguna tabla, métrica o gráfico está guardado de antemano.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_samples, silhouette_score
from sklearn.preprocessing import StandardScaler

ARCHIVO = Path(__file__).resolve().parent / "data" / "supersociedades_10000_empresas_2025.csv"
FECHA_CONSULTA = "18 de noviembre de 2026"
SEMILLA = 42

# Las cifras del archivo están en miles de pesos colombianos.
CUENTAS = {
    "INGRESOS_OPERACIONALES": "Ingresos operacionales",
    "GANANCIA_PERDIDA": "Utilidad neta",
    "TOTAL_ACTIVOS": "Activos",
    "TOTAL_PASIVOS": "Pasivos",
    "TOTAL_PATRIMONIO": "Patrimonio",
}

VARIABLES = {
    "margen_neto": {
        "nombre": "Margen neto",
        "formula": "utilidad neta / ingresos operacionales",
        "lectura": "Cuántos pesos de utilidad deja cada peso vendido.",
        "dimension": "Rentabilidad",
        "formato": "pct",
    },
    "roa": {
        "nombre": "ROA",
        "formula": "utilidad neta / activos",
        "lectura": "Cuánto rinde cada peso invertido en activos.",
        "dimension": "Rentabilidad",
        "formato": "pct",
    },
    "roe": {
        "nombre": "ROE",
        "formula": "utilidad neta / patrimonio",
        "lectura": "Cuánto rinde cada peso que pusieron los dueños.",
        "dimension": "Rentabilidad",
        "formato": "pct",
    },
    "endeudamiento": {
        "nombre": "Endeudamiento",
        "formula": "pasivos / activos",
        "lectura": "Qué parte de los activos se financia con deuda.",
        "dimension": "Estructura financiera",
        "formato": "pct",
    },
    "rotacion": {
        "nombre": "Rotación de activos",
        "formula": "ingresos operacionales / activos",
        "lectura": "Cuántas veces al año vende el valor de sus activos.",
        "dimension": "Eficiencia",
        "formato": "veces",
    },
    "tamano": {
        "nombre": "Tamaño",
        "formula": "log10(ingresos operacionales)",
        "lectura": "Más alto = empresa que vende más. Cada unidad es diez veces más ventas.",
        "dimension": "Tamaño",
        "formato": "log",
    },
}

# Actividades cercanas a una empresa de mariscos: de la captura a la mesa.
CADENA_ALIMENTOS = {
    "Pesca y acuicultura": ["A0311", "A0312", "A0321", "A0322"],
    "Procesamiento de pescado y mariscos": ["C1012"],
    "Comercio minorista de carnes y pescados": ["G4723"],
    "Comercio mayorista de alimentos": ["G4631"],
    "Supermercados y tiendas de alimentos": ["G4711"],
    "Restaurantes y servicios de comida": ["I5611", "I5612", "I5613", "I5619", "I5621", "I5629"],
}


def nombre(variable):
    return VARIABLES[variable]["nombre"] if variable in VARIABLES else variable


# --------------------------------------------------------------------------
# 1. Carga
# --------------------------------------------------------------------------
def cargar_crudo():
    """Lee el archivo y separa el código CIIU de su descripción."""
    crudo = pd.read_csv(ARCHIVO)
    partes = crudo["CIIU"].str.extract(r"^([A-Z])(\d{4}) - (.*)$")
    crudo["CIIU_CODIGO"] = partes[0] + partes[1]
    crudo["ACTIVIDAD"] = partes[2]
    crudo["MACROSECTOR"] = crudo["MACROSECTOR"].str.title().str.replace(" - ", " e ")
    crudo["REGION"] = crudo["REGION"].str.title().str.replace("-", " y ")
    eslabon = {codigo: nombre_e for nombre_e, codigos in CADENA_ALIMENTOS.items() for codigo in codigos}
    crudo["ESLABON"] = crudo["CIIU_CODIGO"].map(eslabon)
    return crudo


# --------------------------------------------------------------------------
# 2. Limpieza con registro de decisiones
# --------------------------------------------------------------------------
def limpiar(crudo):
    """Devuelve (empresas del modelo, empresas apartadas, bitácora de decisiones)."""
    bitacora = []

    faltantes = int(crudo[list(CUENTAS)].isna().sum().sum())
    duplicados = int(crudo["NIT"].duplicated().sum())
    bitacora.append({
        "Revisión": "Valores faltantes y NIT duplicados",
        "Resultado": f"{faltantes} celdas vacías en las cinco cuentas y {duplicados} NIT repetidos.",
        "Decisión": "No hay nada que imputar ni que eliminar.",
        "Empresas": 0,
    })

    descuadre = (crudo["TOTAL_ACTIVOS"] - crudo["TOTAL_PASIVOS"] - crudo["TOTAL_PATRIMONIO"]).abs()
    descuadradas = int((descuadre > 0.01 * crudo["TOTAL_ACTIVOS"].abs()).sum())
    bitacora.append({
        "Revisión": "Ecuación contable (activos = pasivos + patrimonio)",
        "Resultado": f"{descuadradas} empresas con una diferencia mayor al 1 % de sus activos.",
        "Decisión": "Las cifras son coherentes; se usan como vienen.",
        "Empresas": 0,
    })

    sin_base = (crudo["TOTAL_ACTIVOS"] <= 0) | (crudo["INGRESOS_OPERACIONALES"] <= 0)
    bitacora.append({
        "Revisión": "Activos o ingresos en cero o negativos",
        "Resultado": f"{int(sin_base.sum())} empresas.",
        "Decisión": "Todas las razones tienen denominador válido.",
        "Empresas": int(sin_base.sum()),
    })

    negativo = crudo["TOTAL_PATRIMONIO"] <= 0
    bitacora.append({
        "Revisión": "Patrimonio en cero o negativo",
        "Resultado": f"{int(negativo.sum())} empresas deben más de lo que tienen.",
        "Decisión": "Se apartan del modelo: con patrimonio negativo el ROE no se puede interpretar "
                    "(una pérdida dividida por un patrimonio negativo da un ROE «positivo»). "
                    "Se describen por separado como un grupo definido por regla.",
        "Empresas": int(negativo.sum()),
    })

    bitacora.append({
        "Revisión": "Columnas que no son medidas",
        "Resultado": "NIT, razón social, supervisor, región, departamento, ciudad, CIIU y macrosector.",
        "Decisión": "No entran al modelo. El sector y la región se guardan para contrastar los grupos.",
        "Empresas": 0,
    })

    datos = crudo[~negativo & ~sin_base].reset_index(drop=True)
    apartadas = crudo[negativo].reset_index(drop=True)
    return datos, apartadas, pd.DataFrame(bitacora)


# --------------------------------------------------------------------------
# 3. Variables de análisis
# --------------------------------------------------------------------------
def indicadores(ingresos, utilidad, activos, pasivos, patrimonio):
    """Los seis indicadores a partir de las cinco cuentas (sirve para series o para un número)."""
    return {
        "margen_neto": utilidad / ingresos,
        "roa": utilidad / activos,
        "roe": utilidad / patrimonio,
        "endeudamiento": pasivos / activos,
        "rotacion": ingresos / activos,
        "tamano": np.log10(ingresos),
    }


def construir_variables(datos):
    variables = pd.DataFrame(indicadores(
        datos["INGRESOS_OPERACIONALES"], datos["GANANCIA_PERDIDA"], datos["TOTAL_ACTIVOS"],
        datos["TOTAL_PASIVOS"], datos["TOTAL_PATRIMONIO"],
    ), index=datos.index)
    identificacion = datos[["RANKING", "NIT", "RAZON_SOCIAL", "MACROSECTOR", "REGION", "DEPARTAMENTO",
                            "CIUDAD", "CIIU_CODIGO", "ACTIVIDAD", "ESLABON"]].rename(columns={
        "RANKING": "Puesto", "RAZON_SOCIAL": "Empresa", "MACROSECTOR": "Macrosector", "REGION": "Región",
        "DEPARTAMENTO": "Departamento", "CIUDAD": "Ciudad", "CIIU_CODIGO": "CIIU", "ACTIVIDAD": "Actividad",
        "ESLABON": "Eslabón",
    })
    return identificacion, variables


def winsorizar(variables, nivel):
    """Recorta cada variable en los percentiles nivel y 1 − nivel."""
    if nivel <= 0:
        limites = pd.DataFrame({"Límite inferior": variables.min(), "Límite superior": variables.max(),
                                "Valores recortados": 0})
        return variables.copy(), limites
    inferior = variables.quantile(nivel)
    superior = variables.quantile(1 - nivel)
    recortadas = variables.clip(lower=inferior, upper=superior, axis=1)
    limites = pd.DataFrame({
        "Límite inferior": inferior,
        "Límite superior": superior,
        "Valores recortados": variables.ne(recortadas).sum(),
    })
    return recortadas, limites


def preparar(nivel_winsor=0.01, seleccion=None):
    """Recorrido completo de preparación. Devuelve un diccionario con cada tramo."""
    crudo = cargar_crudo()
    datos, apartadas, bitacora = limpiar(crudo)
    identificacion, variables = construir_variables(datos)
    if seleccion is not None:
        variables = variables[list(seleccion)]
    recortadas, limites = winsorizar(variables, nivel_winsor)
    escalador = StandardScaler().fit(recortadas)
    z = pd.DataFrame(escalador.transform(recortadas), index=variables.index, columns=variables.columns)
    return {
        "crudo": crudo, "datos": datos, "apartadas": apartadas, "bitacora": bitacora,
        "identificacion": identificacion, "variables": variables, "recortadas": recortadas,
        "limites": limites, "escalador": escalador, "z": z,
    }


# --------------------------------------------------------------------------
# 4. Análisis de componentes principales
# --------------------------------------------------------------------------
def ajustar_acp(z):
    """ACP con todos los componentes: varianza, valores propios y cargas."""
    acp = PCA(random_state=SEMILLA).fit(z)
    etiquetas = [f"CP{i + 1}" for i in range(acp.n_components_)]
    varianza = pd.DataFrame({
        "Componente": etiquetas,
        "Valor propio": acp.explained_variance_,
        "Varianza explicada": acp.explained_variance_ratio_,
        "Varianza acumulada": np.cumsum(acp.explained_variance_ratio_),
    })
    cargas = pd.DataFrame(acp.components_.T, index=z.columns, columns=etiquetas)
    # Correlación entre cada variable y cada componente (carga × √valor propio).
    correlaciones = cargas * np.sqrt(acp.explained_variance_)
    puntajes = pd.DataFrame(acp.transform(z), index=z.index, columns=etiquetas)
    return {"modelo": acp, "varianza": varianza, "cargas": cargas,
            "correlaciones": correlaciones, "puntajes": puntajes}


def sugerir_componentes(varianza, umbral=0.80):
    """Componentes según Kaiser (valor propio > 1) y según el umbral de varianza acumulada."""
    kaiser = int((varianza["Valor propio"] > 1).sum())
    acumulada = int((varianza["Varianza acumulada"] < umbral).sum() + 1)
    return {"kaiser": max(kaiser, 1), "acumulada": min(acumulada, len(varianza))}


# --------------------------------------------------------------------------
# 5. K-Means
# --------------------------------------------------------------------------
def evaluar_k(matriz, k_min=2, k_max=8, semilla=SEMILLA):
    filas = []
    for k in range(k_min, k_max + 1):
        modelo = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(matriz)
        filas.append({
            "K": k,
            "Inercia": modelo.inertia_,
            "Silueta": silhouette_score(matriz, modelo.labels_),
            "Grupo más pequeño": int(np.bincount(modelo.labels_).min()),
        })
    return pd.DataFrame(filas)


def ajustar_kmeans(matriz, k, semilla=SEMILLA):
    modelo = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(matriz)
    # Se renombran los grupos por tamaño (A = el más numeroso) para que la etiqueta
    # no cambie entre corridas. La letra no implica orden ni jerarquía.
    tamanos = pd.Series(modelo.labels_).value_counts()
    letras = {grupo: chr(65 + i) for i, grupo in enumerate(tamanos.index)}
    etiquetas = pd.Series(modelo.labels_, index=matriz.index).map(letras)
    silueta = pd.Series(silhouette_samples(matriz, modelo.labels_), index=matriz.index)
    centroides = pd.DataFrame(modelo.cluster_centers_, columns=matriz.columns,
                              index=[letras[i] for i in range(k)]).sort_index()
    return {"grupo": etiquetas.rename("Grupo"), "silueta": silueta.rename("Silueta"), "centroides": centroides}


def estabilidad(matriz, k, semillas=range(10)):
    """Índice de Rand ajustado entre la solución base y otras semillas."""
    base = KMeans(n_clusters=k, n_init=10, random_state=SEMILLA).fit_predict(matriz)
    valores = [
        adjusted_rand_score(base, KMeans(n_clusters=k, n_init=1, random_state=s).fit_predict(matriz))
        for s in semillas
    ]
    return float(np.mean(valores)), float(np.min(valores))


def perfiles(variables, grupo):
    """Mediana de cada variable original por grupo, más el total."""
    tabla = variables.groupby(grupo).median().T
    tabla["Total"] = variables.median()
    return tabla


def perfiles_z(z, grupo):
    """Media estandarizada por grupo: cuántas desviaciones se aleja del promedio."""
    return z.groupby(grupo).mean().T


def composicion(categoria, grupo):
    """Porcentaje de cada categoría (sector, región, eslabón) que cae en cada grupo."""
    return pd.crosstab(categoria, grupo, normalize="index")


# --------------------------------------------------------------------------
# 6. Ubicar una empresa nueva
# --------------------------------------------------------------------------
def ubicar_empresa(cuentas, prep, acp, km, n_comp, vecinos=8):
    """Aplica a una empresa nueva los mismos pasos del modelo y le asigna el grupo más cercano.

    cuentas: diccionario con ingresos, utilidad, activos y pasivos (misma unidad para los cuatro).
    """
    patrimonio = cuentas["activos"] - cuentas["pasivos"]
    fila = pd.DataFrame([indicadores(cuentas["ingresos"], cuentas["utilidad"], cuentas["activos"],
                                     cuentas["pasivos"], patrimonio)])[list(prep["variables"].columns)]
    limites = prep["limites"]
    recortada = fila.clip(lower=limites["Límite inferior"], upper=limites["Límite superior"], axis=1)
    z = pd.DataFrame(prep["escalador"].transform(recortada), columns=fila.columns)
    puntaje = pd.DataFrame(acp["modelo"].transform(z), columns=acp["puntajes"].columns).iloc[:, :n_comp]
    distancias = ((km["centroides"] - puntaje.iloc[0]) ** 2).sum(axis=1) ** 0.5
    cercania = ((acp["puntajes"].iloc[:, :n_comp] - puntaje.iloc[0]) ** 2).sum(axis=1) ** 0.5
    return {
        "indicadores": fila.iloc[0],
        "recortada": recortada.iloc[0],
        "puntaje": puntaje.iloc[0],
        "grupo": distancias.idxmin(),
        "distancias": distancias.sort_values(),
        "vecinos": cercania.nsmallest(vecinos).index,
    }
