# Salud financiera de las empresas colombianas · ACP y K-Means

Taller individual · Machine Learning aplicado a Economía y Finanzas · Pontificia Universidad Javeriana Cali
Autora: Camila Andrea Granja Castro

- **Aplicación:** https://salud-financiera-empresas-colombia.streamlit.app/
- **Video de sustentación:** `<enlace al video>`

## Contexto y pregunta

Tengo una empresa de mariscos y quería saber si sus números son buenos o malos. La respuesta depende de con quién se compare: un margen del 3 % es pobre para una inmobiliaria y normal para una distribuidora de alimentos. Este proyecto usa los estados financieros de las 10.000 empresas más grandes de Colombia para encontrar los perfiles de salud financiera que existen en el país y ubicar en ellos a las empresas de pesca, procesamiento de pescado y comercio de alimentos.

1. ¿Cuántas ideas distintas hay detrás de los indicadores financieros básicos?
2. ¿Existen grupos de empresas con una salud financiera parecida? ¿Qué los define?
3. ¿El perfil depende del sector? ¿Dónde caen las empresas de la cadena del pescado y los alimentos, y dónde quedaría una empresa como la mía?

**Por qué ACP y K-Means.** Los indicadores comparten ingredientes: la utilidad está en el numerador de tres de ellos y los activos en el denominador de otros tres, así que están correlacionados por construcción. El ACP junta los que se mueven a la par en ejes no correlacionados, mide cuánta información conserva cada eje y evita que la rentabilidad cuente tres veces en las distancias. Nadie ha clasificado a estas empresas como «sanas» o «en riesgo», así que el problema es no supervisado: K-Means agrupa a las empresas cercanas en el espacio de componentes y entrega centroides que se leen como perfiles. El sector y la región no entran al modelo; se usan al final como contraste.

## Datos

| | |
|---|---|
| Base | Informe *10.000 empresas más grandes del país 2026*, Superintendencia de Sociedades ([comunicado del 30 de junio de 2026](https://www.supersociedades.gov.co/noticias-supersociedades/-/asset_publisher/atwl/content/las-10.000-empresas-m%C3%A1s-grandes-de-colombia-registraron-ingresos-por-1.852-9-billones-y-utilidades-por-138-1-billones-al-cierre-de-2025)) |
| Corte | Estados financieros al 31 de diciembre de 2025 |
| Observación | Una empresa por fila: 10.000 sociedades, ordenadas por ingresos operacionales |
| Columnas | Puesto, NIT, razón social, supervisor, región, departamento, ciudad, CIIU, macrosector, ingresos operacionales, utilidad neta, activos, pasivos y patrimonio (cifras en miles de pesos) |
| Archivo | `data/supersociedades_10000_empresas_2025.csv` |

**Cómo se obtuvieron.** La Supersociedades publica esta base en un [tablero de Power BI](https://app.powerbi.com/view?r=eyJrIjoiM2FmYWUxMGQtM2M2ZC00NDhjLWJiMGEtYjJlNTBlZWQ3ZWViIiwidCI6IjZlZTk0YzM0LWJiZDYtNDY0Ny1hNDgzLTBlMTk2YTRkZTBmZiIsImMiOjR9) que no tiene botón de descarga. El 18 de noviembre de 2026 se leyó la tabla «Matriz comparativa 2025-2024» del modelo público de ese tablero y se guardaron las filas de la vigencia 2025 como CSV, sin modificar ninguna cifra. Los totales del archivo son: ingresos por 1.853,8 billones de pesos, utilidades por 138,7 billones, activos por 2.560,4 billones, pasivos por 1.376,7 billones y patrimonio por 1.183,7 billones. El comunicado, en la versión consultada ese día, reporta 1.853,8; 138,6; 2.560,4; 1.376,7 y 1.183,7 billones: la única diferencia está en las utilidades (el archivo suma 138,67), de 0,05 %. La primera versión del comunicado, que quedó en la dirección de la página y en las notas de prensa del 30 de junio, hablaba de ingresos por 1.852,9 billones y utilidades por 138,1 billones; frente a esas cifras el archivo difiere en 0,05 % y 0,4 %. La copia de esta base en datos.gov.co no sirve para este análisis porque redondea las cifras a dos decimales de billón, lo que deja la utilidad de la mayoría de las empresas en cero.

**Indicadores del modelo** (construidos en `analisis.py`):

| Indicador | Cálculo | Dimensión |
|---|---|---|
| Margen neto | utilidad neta / ingresos operacionales | Rentabilidad |
| ROA | utilidad neta / activos | Rentabilidad |
| ROE | utilidad neta / patrimonio | Rentabilidad |
| Endeudamiento | pasivos / activos | Estructura financiera |
| Rotación de activos | ingresos operacionales / activos | Eficiencia |
| Tamaño | log10(ingresos operacionales) | Tamaño |

**Limitaciones de la fuente.** Solo incluye empresas grandes (la más pequeña vendió 21,3 mil millones de pesos); trae cinco cuentas de un solo año, sin utilidad operacional, flujo de caja ni activos corrientes, así que no se puede medir liquidez; son estados individuales, por lo que las casas matrices muestran márgenes inflados; y cada empresa tiene un único código CIIU, el de su actividad principal declarada.

## Metodología

1. **Revisión de calidad** (bitácora en la aplicación, sección 3): no hay celdas vacías ni NIT duplicados, y la ecuación contable cuadra en todas las empresas.
2. **Patrimonio negativo:** 254 empresas deben más de lo que tienen. Se apartan del modelo porque su ROE no se puede interpretar (una pérdida sobre un patrimonio negativo da un ROE «positivo») y se describen por separado. Quedan 9.746 empresas.
3. **Valores extremos:** winsorización al 1 % por cola (ajustable en la aplicación). Sin ella, unos pocos casos con ROE de más de 200 o rotación de más de 80 dominan la estandarización.
4. **Estandarización:** z-score.
5. **ACP:** se calculan todos los componentes y se conservan los que cumplen el criterio de Kaiser (λ > 1), contrastado con el 80 % de varianza acumulada y con el gráfico de sedimentación.
6. **K-Means** sobre los puntajes de los componentes conservados: K de 2 a 8, método del codo, silueta y estabilidad entre semillas (índice de Rand ajustado). 10 inicios aleatorios, semilla 42.
7. **Perfiles** con los indicadores originales (medianas y medias estandarizadas) y composición por macrosector, región y eslabón de la cadena de alimentos.
8. **Ubicación de una empresa nueva:** a partir de cuatro cifras se calculan los indicadores, se les aplican el mismo recorte, la misma estandarización y los mismos componentes, y se asigna el centroide más cercano.

## Resultados principales

**ACP.** Tres componentes tienen λ > 1 y conservan el **75,5 %** de la varianza (el 80 % exige cuatro; el cuarto tiene λ = 0,78):

| Componente | Varianza | Lectura |
|---|---|---|
| CP1 | 36,9 % | **Rentabilidad:** ROA, ROE y margen suben juntos; el endeudamiento baja |
| CP2 | 21,2 % | **Modelo de volumen con deuda:** rotación, endeudamiento y ROE frente a margen |
| CP3 | 17,5 % | **Tamaño:** ingresos, con la rotación en contra |

**K-Means.** K = 5 maximiza la silueta (0,307; K = 2 da 0,291) y es estable entre semillas (ARI medio de 0,89; con K = 3, 4 o 6 queda entre 0,50 y 0,69). El 7,1 % de las empresas tiene silueta negativa: la estructura es débil a moderada.

| Grupo | Empresas | Perfil (medianas) | Ejemplos |
|---|---:|---|---|
| A · Solventes de rentabilidad moderada | 5.085 | Endeudamiento 47 %, margen 3,6 %, ROE 10 %, ventas de 38 mil millones | Constructoras, distribuidoras de materiales, transportadoras |
| B · Grandes y apalancadas | 2.358 | Ventas de 196 mil millones, endeudamiento 67 %, margen 2,5 %; concentran el 69 % de los ingresos | Ecopetrol, Terpel, D1, EPM, Éxito |
| C · Rentables y poco endeudadas | 901 | Margen 23 %, ROA 21 %, endeudamiento 27 %, rotación 0,97 | Cenit, Bavaria, Grupo Argos, inmobiliarias |
| D · Bajo margen y alto volumen | 831 | Rotación 6,0 veces, margen 1,8 %, endeudamiento 67 %, ROE 31 % | Estaciones de servicio, mayoristas de alimentos, supermercados |
| E · En pérdidas | 571 | Todas con pérdidas: margen −5,9 %, ROE −28 %, endeudamiento 76 % | Jerónimo Martins, Colombia Telecomunicaciones, MercadoLibre |

Las letras solo nombran a los grupos (A es el más numeroso); no indican orden. Los grupos C y D rinden lo mismo sobre el patrimonio (cerca de 31 %) por caminos opuestos: uno con margen y poca deuda, el otro con rotación y mucha deuda.

**Sectores y cadena de alimentos.** Ningún macrosector ni región tiene un grupo propio. En la cadena de alimentos sí hay un patrón: las 20 empresas de pesca, acuicultura y procesamiento de pescado se reparten entre A (10) y B (8), con dos en pérdidas, un margen mediano de 1,6 % y una rotación cercana a dos veces; en cambio, el 29 % de los mayoristas de alimentos, el 31 % de los supermercados y el 42 % de los minoristas de carnes y pescados están en el grupo de alta rotación, frente al 9 % de toda la base.

**Limitaciones de los resultados.** Hay solo 20 empresas de pesca y procesamiento en el modelo; los tres componentes dejan fuera cerca de una cuarta parte de la variación; la base no representa a empresas pequeñas; y un grupo describe un año, no califica a la empresa.

## Estructura del repositorio

```
├── app.py                 Aplicación de Streamlit (8 secciones)
├── analisis.py            Carga, limpieza, indicadores, ACP, K-Means y ubicación de una empresa nueva
├── data/
│   └── supersociedades_10000_empresas_2025.csv
├── requirements.txt
└── .streamlit/config.toml Tema visual
```

## Cómo ejecutar

```bash
python3 -m venv .venv
source .venv/bin/activate          # En Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app.py
```

La aplicación abre en `http://localhost:8501`. Todos los cálculos se hacen al cargarla con el archivo de `data/`; al cambiar los indicadores, el recorte, el número de componentes, K o la semilla, el análisis completo se vuelve a correr.

**Despliegue:** en [share.streamlit.io](https://share.streamlit.io) → *Create app* → repositorio público → rama `main` → archivo `app.py`.
