# Organic Reveal API

API standalone que convierte una imagen PNG, JPEG o WebP en un MP4. El render revela primero los contornos, texturas y detalles **conservando sus colores originales**; después detecta regiones visuales propias de cada imagen y las colorea secuencialmente siguiendo sus límites.

## Características

- FastAPI y documentación OpenAPI en `/docs`.
- Cola de trabajos asíncrona dentro del mismo contenedor.
- `N` renders simultáneos mediante `MAX_CONCURRENT_RENDERS`.
- Hilos de codificación por render configurables mediante `FFMPEG_THREADS`.
- Purga automática mediante `JOB_TTL_HOURS`.
- Resultados reproducibles mediante `seed`.
- Render CPU con OpenCV, NumPy y FFmpeg.
- Segmentación automática local mediante SAM 2.1 Hiera Tiny.
- Orden dinámico por saliencia, posición o tamaño.
- Previsualización numerada de las regiones detectadas.
- Autenticación mediante `X-API-Key`.
- Datos persistentes en `/app/data`.

## Inicio local

```bash
cp .env.example .env
docker compose --env-file .env up --build
```

`compose.yaml` solo expone el puerto al proxy interno, como requiere Coolify. Para probar directamente desde el host:

```bash
docker build -t organic-reveal-api .
docker run --rm -p 8000:8000 \
  -e API_KEYS=dev-secret-change-me \
  -e MAX_CONCURRENT_RENDERS=2 \
  -v organic-reveal-data:/app/data \
  organic-reveal-api
```

La documentación estará en `http://localhost:8000/docs`.

## Crear un render

```bash
curl -X POST http://localhost:8000/v1/renders \
  -H 'X-API-Key: dev-secret-change-me' \
  -F 'image=@/ruta/a/imagen.png' \
  -F 'duration=10' \
  -F 'fps=30' \
  -F 'detail_ratio=0.45' \
  -F 'detail_chroma=1.0' \
  -F 'detail_selectivity=0.5' \
  -F 'detail_mode=regions' \
  -F 'detail_feather=0.006' \
  -F 'brush_radius=0.12' \
  -F 'fill_brushes=3' \
  -F 'direction=reading-order' \
  -F 'segmentation_mode=auto' \
  -F 'region_order=saliency' \
  -F 'max_regions=12' \
  -F 'seed=3842'
```

Consultar el progreso:

```bash
curl -H 'X-API-Key: dev-secret-change-me' \
  http://localhost:8000/v1/renders/ID_DEL_RENDER
```

Descargar el resultado:

```bash
curl -L -H 'X-API-Key: dev-secret-change-me' \
  -o result.mp4 \
  http://localhost:8000/v1/renders/ID_DEL_RENDER/video
```

Descargar la previsualización de regiones:

```bash
curl -L -H 'X-API-Key: dev-secret-change-me' \
  -o regions.png \
  http://localhost:8000/v1/renders/ID_DEL_RENDER/regions
```

Eliminar manualmente:

```bash
curl -X DELETE -H 'X-API-Key: dev-secret-change-me' \
  http://localhost:8000/v1/renders/ID_DEL_RENDER
```

## Parámetros visuales

| Campo | Predeterminado | Función |
|---|---:|---|
| `duration` | `10` | Duración total en segundos |
| `fps` | `30` | Fotogramas por segundo |
| `detail_ratio` | `0.45` | Punto de la línea de tiempo donde termina la fase de detalles |
| `detail_chroma` | `1.0` | Color de la capa de detalle: `0` monocromo, `1` color original |
| `detail_selectivity` | `0.5` | Fuerza mínima del detalle: valores altos conservan menos textura y contornos más definidos |
| `detail_mode` | `regions` | `regions` entinta forma por forma; `legacy` conserva el barrido suave anterior |
| `detail_feather` | `0.006` | Suavidad temporal del frente de entintado en modo `regions` |
| `fill_overlap` | `0.08` | Superposición temporal entre detalles y relleno |
| `final_hold` | `0.6` | Tiempo que permanece la imagen terminada |
| `brush_radius` | `0.12` | Radio respecto al lado menor de la imagen |
| `brush_feather` | `0.16` | Suavidad temporal del borde del pincel |
| `fill_brushes` | `3` | Pinceles continuos que se desplazan simultáneamente |
| `direction` | `reading-order` | Dirección general del revelado |
| `segmentation_mode` | `auto` | `auto` segmenta por regiones; `none` usa el pincel global 0.2 |
| `region_order` | `saliency` | Estrategia para ordenar las regiones detectadas |
| `max_regions` | `12` | Máximo de regiones antes del relleno residual |
| `min_region_area` | `0.002` | Área mínima como fracción de la imagen |
| `background` | `#f1efe9` | Color del papel |
| `output_width` | `0` | Ancho de salida; `0` conserva el original |
| `seed` | `3842` | Reproduce exactamente el mismo movimiento |

Direcciones permitidas: `reading-order`, `right-to-left`, `left-to-right`, `top-to-bottom`, `bottom-to-top`, `center-out`, `organic` y `random-origins`.

`reading-order` comienza en la zona superior izquierda y combina avance de izquierda a derecha con descenso gradual. `random-origins` inicia la capa de detalles simultáneamente desde varios puntos determinados por `seed`.

`detail_chroma` afecta únicamente la primera fase. El relleno y la imagen final siempre recuperan el color original. `detail_selectivity=0.5` conserva la densidad de la versión 0.3.0; valores próximos a `0.7` eliminan más manchas y textura débil. Para separar completamente las dos fases usa `fill_overlap=0`.

Con `detail_mode=regions`, la API reutiliza las máscaras y el orden de SAM 2 para entintar cada forma antes de continuar con la siguiente. La máscara se endurece para evitar la apariencia de humo y `detail_feather` controla únicamente la estrecha transición del frente en movimiento. Los detalles que no pertenecen a una región se pintan al final mediante el pincel global. Si `segmentation_mode=none`, el mismo modo utiliza el pincel global sobre toda la capa de detalle.

Órdenes de región: `saliency`, `reading-order`, `center-first`, `large-first`, `small-first` y `random`. La segmentación no depende de categorías predefinidas: analiza cada imagen y produce máscaras nuevas. Las máscaras se limpian, se desduplican y se convierten en una partición sin solapamientos. Los píxeles no asignados forman una región residual que siempre se revela al final.

## Segmentación automática

La imagen Docker incluye SAM 2.1 Hiera Tiny y su checkpoint. El modelo se carga de forma diferida durante el primer render automático y permanece en memoria. Las inferencias se serializan para evitar duplicar el modelo; la codificación de videos continúa respetando `MAX_CONCURRENT_RENDERS`.

La respuesta final incluye:

```json
{
  "regions_detected": 9,
  "regions_url": "/v1/renders/<id>/regions",
  "video_url": "/v1/renders/<id>/video"
}
```

La previsualización colorea y numera las regiones en el orden en que serán pintadas. Para volver al comportamiento de la versión 0.2, envía `segmentation_mode=none`.

## Concurrencia y purga

`MAX_CONCURRENT_RENDERS=3` permite que tres trabajos rendericen simultáneamente. Los demás permanecen en cola hasta el límite `MAX_QUEUED_JOBS`.

`FFMPEG_THREADS` limita los hilos del codificador por trabajo y `OPENCV_THREADS` los hilos internos de OpenCV. En una máquina pequeña conviene empezar con `2` y `1`, respectivamente, para evitar que varios renders compitan por todos los núcleos.

`SAM2_CPU_THREADS` controla los hilos de inferencia. `SAM2_POINTS_PER_SIDE` determina la densidad de muestreo: `8` es rápido, `16` es el equilibrio predeterminado y valores mayores aumentan coste y detalle. `SAM2_ANALYSIS_SIZE` limita la resolución empleada para proponer máscaras; el video conserva su resolución de salida.

Cada trabajo se guarda en un directorio aislado:

```text
/app/data/<job-id>/
├── input.png
├── output.mp4
└── job.json
```

Un hilo de mantenimiento revisa cada `PURGE_INTERVAL_MINUTES` y elimina trabajos terminados con antigüedad igual o mayor que `JOB_TTL_HOURS`. Los trabajos activos nunca se purgan. Si el contenedor se reinicia durante un render, ese trabajo se marca como fallido; se puede enviar nuevamente con la misma `seed`.

## Variables de entorno

Consulta `.env.example`. En producción, configura `API_KEYS` como secreto de Coolify y nunca confirmes un archivo `.env` al repositorio.
