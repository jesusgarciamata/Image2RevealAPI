# Organic Reveal API

API standalone que convierte una imagen PNG, JPEG o WebP en un MP4. El render revela primero los contornos, texturas y detalles **conservando sus colores originales**; después descubre las zonas de color con un pincel amplio y orgánico.

## Características

- FastAPI y documentación OpenAPI en `/docs`.
- Cola de trabajos asíncrona dentro del mismo contenedor.
- `N` renders simultáneos mediante `MAX_CONCURRENT_RENDERS`.
- Hilos de codificación por render configurables mediante `FFMPEG_THREADS`.
- Purga automática mediante `JOB_TTL_HOURS`.
- Resultados reproducibles mediante `seed`.
- Render CPU con OpenCV, NumPy y FFmpeg.
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
  -F 'brush_radius=0.12' \
  -F 'direction=right-to-left' \
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
| `fill_overlap` | `0.08` | Superposición temporal entre detalles y relleno |
| `final_hold` | `0.6` | Tiempo que permanece la imagen terminada |
| `brush_radius` | `0.12` | Radio respecto al lado menor de la imagen |
| `brush_feather` | `0.16` | Suavidad temporal del borde del pincel |
| `direction` | `right-to-left` | Dirección general del revelado |
| `background` | `#f1efe9` | Color del papel |
| `output_width` | `0` | Ancho de salida; `0` conserva el original |
| `seed` | `3842` | Reproduce exactamente el mismo movimiento |

Direcciones permitidas: `right-to-left`, `left-to-right`, `top-to-bottom`, `bottom-to-top`, `center-out` y `organic`.

## Concurrencia y purga

`MAX_CONCURRENT_RENDERS=3` permite que tres trabajos rendericen simultáneamente. Los demás permanecen en cola hasta el límite `MAX_QUEUED_JOBS`.

`FFMPEG_THREADS` limita los hilos del codificador por trabajo y `OPENCV_THREADS` los hilos internos de OpenCV. En una máquina pequeña conviene empezar con `2` y `1`, respectivamente, para evitar que varios renders compitan por todos los núcleos.

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
