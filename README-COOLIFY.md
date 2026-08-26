# Despliegue en Coolify

## 1. Crear el recurso

1. Sube este directorio a un repositorio privado de GitHub o GitLab.
2. En Coolify, crea un recurso desde **Private Repository**.
3. Conecta el repositorio mediante GitHub App o deploy key.
4. Selecciona **Docker Compose** como build pack.
5. Usa `/` como Base Directory y `/compose.yaml` como Docker Compose Location.

No agregues una sección `networks` al Compose: Coolify crea la red y conecta el proxy automáticamente.

## 2. Variables

Define en Coolify:

```env
API_KEYS=una-llave-larga-generada-aleatoriamente
MAX_CONCURRENT_RENDERS=2
MAX_QUEUED_JOBS=50
FFMPEG_THREADS=2
OPENCV_THREADS=1
SAM2_DEVICE=cpu
SAM2_CPU_THREADS=4
SAM2_POINTS_PER_SIDE=16
SAM2_POINTS_PER_BATCH=16
SAM2_ANALYSIS_SIZE=768
MAX_UPLOAD_MB=20
MAX_PIXELS=20000000
JOB_TTL_HOURS=24
PURGE_INTERVAL_MINUTES=15
LOG_LEVEL=INFO
```

Una llave puede generarse localmente con:

```bash
openssl rand -hex 32
```

Para rotar sin interrumpir clientes, `API_KEYS` admite varias llaves separadas por comas. Añade la nueva, actualiza los clientes y finalmente retira la anterior.

## 3. Dominio

Asigna un dominio al servicio `organic-reveal-api` y selecciona el puerto interno `8000`, por ejemplo:

```text
https://reveal-api.example.com
```

Coolify/Traefik publicará el HTTPS exterior en el puerto estándar; el proceso dentro del contenedor continúa escuchando en `8000`.

## 4. Persistencia

El volumen nombrado `reveal-data` está declarado en `compose.yaml`. Contiene entradas, resultados y metadatos bajo `/app/data`. No lo borres al redeplegar.

La purga se controla con:

- `JOB_TTL_HOURS`: tiempo desde que termina el render hasta su eliminación.
- `PURGE_INTERVAL_MINUTES`: frecuencia de revisión.

## 5. Recursos

Cada render concurrente usa CPU y aproximadamente varios buffers RGB del tamaño de la imagen. La versión 0.3 mantiene además SAM 2 cargado en memoria. Empieza con `MAX_CONCURRENT_RENDERS=1` o `2`, `FFMPEG_THREADS=2`, `OPENCV_THREADS=1` y `SAM2_CPU_THREADS=4`. Incrementa la concurrencia solamente después de observar memoria y CPU con imágenes reales.

La segmentación se serializa: nunca habrá dos inferencias SAM 2 simultáneas dentro del contenedor. Los renders que ya terminaron de segmentar sí pueden codificarse concurrentemente.

No aumentes el número de workers de Uvicorn: la cola vive dentro del proceso. La concurrencia de render se controla exclusivamente con `MAX_CONCURRENT_RENDERS`.

## 6. Verificación

Después del despliegue:

```bash
curl https://reveal-api.example.com/healthz
```

Debe devolver algo similar a:

```json
{
  "status": "ok",
  "concurrent_limit": 2,
  "active_jobs": 0,
  "queued_jobs": 0
}
```

Después abre `/docs` para ejecutar el primer render desde Swagger UI.

## Actualizar a 0.4.2

1. Actualiza el código del repositorio a la versión 0.4.2 y realiza `push`.
2. En Coolify pulsa **Redeploy** si el webhook no lo hace automáticamente.
3. No borres el volumen `reveal-data`; no requiere migración.
4. Confirma que `/docs` muestre `detail_mode`, `detail_feather`, `segmentation_mode`, `region_order`, `max_regions` y `min_region_area`.

La primera compilación será considerablemente mayor que en 0.2 porque descarga PyTorch CPU, el código de SAM 2 fijado a una revisión concreta y un checkpoint verificado de aproximadamente 156 MB. El servidor de build necesita acceso saliente a `download.pytorch.org`, `github.com` y `huggingface.co`.

El primer render después de iniciar el contenedor carga el modelo y puede tardar más. Las siguientes solicitudes reutilizan la misma instancia.

Los clientes que necesiten el barrido de detalle anterior pueden enviar `detail_mode=legacy`. `segmentation_mode=none` mantiene el pincel global sin SAM 2. No hay migración del volumen ni cambio en los endpoints existentes.

Para obtener un solo frente visible, usa `fill_brushes=1` y `fill_overlap=0`. En 0.4.2 ese único cabezal pinta secuencialmente cada región SAM 2 y finalmente el residual.
