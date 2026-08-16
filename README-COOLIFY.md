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

Cada render concurrente usa CPU y aproximadamente varios buffers RGB del tamaño de la imagen. Empieza con `MAX_CONCURRENT_RENDERS=2`, `FFMPEG_THREADS=2` y `OPENCV_THREADS=1`. Si la memoria o CPU permanecen holgadas durante varios trabajos 1080p, incrementa la concurrencia a `3` o `4` y vuelve a observar el servidor.

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
