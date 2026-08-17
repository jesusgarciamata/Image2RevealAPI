# Historial de cambios

## 0.3.0

- Segmentación automática distinta para cada imagen mediante SAM 2.1 Hiera Tiny.
- Limpieza, deduplicación, agrupación y partición no solapada de máscaras.
- Revelado secuencial recortado por la forma de cada región.
- Pincel adaptativo basado en orientación principal, textura y distancia al contorno.
- Órdenes `saliency`, `reading-order`, `center-first`, `large-first`, `small-first` y `random`.
- Parámetros `segmentation_mode`, `region_order`, `max_regions` y `min_region_area`.
- Endpoint `GET /v1/renders/{id}/regions` con previsualización numerada.
- Segmentación serializada para compartir una sola instancia del modelo entre workers.
- Fallback explícito `segmentation_mode=none` compatible con la versión 0.2.

## 0.2.0

- `reading-order` es ahora la dirección predeterminada: inicia arriba a la izquierda y progresa hacia la derecha y hacia abajo.
- Nuevo modo `random-origins` para iniciar la capa de detalles desde varios puntos reproducibles mediante `seed`.
- El relleno dejó de ordenar discos independientes. Ahora utiliza entre uno y cinco pinceles simultáneos que recorren trayectorias continuas y dejan una estela acumulativa.
- Nuevo parámetro de API `fill_brushes`, con valor predeterminado `3`.
- Las trayectorias incorporan ondulación suave y variación gradual de radio para evitar una apariencia cuadriculada.
- Se mantienen las direcciones anteriores para conservar compatibilidad con clientes existentes.

## 0.1.0

- Primera versión del MVP.
