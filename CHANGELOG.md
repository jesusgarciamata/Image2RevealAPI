# Historial de cambios

## 0.4.1

- El residual no identificado por SAM 2 abandona la trayectoria serpenteante y visita aleatoriamente zonas pendientes cercanas.
- Las transiciones entre destinos se interpolan densamente para que el movimiento permanezca continuo incluso cuando cambia de dirección.
- Los sellos residuales se solapan con un retardo radial mínimo, eliminando la apariencia de discos o chunks sucesivos.
- El recorrido continúa siendo completamente reproducible mediante `seed`.

## 0.4.0

- Nuevo modo predeterminado `detail_mode=regions`: entinta progresivamente cada forma detectada por SAM 2 antes de pasar a la siguiente.
- La primera fase reutiliza `region_order` y las máscaras de la fase de color, pero genera un movimiento independiente para evitar repetir exactamente el mismo recorrido.
- Nuevo parámetro `detail_feather`, con una transición corta que elimina el aspecto de humo sin perder antialiasing.
- La máscara de detalle se endurece solo en modo `regions`; `detail_mode=legacy` conserva la animación anterior.
- La fase de color mantiene su algoritmo, sus parámetros y la secuencia pseudoaleatoria de la versión 0.3.1.

## 0.3.1

- Nuevo parámetro `detail_chroma` para desaturar únicamente la capa de detalle sin alterar el relleno ni la imagen final.
- Nuevo parámetro `detail_selectivity` para conservar solo los contornos y texturas de mayor intensidad.
- Los valores predeterminados mantienen el aspecto de la versión 0.3.0 para no cambiar renders existentes.

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
