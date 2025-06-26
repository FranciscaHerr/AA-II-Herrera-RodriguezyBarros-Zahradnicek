## Flappy Bird: Ingeniería de Características

### Estado crudo del entorno:

El entorno del juego Flappy Bird proporciona el siguiente estado en cada paso:

- `player_y`: Posición vertical (Y) del jugador (centro del pájaro).
- `player_vel`: Velocidad vertical actual del jugador.
- `next_pipe_dist_to_player`: Distancia horizontal entre el jugador y el siguiente tubo.
- `next_pipe_top_y`: Posición Y de la parte superior del siguiente tubo.
- `next_pipe_bottom_y`: Posición Y de la base del siguiente tubo.
- `next_next_pipe_dist_to_player`: Distancia horizontal al segundo tubo próximo.
- `next_next_pipe_top_y`: Posición Y de la parte superior del segundo tubo.
- `next_next_pipe_bottom_y`: Posición Y de la base del segundo tubo.

**Nota:** No todas estas variables son utilizadas actualmente en la discretización. El agente basado en Q-Learning solo extrae y procesa un subconjunto de estas características.

---

## Características del Estado Procesadas y Discretizadas:

Estado crudo → Tupla de 6 números enteros.

Se discretizan las siguientes características para simplificar la percepción del agente:

### 1. ¿Dónde está el jugador verticalmente en el escenario?

**Idea:** Determinar en qué zona vertical se encuentra el jugador.

**Simplificación:** El campo se divide en 10 zonas de igual altura.

**Importancia:** Permite al agente saber si está muy abajo, en el centro o muy arriba.

**Ejemplo:** Jugador en la zona 3 de altura → **3**

---

### 2. ¿Cómo se está moviendo el jugador ahora mismo?

**Idea:** Clasificar la velocidad vertical en 5 categorías:

- 0: Subiendo muy rápido.
- 1: Subiendo rápido.
- 2: Lento o quieto.
- 3: Bajando rápido.
- 4: Bajando muy rápido.

**Importancia:** Informa al agente sobre el movimiento actual, fundamental para planificar saltos.

**Ejemplo:** Jugador bajando rapido → **3**

---

### 3. ¿Dónde está el centro del hueco (gap) respecto al jugador?

**Idea:** Diferencia vertical entre el jugador y el centro del hueco, normalizada y discretizada en 10 zonas.

**Importancia:** Ayuda a alinearse con el centro del hueco para evitar choques.

**Ejemplo:** Centro del hueco alineado → **5**

---

### 4. ¿Qué tan cerca está la base del tubo respecto al jugador?

**Idea:** Diferencia vertical entre el jugador y la base del tubo, normalizada y discretizada en 10 zonas.

**Importancia:** Detecta proximidad de colisión por abajo.

**Ejemplo:** Base del tubo algo cerca por abajo → **6**

---

### 5. ¿Qué tan cerca está la parte superior del tubo respecto al jugador?

**Idea:** Diferencia vertical entre el jugador y la parte superior del tubo, normalizada y discretizada en 10 zonas.

**Importancia:** Detecta peligro de colisión por arriba.

**Ejemplo:** Parte superior del tubo algo cerca por arriba → **3**

---

### 6. ¿Qué tan lejos está horizontalmente el próximo tubo?

**Idea:** Distancia horizontal al siguiente tubo, normalizada y discretizada en 10 zonas.

**Importancia:** Permite anticipar los movimientos con base en la cercanía del obstáculo.

**Ejemplo:** Próximo tubo a media distancia → **5**

---

## Ejemplo completo de estado discreto:

(3, 3, 5, 6, 3, 5)

Este vector resume la percepción del agente y es lo que se utiliza para consultar la Q-Table y decidir si realizar un salto o no.

---

## Conclusiones sobre el rendimiento de los agentes

La **máxima recompensa promedio durante el entrenamiento** del Q-Agent fue de **23**. Además, la mayor cantidad de tuberías superadas durante el testeo fue de **3062**.

En términos generales, el agente maneja correctamente la mayoría de las situaciones del entorno, incluyendo aquellas donde debe realizar una **subida brusca**, es decir, cuando el hueco entre los tubos cambia repentinamente de una altura media-baja a una muy elevada. En esos casos, el agente suele alcanzar el hueco, pero lo hace con un margen extremadamente ajustado. Por ese motivo, una mínima variación en el entorno puede generar una situación en la que, aunque el agente reaccione a tiempo, físicamente ya no disponga del tiempo necesario para completar la maniobra y termina colisionando.

Por otro lado, se entrenó una red neuronal que aproxima la Q-Table, alcanzando un **accuracy de 0.995** sobre los datos de entrenamiento. Sin embargo, este agente no pudo ser evaluado de forma práctica debido a las limitaciones de hardware. La red neuronal requiere entre **40 y 50 milisegundos por acción**, lo que genera un aumento significativo en el tiempo de ejecución.

Para comparar adecuadamente el rendimiento de ambos agentes sería necesario realizar múltiples episodios completos, observando la **recompensa total**. En esos episodios, el Q-Agent obtiene valores variables, como 300, 500 o incluso más de 1000 tuberías superadas. Dado el tiempo de inferencia de la red, este tipo de evaluación se vuelve inviable en la práctica, ya que incluso probar un único episodio completo podría requerir un tiempo excesivo.

Hasta el momento, se ha dejado correr al agente basado en la red neuronal durante una cantidad prolongada de tiempo y no ha perdido, lo que indica que su rendimiento **aparenta ser similar al del Q-Agent**. Sin embargo, no se ha podido realizar una comparación formal con múltiples episodios para determinar con certeza si su rendimiento es exactamente el mismo o existe una pequeña diferencia. Las métricas de entrenamiento sugieren que debería tener un comportamiento prácticamente equivalente, pero la validación definitiva queda pendiente debido a las restricciones técnicas mencionadas.

También podría resultar interesante evaluar si una red neuronal de menor complejidad alcanza una métrica similar de accuracy. En este trabajo se priorizó obtener el mayor accuracy posible, sin considerar el costo computacional asociado. Sin embargo, si una arquitectura más simple logra un rendimiento comparable, se podría reducir significativamente el tiempo de inferencia y permitir así una comparación práctica y directa entre ambos agentes.


