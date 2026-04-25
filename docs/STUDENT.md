# Guia para estudiantes: entrenar PequeLLM en `renna`

Este documento te lleva, paso a paso, desde "acabo de tener acceso a renna" hasta "mi entrenamiento esta corriendo en la GPU y mis checkpoints estan a salvo". Asume que sabes Python pero no necesariamente Docker. Cada comando esta disenado para que lo copies y pegues tal cual.

> **¿Nunca has usado contenedores (Docker/podman)?** Lee primero `docs/DOCKER.md`. Esa guia explica los conceptos basicos (que es una imagen, un contenedor, un volumen, que hace cada bandera de `podman run`) para que entiendas que pasa cuando corres `./run.sh` y puedas depurar cuando algo falle. Si ya conoces Docker, puedes seguir leyendo aqui sin escalar.

**Indice**

1. [Antes de empezar](#1-antes-de-empezar)
2. [Setup la primera vez (3 comandos)](#2-setup-la-primera-vez-3-comandos)
3. [Validar que todo funciona (5 minutos, sin red)](#3-validar-que-todo-funciona-5-minutos-sin-red)
4. [Como funciona por dentro](#4-como-funciona-por-dentro)
5. [Como ejecutar scripts (recetas concretas)](#5-como-ejecutar-scripts-recetas-concretas)
6. [Como NO perder tu trabajo: el volumen montado](#6-como-no-perder-tu-trabajo-el-volumen-montado)
7. [Como ve la GPU el contenedor (banderas explicadas)](#7-como-ve-la-gpu-el-contenedor-banderas-explicadas)
8. [Por que NO usamos Distrobox](#8-por-que-no-usamos-distrobox)
9. [Errores comunes y como arreglarlos](#9-errores-comunes-y-como-arreglarlos)
10. [Como subir cambios (fork -> branch -> PR)](#10-como-subir-cambios-fork---branch---pr)
11. [Donde leer mas](#11-donde-leer-mas)

---

## 1. Antes de empezar

### Que es renna

`renna` es el servidor compartido del homelab. Esta detras de Tailscale (necesitas estar conectado a la VPN para alcanzarlo) y corre Fedora Kinoite. La GPU es una **AMD Strix Halo (Radeon 8060S)** con hasta **96 GB de memoria unificada** (la GPU comparte la RAM del sistema; no hay VRAM dedicada).

### Tu cuenta

Tu profesor te dio una cuenta del estilo `pequenllm-N` (donde N es tu numero). Te conectas por SSH:

```bash
ssh pequenllm-N@renna
```

Tu directorio personal (`$HOME`, equivalente a `~`) esta cifrado con LUKS. Todo lo que pongas ahi es tuyo y no es accesible para otros estudiantes. Fuera de tu home no puedes escribir nada.

> **No tienes sudo.** No puedes instalar paquetes del sistema, agregarte a grupos, ni modificar nada fuera de tu home. Eso esta bien — todo lo que necesitas vive en tu home, dentro del contenedor que te enseniamos a usar abajo.

---

## 2. Setup la primera vez (3 comandos)

### 2.1 Hacer fork del repo en GitHub

Ve a `https://github.com/uumami/PequeLLM` y dale clic a **Fork** (esquina superior derecha). Esto crea una copia bajo tu cuenta de GitHub: `https://github.com/<tu-usuario>/PequeLLM`.

### 2.2 Clonar tu fork dentro de tu home en renna

```bash
ssh pequenllm-N@renna
cd ~
git clone https://github.com/<tu-usuario>/PequeLLM.git
cd PequeLLM
```

(Si prefieres SSH para Git en lugar de HTTPS, configura tu llave SSH en GitHub primero.)

### 2.3 Construir y arrancar el contenedor

```bash
./run.sh
```

Eso es todo. La primera vez:

1. `run.sh` ve que la imagen `pequellm:rocm` no existe y la construye (puede tomar varios minutos: descarga `rocm/pytorch:latest`, ~30 GB, mas las dependencias Python del repo).
2. Crea dos volumenes persistentes (`pequellm-data` y `pequellm-cache`) donde se guardaran tus datos y caches **entre corridas**.
3. Corre el smoke test de GPU para confirmar que el contenedor ve la Radeon 8060S.
4. Arranca el entrenamiento de `Embeddings/emb_gpt2.py`.

> **La primera corrida es lenta** (la descarga de la imagen domina). Las siguientes son inmediatas porque la imagen y los volumenes ya existen.

> **¿Y si no hay `train.bin` todavia?** Entonces el entrenamiento fallara con `FileNotFoundError`. Lee la siguiente seccion: con `./run.sh synth` puedes generar bins sinteticos en menos de un minuto y validar todo el pipeline sin red.

---

## 3. Validar que todo funciona (5 minutos, sin red)

Antes de invertir tiempo en datos reales (CulturaX se baja en horas), confirma que tu cuenta + el contenedor + la GPU funcionan juntos:

```bash
cd ~/PequeLLM

# (a) ¿El contenedor ve la GPU?
./run.sh smoke
#  -> deberias ver:
#     [GPU CHECK] device 0 name = Radeon 8060S Graphics
#     [GPU CHECK] matmul 256x256 OK
#     [GPU CHECK] OK

# (b) Generar datos sinteticos en el volumen (uint16 aleatorios, ~9 MB total).
./run.sh synth
#  -> deberias ver dos archivos creados en /workspace/data dentro del volumen.

# (c) Entrenar 50 iteraciones contra esos datos sinteticos.
./run.sh train --max-iters 50 --eval-interval 25 --save-interval 25 --skip-presentation
#  -> deberias ver al menos:
#     [INFO] device = cuda
#     [iter 000000] train_loss=11.27 ...
#     [INFO] checkpoint saved at iter=25 ...
```

Si esos tres comandos corren limpios, **tu entorno esta sano**. El loss de ~11.27 es normal con datos sinteticos (es `ln(65536)`); el modelo no puede aprender ruido uniforme. El punto del paso (c) es solo probar que el bucle completo (GPU -> training loop -> checkpoint -> volumen) funciona.

Una vez validado, ya puedes pasar a entrenar con datos reales (`./run.sh prepare-data` y luego `./run.sh`), o a iterar sobre los scripts del repo.

---

## 4. Como funciona por dentro

Cuando corres `./run.sh` pasan tres cosas en cadena:

```
Tu shell de SSH en renna
        |
        v
./run.sh   (un script bash que vive en tu fork)
        |
        v
podman run   (lanza un contenedor con la imagen pequellm:rocm)
        |
        v
Adentro del contenedor:
  - Python 3.12 + PyTorch compilado contra ROCm
  - El device /dev/kfd (GPU AMD) montado desde el host
  - Tu repo ~/PequeLLM montado en /workspace/repo (bind mount)
  - El volumen pequellm-data montado en /workspace/data
        |
        v
python /workspace/repo/Embeddings/emb_gpt2.py [args]
```

Los puntos clave:

- **El codigo que ejecuta el contenedor es el del host.** Cuando editas `Embeddings/emb_gpt2.py` en tu maquina (vim, VS Code remoto, lo que sea), el contenedor lo ve cambiado al instante. **No necesitas reconstruir la imagen** salvo que cambies el `Dockerfile` o las dependencias Python.
- **Los datos NO viven dentro del contenedor.** Viven en el volumen `pequellm-data`, que existe antes y despues de cada corrida. Por eso tu checkpoint del lunes sigue ahi el martes aunque el contenedor del lunes ya no exista.
- **El contenedor se autodestruye al terminar.** `run.sh` usa `--rm`, asi que despues de que el script de Python sale, el contenedor desaparece. Lo persistente es el codigo (en tu repo) y los datos (en el volumen).

Esto significa que **nunca pierdes nada porque "se borro el contenedor"** — siempre que pongas tus archivos en el volumen o en el repo, estaran ahi la proxima vez.

> Para una explicacion mas profunda de imagenes/contenedores/volumenes y banderas, lee `docs/DOCKER.md`.

---

## 5. Como ejecutar scripts (recetas concretas)

`run.sh` acepta subcomandos. Tabla rapida:

| Comando | Que hace |
|---|---|
| `./run.sh` | Smoke de GPU + entrenamiento default. |
| `./run.sh build` | Solo construye la imagen `pequellm:rocm`. |
| `./run.sh smoke` | Solo verifica que la GPU se ve desde el contenedor. |
| `./run.sh synth` | Genera `train.bin`/`val.bin` sinteticos en el volumen (sin red). |
| `./run.sh prepare-data` | Prepara datos reales desde CulturaX (lento, requiere red). |
| `./run.sh train [args]` | Entrena, pasando argumentos extras a `emb_gpt2.py`. |
| `./run.sh shell` | Bash interactivo dentro del contenedor con todos los volumenes montados. |
| `./run.sh help` | Imprime esta tabla. |

Todos los subcomandos auto-construyen la imagen y auto-crean los volumenes si no existen. No tienes que recordarlo.

### 5.1 Recetas comunes

```bash
# Entrenamiento default (smoke + train con la config por defecto)
./run.sh

# Entrenamiento corto para iterar rapido
./run.sh train --max-iters 200 --eval-interval 50 --save-interval 50

# Entrenamiento con menos memoria (si te dio OOM)
./run.sh train --batch-size 4 --block-size 64

# Entrenamiento con un modelo mas chico
./run.sh train --n-embd 384 --n-head 12 --n-layer 4 --max-iters 1000

# Forzar empezar desde cero (ignora el checkpoint del volumen)
./run.sh train --no-resume --max-iters 500

# Brincar la generacion automatica del PDF al final
./run.sh train --max-iters 500 --skip-presentation
```

Cualquier flag que acepte `Embeddings/emb_gpt2.py` lo puedes pasar despues de `train`. Para ver la lista completa:

```bash
./run.sh shell
python /workspace/repo/Embeddings/emb_gpt2.py --help
exit
```

### 5.2 Correr OTROS scripts del repo (no solo emb_gpt2)

`./run.sh train` esta cableado a `emb_gpt2.py`. Para correr otra cosa, usa `shell` y ejecuta a mano:

```bash
./run.sh shell
# ahora estas adentro del contenedor; tu repo esta en /workspace/repo
cd /workspace/repo

# Ejemplo 1: tutorial paso 1 (forward pass simple)
python Embeddings/emb_ingenuo.py

# Ejemplo 2: tutorial con bucle de entrenamiento basico
python Embeddings/emb_bucle.py

# Ejemplo 3: el script de GPT-2 estilo Karpathy
python GPT-2/train_gpt2.py

# Ejemplo 4: tests del tokenizador
python -m unittest discover -s Tokenizador/tests

exit
```

> **Importante:** los scripts en `Embeddings/emb_ingenuo.py`, `emb_attention.py`, `emb_multi-attention.py` esperan `train.bin` en la **raiz del repo** (no en el volumen). Si quieres correrlos, ya sea (a) hazle `cp` desde el volumen al repo, o (b) edita el script para apuntar al `/workspace/data/train.bin`. Los scripts mas serios (`emb_bucle.py`, `emb_bb.py`, `emb_gpt2.py`) ya soportan paths via CLI o variables.

### 5.3 Inspeccionar lo que produjo una corrida

Las corridas de `emb_gpt2.py` escriben artefactos en el volumen, en `/workspace/data/artifacts_gpt2/<run_name>/`. Para verlos:

```bash
# Listar corridas
podman volume inspect pequellm-data --format '{{.Mountpoint}}' | xargs -I{} ls -lh {}/artifacts_gpt2/

# Ver metricas de una corrida especifica
RUN_DIR="$(podman volume inspect pequellm-data --format '{{.Mountpoint}}')/artifacts_gpt2/<run_name>"
ls -lh "$RUN_DIR"
cat "$RUN_DIR/train_metrics.csv" | head
cat "$RUN_DIR/parameter_guide.md"
```

O si prefieres desde adentro del contenedor:

```bash
./run.sh shell
ls -lh /workspace/data/artifacts_gpt2/
exit
```

---

## 6. Como NO perder tu trabajo: el volumen montado

Esta es **la seccion mas importante de la guia**. Lee con calma.

### 6.1 Los tres lugares donde puede vivir un archivo

| Ubicacion | Que va aqui | Sobrevive a | Va a tu fork de GitHub? |
|---|---|---|---|
| `~/PequeLLM/` (tu fork local) | Codigo `.py`, `run.sh`, `Dockerfile`, scripts. | Reboots, rebuilds, `podman rm`. | **Si** (lo que hagas commit). |
| Volumen `pequellm-data` | `train.bin`, `val.bin`, checkpoints `*.pth`, `artifacts_gpt2/<run>/...`. | Reboots, rebuilds, `podman rm`. | **No.** Datos y pesos no se versionan. |
| Volumen `pequellm-cache` | Cache de HuggingFace (`HF_HOME`). | Reboots, rebuilds, `podman rm`. | **No.** Es cache, no codigo. |
| FS efimero del contenedor (cualquier otra ruta) | Cualquier cosa que escribas en `/tmp`, `/opt`, `/home/...` adentro. | **NO.** Se borra cuando el contenedor termina. | **No.** |

### 6.2 La regla de oro

> **Codigo va al fork. Datos y checkpoints van al volumen. Nunca pongas algo importante en otro lado.**

Si te aseguras de que tu trabajo cae en uno de los dos primeros, **es imposible perderlo accidentalmente**. La unica forma de perder algo es ejecutando manualmente un comando que lo borra (`git reset --hard`, `podman volume rm`, `rm -rf`, etc.).

### 6.3 Como ver el volumen montado en el contenedor

Cuando `run.sh` lanza el contenedor, hace este montaje:

```
HOST                                       CONTENEDOR
~/PequeLLM/                  <----->       /workspace/repo     (bind mount)
volumen pequellm-data        <----->       /workspace/data
volumen pequellm-cache       <----->       /workspace/cache
```

Por eso `Embeddings/emb_gpt2.py` recibe via flags:

```
--train-bin       /workspace/data/train.bin
--val-bin         /workspace/data/val.bin
--checkpoint-path /workspace/data/pequellm_pesado_checkpoint.pth
--output-root     /workspace/data/artifacts_gpt2
```

Todo lo que el script escribe en `/workspace/data/...` va al volumen, que existe **fuera** del contenedor. Cuando termina la corrida, el contenedor se borra (`--rm`), pero los archivos siguen ahi.

### 6.4 Verificar que tu checkpoint se guardo

Despues de un entrenamiento:

```bash
# Donde vive fisicamente el volumen en el host:
podman volume inspect pequellm-data --format '{{.Mountpoint}}'
#  -> /var/home/pequenllm-N/.local/share/containers/storage/volumes/pequellm-data/_data

# Inspeccionar el contenido:
ls -lh "$(podman volume inspect pequellm-data --format '{{.Mountpoint}}')/"
#  -> deberias ver: train.bin, val.bin, pequellm_pesado_checkpoint.pth, artifacts_gpt2/

# Tamanio de tu checkpoint:
ls -lh "$(podman volume inspect pequellm-data --format '{{.Mountpoint}}')/pequellm_pesado_checkpoint.pth"
```

### 6.5 Resumir entrenamiento desde un checkpoint

`emb_gpt2.py` resume **automaticamente** del checkpoint si existe (`cfg.resume = True` por default). O sea:

```bash
# Lunes: entrenas 1000 iteraciones.
./run.sh train --max-iters 1000

# Martes: continuas hasta 2000 iteraciones (sin perder el progreso).
./run.sh train --max-iters 2000
#  -> veras: [INFO] start_iter = 999  (continua donde quedaste)
```

Si quieres **borrar el progreso** y empezar de cero:

```bash
./run.sh train --no-resume --max-iters 1000
```

(Esto sobreescribe el checkpoint pero NO borra los CSVs/PNGs anteriores en `artifacts_gpt2/`.)

### 6.6 Respaldar tu volumen (recomendado antes de experimentos riesgosos)

Si vas a hacer algo que podria corromper tu checkpoint (por ejemplo, modificar el modelo y reanudar), respalda primero:

```bash
# Copia el checkpoint a tu home (fuera del volumen, fuera del repo)
mkdir -p ~/backups
cp "$(podman volume inspect pequellm-data --format '{{.Mountpoint}}')/pequellm_pesado_checkpoint.pth" \
   ~/backups/checkpoint-$(date +%Y%m%d-%H%M).pth

# Para restaurar:
cp ~/backups/checkpoint-20260424-1530.pth \
   "$(podman volume inspect pequellm-data --format '{{.Mountpoint}}')/pequellm_pesado_checkpoint.pth"
```

### 6.7 Cosas que SI borran tus datos del volumen

Cuidado con estas:

```bash
podman volume rm pequellm-data       # BORRA todo el volumen
podman volume prune                  # borra TODOS los volumenes no usados
podman system prune -a --volumes     # borra TODO (imagenes + volumenes + contenedores)
```

Si dudas, **inspecciona antes de borrar**:

```bash
podman volume ls
podman volume inspect pequellm-data
```

### 6.8 Empezar de cero limpio

Cuando quieres reiniciar todo:

```bash
podman volume rm pequellm-data pequellm-cache    # borra TUS datos (cuidado)
podman rmi localhost/pequellm:rocm                # borra la imagen del repo
# (la base rocm/pytorch:latest se queda; ocupa ~30 GB pero la reusas)

cd ~/PequeLLM
./run.sh                                          # recrea todo
```

---

## 7. Como ve la GPU el contenedor (banderas explicadas)

`run.sh` agrega estas banderas a `podman run`:

| Bandera | Para que sirve |
|---|---|
| `--device /dev/kfd` | Pasa el dispositivo de compute de AMD ROCm al contenedor. **Sin esto, no hay GPU.** |
| `--device /dev/dri` | Pasa los dispositivos de render (`renderD128`, `card0`). Tambien necesario. |
| `--group-add keep-groups` | Mantiene los grupos suplementarios de tu usuario host dentro del contenedor (por si en algun host estas en `render` o `video`). En renna los devices son world-readable, asi que esto es seguro de mas. |
| `--security-opt label=disable` | Le dice a podman que NO relabele los devices del host con SELinux. El contenedor sigue siendo no-privilegiado en todo lo demas. |
| `--shm-size 8g` | El DataLoader de PyTorch usa `/dev/shm` para pasar datos entre workers. El default (64 MB) es muy poco. |

El smoke test (`scripts/check_gpu.py`) corre antes del entrenamiento. Si falla, te imprime un bloque de triage que te dice exactamente que dispositivo le falto y donde leer mas (esta seccion).

---

## 8. Por que NO usamos Distrobox

Es probable que tu cuenta tenga ademas un **contenedor Distrobox** (algo como `pequenllm-N-env`) preconfigurado para trabajo de desarrollo general. **No entres a Distrobox antes de correr `./run.sh`.** Razones:

- **No anidamos contenedores.** `./run.sh` arranca un contenedor de podman. Hacerlo desde adentro de Distrobox seria un contenedor dentro de un contenedor — pasar `/dev/kfd` correctamente se vuelve fragil y depende de la configuracion exacta de Distrobox.
- **Distrobox es para dev general** (linters, editores, paquetes Python que no quieres en el host). El contenedor ML de PequeLLM es **separado** y autonomo.
- **El flujo correcto** es: SSH a renna -> shell normal -> `cd ~/PequeLLM` -> `./run.sh`. Sin Distrobox de por medio.

Si por accidente entraste a Distrobox, sal con `exit` antes de correr `./run.sh`.

---

## 9. Errores comunes y como arreglarlos

### 9.1 "GPU no visible" / `torch.cuda.is_available() == False`

`./run.sh smoke` te imprimira algo asi:

```
[GPU CHECK] FAILED: torch.cuda.is_available() == False
Dispositivos visibles dentro del contenedor:
  [MISSING] /dev/kfd
  [OK] /dev/dri
  ...
```

Causas posibles:

- **Faltan los `--device` en `podman run`.** Si modificaste `run.sh`, restaura las banderas de la seccion 7.
- **`/dev/kfd` no es accesible para tu usuario** en el host. En renna deberia ser `crw-rw-rw-` (todo el mundo). Si tu admin lo cambio a `crw-rw---- root render` y no estas en `render`, hay dos opciones (admin only):

  ```bash
  # Opcion A (recomendada en homelab): setfacl, una sola vez por usuario
  sudo setfacl -m u:pequenllm-N:rw /dev/kfd
  sudo setfacl -m u:pequenllm-N:rw /dev/dri/renderD128

  # Opcion B (afecta el host): agregar el usuario al grupo render
  sudo usermod -aG render pequenllm-N
  # (requiere logout y login de nuevo)
  ```

  **No corras esto tu mismo.** Pidele al admin que lo haga (probablemente ya lo hizo, por eso `/dev/kfd` esta en `0666`).

### 9.2 Out-of-memory (OOM) en la GPU

Strix Halo usa **memoria unificada**: la GPU usa la misma RAM que el sistema. La maquina tiene 96 GB *en total*, no 96 GB de VRAM ademas de RAM. Si abres muchos procesos pesados en otras sesiones SSH, le quitas memoria a tu entrenamiento.

Sintomas: `RuntimeError: HIP out of memory` o muerte silenciosa del proceso.

Soluciones:

```bash
# Bajar batch_size (default: 16)
./run.sh train --batch-size 8

# Bajar block_size (contexto, default: 128)
./run.sh train --block-size 64

# Forzar fp16 (default es bf16 en auto)
./run.sh train --precision fp16

# Modelo mas chico (default: n_embd 768, n_head 24, n_layer 4)
./run.sh train --n-embd 384 --n-head 12 --n-layer 4
```

### 9.3 Permisos del volumen / "Permission denied" escribiendo en `/workspace/data`

Esto pasa cuando el volumen fue creado por otro usuario o con uid distinto. Solucion (cuidado: BORRA el volumen):

```bash
podman volume rm pequellm-data
./run.sh synth   # o ./run.sh prepare-data segun corresponda
```

### 9.4 El entrenamiento empieza desde iter 0 cuando deberia resumir

Verifica que el checkpoint este en el volumen:

```bash
ls -lh "$(podman volume inspect pequellm-data --format '{{.Mountpoint}}')/pequellm_pesado_checkpoint.pth"
```

- Si **no existe**: tu corrida anterior no llego a guardar (default: cada 500 iter). Entrena con `--save-interval 25` para ver checkpoints mas seguido.
- Si **existe**: revisa que no estes pasando `--no-resume`. La default es resumir.

### 9.5 "cannot execute binary file" o errores raros en el ENTRYPOINT

Si modificaste el `Dockerfile`, asegurate de **no** poner `ENTRYPOINT ["bash"]` solo. La imagen usa `CMD ["bash", "-lc", "..."]` sin ENTRYPOINT precisamente para que `podman run image python script.py` ejecute `python script.py` directamente.

### 9.6 La descarga de `rocm/pytorch:latest` es lentisima

Es ~30 GB. Si el ancho de banda esta saturado, espera. Para corridas posteriores la imagen ya esta cacheada en `~/.local/share/containers/` y es instantanea.

---

## 10. Como subir cambios (fork -> branch -> PR)

Tu trabajo en codigo va a tu fork, no al repo upstream directamente.

```bash
# Asegurate que tu main local esta al dia con upstream
cd ~/PequeLLM
git remote add upstream https://github.com/uumami/PequeLLM.git   # solo la primera vez
git fetch upstream
git checkout main
git merge upstream/main

# Crea una branch para tu cambio
git checkout -b mi-cambio-XYZ

# ... edita archivos, corre `./run.sh train` para validar ...

# Commit
git add archivo1.py archivo2.py
git commit -m "Descripcion clara de tu cambio"

# Push a TU fork (no a upstream)
git push origin mi-cambio-XYZ
```

Despues, en GitHub, abres un Pull Request desde `<tu-usuario>:mi-cambio-XYZ` hacia `uumami:main`. Tu profesor lo revisara.

> **Nunca hagas push directo a upstream.** Aunque tuvieras permisos, el flujo es PR -> revision -> merge.

---

## 11. Donde leer mas

- `docs/DOCKER.md`: que es un contenedor, una imagen, un volumen. Que hace cada bandera de `podman run`. Comandos basicos para inspeccionar y limpiar. Cheatsheet al final.
- `Embeddings/README.md`: progresion didactica de los scripts de modelado (de `emb_ingenuo.py` a `emb_gpt2.py`).
- `Embeddings/GPT2_DIAGNOSTICS.md`: como interpretar las metricas y artefactos de entrenamiento.
- Pregunta a tu profesor o abre un issue en el repo upstream.

Buena suerte y happy training.
