# Guia para estudiantes: entrenar PequeLLM en `renna`

Este documento te lleva, paso a paso, desde "acabo de tener acceso a renna" hasta "mi entrenamiento esta corriendo en la GPU". Asume que sabes Python pero no necesariamente Docker. Cada comando esta disenado para que lo copies y pegues tal cual.

> **¿Nunca has usado contenedores (Docker/podman)?** Lee primero `docs/DOCKER.md`. Esta guia explica los conceptos basicos (que es una imagen, un contenedor, un volumen, que hace cada bandera de `podman run`) para que entiendas que pasa cuando corres `./run.sh` y puedas depurar cuando algo falle. Si ya conoces Docker, sigue leyendo aqui.

---

## 1. Antes de empezar

### Que es renna

`renna` es el servidor compartido del homelab. Esta detras de Tailscale (necesitas estar conectado a la VPN para alcanzarlo) y corre Fedora Kinoite. La GPU es una **AMD Strix Halo (Radeon 8060S)** con hasta **96 GB de memoria unificada** (la GPU comparte la RAM del sistema; no hay VRAM dedicada).

### Tu cuenta

Tu profesor te dio una cuenta del estilo `pequenllm-N` (donde N es tu numero). Vas a conectarte por SSH:

```bash
ssh pequenllm-N@renna
```

Tu directorio personal (`$HOME`, equivalente a `~`) esta cifrado con LUKS. Todo lo que pongas ahi es tuyo y no es accesible para otros estudiantes. Fuera de tu home no puedes escribir nada.

> **No tienes sudo.** No puedes instalar paquetes del sistema, agregarte a grupos, ni modificar nada fuera de tu home. Eso esta bien — todo lo que necesitas vive en tu home, dentro del contenedor que te enseniamos a usar abajo.

---

## 2. Setup la primera vez

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

1. `run.sh` ve que la imagen `pequellm:rocm` no existe y la construye (puede tomar unos minutos: descarga `rocm/pytorch:latest`, ~30 GB, mas las dependencias Python del repo).
2. Crea dos volumenes persistentes (`pequellm-data` y `pequellm-cache`) donde se guardaran tus datos y caches entre corridas.
3. Corre el smoke test de GPU para confirmar que el contenedor ve la Radeon 8060S.
4. Arranca el entrenamiento de `Embeddings/emb_gpt2.py`.

> **La primera corrida es lenta** (la descarga de la imagen domina). Las siguientes son inmediatas porque la imagen y los volumenes ya existen.

> **¿Y si no hay `train.bin` todavia?** Entonces el entrenamiento fallara con `FileNotFoundError`. Tienes dos opciones (seccion 4 abajo): (a) generar bins sinteticos para verificar que todo el pipeline funciona, o (b) correr la preparacion real desde CulturaX (lenta).

---

## 3. Como funciona `./run.sh`

`run.sh` es un wrapper delgado sobre `podman run`. Acepta subcomandos:

| Comando | Que hace |
|---|---|
| `./run.sh` | Smoke de GPU + entrenamiento (default). |
| `./run.sh build` | Solo construye la imagen `pequellm:rocm`. |
| `./run.sh smoke` | Solo verifica que la GPU se ve desde el contenedor. |
| `./run.sh synth` | Genera `train.bin`/`val.bin` sinteticos en el volumen (ruido aleatorio, sin red). |
| `./run.sh prepare-data` | Corre la preparacion real (CulturaX, tokenizador). Lento, requiere red. |
| `./run.sh train [args]` | Entrena, pasando argumentos extras a `emb_gpt2.py`. Ejemplo: `./run.sh train --max-iters 200 --batch-size 8`. |
| `./run.sh shell` | Abre una shell bash interactiva dentro del contenedor con todos los volumenes montados. Util para depurar. |
| `./run.sh help` | Imprime esta tabla. |

Todos los subcomandos auto-construyen la imagen y auto-crean los volumenes si no existen. No tienes que recordarlo.

---

## 4. Donde vive tu trabajo (importante)

Hay **tres lugares** donde se guardan archivos cuando corres el contenedor:

| Ubicacion | Que hay aqui | Sobrevive a... | Va al fork? |
|---|---|---|---|
| `~/PequeLLM/` (tu fork) | Codigo Python, `run.sh`, `Dockerfile`, scripts. | Reboots, rebuilds, `podman rm`. | **Si**. Aqui haces commits. |
| Volumen `pequellm-data` | `train.bin`, `val.bin`, `pequellm_pesado_checkpoint.pth`, `artifacts_gpt2/<run>/...` | Reboots, rebuilds, `podman rm`. | **No**. Datos y pesos no se versionan. |
| Volumen `pequellm-cache` | Cache de HuggingFace (`HF_HOME`). | Reboots, rebuilds, `podman rm`. | **No**. Cache, no codigo. |

### Como ve esto el contenedor

Dentro del contenedor:

```
/workspace/repo   <-  bind mount de tu ~/PequeLLM (lectura+escritura)
/workspace/data   <-  volumen pequellm-data
/workspace/cache  <-  volumen pequellm-cache (HF_HOME)
```

Si editas `Embeddings/emb_gpt2.py` en el host con vim/VS Code y vuelves a correr `./run.sh`, el contenedor ve tus cambios inmediatamente — **no hay que reconstruir la imagen** salvo que cambies el `Dockerfile` o las dependencias Python.

### Inspeccionar el volumen desde el host

```bash
# Donde vive fisicamente:
podman volume inspect pequellm-data --format '{{.Mountpoint}}'

# Que hay adentro:
ls -lh "$(podman volume inspect pequellm-data --format '{{.Mountpoint}}')/"
```

### Limpiar todo y empezar de cero

```bash
podman volume rm pequellm-data pequellm-cache       # borra datos y cache
podman rmi localhost/pequellm:rocm                  # borra la imagen del repo
# (la base rocm/pytorch:latest se queda; ocupa ~30 GB pero la reusas)
```

---

## 5. Como ve la GPU el contenedor

`run.sh` agrega estas banderas a `podman run`:

| Bandera | Para que sirve |
|---|---|
| `--device /dev/kfd` | Pasa el dispositivo de compute de AMD ROCm al contenedor. Sin esto, no hay GPU. |
| `--device /dev/dri` | Pasa los dispositivos de render (`renderD128`, `card0`). Tambien necesario. |
| `--group-add keep-groups` | Mantiene los grupos suplementarios de tu usuario host dentro del contenedor (por si en algun host estas en `render` o `video`). En renna los devices son world-readable, asi que esto es seguro de mas. |
| `--security-opt label=disable` | Le dice a podman que NO relabele los devices del host con SELinux (no podriamos relabelearlos aunque quisieramos). El contenedor sigue siendo no-privilegiado en todo lo demas. |
| `--shm-size 8g` | El DataLoader de PyTorch usa `/dev/shm` para pasar datos entre workers. El default (64 MB) es muy poco. |

El smoke test (`scripts/check_gpu.py`) corre antes del entrenamiento. Si falla, te imprime un bloque de triage que te dice exactamente que dispositivo le falto y donde leer mas (esta seccion).

---

## 6. Por que NO usamos Distrobox para correr esto

Es probable que tu cuenta tenga ademas un **contenedor Distrobox** (algo como `pequenllm-N-env`) preconfigurado para trabajo de desarrollo general. **No entres a Distrobox antes de correr `./run.sh`.** Razones:

- **No anidamos contenedores.** `./run.sh` arranca un contenedor de podman. Hacerlo desde adentro de Distrobox seria un contenedor dentro de un contenedor — pasar `/dev/kfd` correctamente se vuelve fragil y depende de la configuracion exacta de Distrobox.
- **Distrobox es para dev general** (linters, editores, paquetes Python que no quieres en el host). El contenedor ML de PequeLLM es **separado** y autonomo.
- **El flujo correcto** es: SSH a renna -> shell normal -> `cd ~/PequeLLM` -> `./run.sh`. Sin Distrobox de por medio.

Si por accidente entraste a Distrobox, sal con `exit` antes de correr `./run.sh`.

---

## 7. Errores comunes y como arreglarlos

### 7.1 "GPU no visible" / `torch.cuda.is_available() == False`

`./run.sh smoke` te imprimira algo asi:

```
[GPU CHECK] FAILED: torch.cuda.is_available() == False
Dispositivos visibles dentro del contenedor:
  [MISSING] /dev/kfd
  [OK] /dev/dri
  ...
```

Causas posibles:

- **Faltan los `--device` en `podman run`**. Si modificaste `run.sh`, restaura las banderas de la seccion 5.
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

### 7.2 Out-of-memory (OOM) en la GPU

Strix Halo usa **memoria unificada**: la GPU usa la misma RAM que el sistema. La maquina tiene 96 GB *en total*, no 96 GB de VRAM ademas de RAM. Si abres muchos procesos pesados en otras sesiones SSH, le quitas memoria a tu entrenamiento.

Sintomas: `RuntimeError: HIP out of memory` o muerte silenciosa del proceso.

Soluciones:

```bash
# Bajar batch_size (default: 16)
./run.sh train --batch-size 8

# Bajar block_size (contexto, default: 128)
./run.sh train --block-size 64

# Forzar fp16 (default es bf16 en auto, mismo costo en memoria pero a veces ayuda)
./run.sh train --precision fp16

# Modelo mas chico (default: n_embd 768, n_head 24, n_layer 4)
./run.sh train --n-embd 384 --n-head 12 --n-layer 4
```

Tambien puedes monitorear la GPU desde otra sesion SSH con `rocm-smi` (si esta instalado en el host) o `radeontop`.

### 7.3 Permisos del volumen / "Permission denied" escribiendo en `/workspace/data`

Esto pasa cuando el volumen fue creado por otro usuario o con uid distinto. Solucion:

```bash
podman volume rm pequellm-data
./run.sh synth   # o ./run.sh prepare-data segun corresponda
```

Recreas el volumen como tu usuario y queda con los permisos correctos.

### 7.4 "cannot execute binary file" o errores en el ENTRYPOINT

Si modificaste el `Dockerfile`, asegurate de **no** poner `ENTRYPOINT ["bash"]` solo. La imagen usa `CMD ["bash", "-lc", "..."]` sin ENTRYPOINT precisamente para que `podman run image python script.py` ejecute `python script.py` directamente.

### 7.5 La descarga de `rocm/pytorch:latest` es lentisima

Es ~30 GB. Si el ancho de banda esta saturado, espera. Para corridas posteriores la imagen ya esta cacheada en `~/.local/share/containers/` y es instantanea.

---

## 8. Como subir cambios (fork → branch → push → PR)

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

> **Nunca hagas push directo a upstream.** Aunque tuvieras permisos, el flujo es PR → revision → merge.

---

## 9. Donde mirar si algo no esta cubierto aqui

- `docs/DOCKER.md`: que es un contenedor, una imagen, un volumen. Que hace cada bandera de `podman run`. Comandos basicos para inspeccionar y limpiar.
- `Embeddings/README.md`: progresion didactica de los scripts de modelado.
- `Embeddings/GPT2_DIAGNOSTICS.md`: como interpretar los artefactos de entrenamiento.
- Pregunta a tu profesor o abre un issue en el repo upstream.

Buena suerte y happy training.
