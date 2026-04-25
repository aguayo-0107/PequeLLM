# Que son los contenedores y como usarlos en PequeLLM

Esta es una guia desde cero para estudiantes que nunca han usado Docker o podman. El objetivo es que entiendas **que hace `./run.sh` por dentro** para que puedas depurar cuando algo no funciona, no solo correr el comando a ciegas.

> Si solo quieres "como entreno", lee `docs/STUDENT.md`. Esta guia explica el "como funciona".

---

## 1. ¿Que es un contenedor?

Imagina que quieres correr PyTorch con soporte de GPU AMD ROCm. Necesitas:

- Una version especifica de Python.
- PyTorch compilado contra ROCm (no la version normal de PyPI).
- Las librerias del sistema correctas (HIP, libhsa, librerias de Mesa).
- Numpy, transformers, datasets, todo en versiones compatibles.

Configurar todo esto en tu maquina te tomaria horas (o dias) y romperia otras cosas que ya tienes instaladas.

Un **contenedor** es como una **caja sellada** donde alguien ya hizo todo ese trabajo por ti. La caja contiene un mini-sistema-de-archivos completo con Python, PyTorch, librerias, y configuracion lista. Cuando lo "ejecutas", tu codigo corre dentro de esa caja, usando las herramientas de adentro, sin tocar tu maquina.

```
+----------------------------- TU SERVIDOR (renna) -----------------------------+
|                                                                              |
|  Sistema host: Fedora Kinoite, kernel 6.17, driver amdgpu                    |
|  Tu home: /var/home/pequenllm-N/  (cifrado, tuyo)                            |
|                                                                              |
|  +----------------- CONTENEDOR (la "caja") --------------------+             |
|  |                                                             |             |
|  |  Python 3.12, PyTorch+ROCm, transformers, etc.              |             |
|  |  /workspace/repo  -> tu codigo de PequeLLM (montado)        |             |
|  |  /workspace/data  -> tus train.bin, checkpoints (montado)   |             |
|  |                                                             |             |
|  |  [Tu script python corre AQUI ADENTRO]                      |             |
|  +-------------------------------------------------------------+             |
|                                                                              |
+------------------------------------------------------------------------------+
```

El contenedor ve **solo lo que le diste** (el codigo y los datos que montaste). No puede leer ni escribir el resto del sistema. Cuando termina, desaparece, pero los archivos en los volumenes/bind-mounts persisten.

### Analogia rapida

- **Imagen** = la **receta + ingredientes pre-empacados** para hacer una pizza. Es estatica, esta guardada en disco.
- **Contenedor** = una **pizza concreta** que cocinaste con esa receta. Es efimero; cuando te la comes (cierras el contenedor) se acaba.
- **Volumen** = el **plato** donde pones la pizza. Es persistente; existe antes y despues de la pizza.

Puedes hacer 100 pizzas (contenedores) de la misma receta (imagen). Y puedes guardar la salsa que sobre en el plato (volumen) para la proxima pizza.

---

## 2. ¿Docker o podman? ¿Cual usamos?

`podman` y `docker` son dos herramientas que hacen practicamente lo mismo: ejecutar contenedores. Aceptan los mismos comandos y los mismos `Dockerfile`.

**En `renna` usamos `podman`** por dos razones:

1. **No requiere `sudo`.** podman corre como tu usuario normal (rootless). docker tradicional requiere ser root o estar en el grupo `docker`, y tu no tienes permisos para nada de eso.
2. Ya viene instalado en Fedora.

Donde sea que veas "docker" en internet, mentalmente reemplaza por "podman" — los flags son iguales. Por ejemplo:

```bash
docker run hello-world      # equivalente a:
podman run hello-world
```

`run.sh` autodetecta cual esta disponible y usa podman primero.

---

## 3. Anatomia de un comando `podman run`

El corazon de `./run.sh` es un `podman run` con varias banderas. Veamos uno de verdad, descompuesto:

```bash
podman run \
  --rm \                                        # 1. borrar el contenedor al terminar
  --device /dev/kfd \                           # 2. dar acceso a GPU AMD
  --device /dev/dri \                           # 3. dar acceso a la salida de video / render
  --group-add keep-groups \                     # 4. mantener grupos del usuario host
  --security-opt label=disable \                # 5. desactivar relabel de SELinux en devices
  --shm-size 8g \                               # 6. memoria compartida grande para PyTorch
  -v pequellm-data:/workspace/data \            # 7. montar volumen persistente
  -v pequellm-cache:/workspace/cache \          # 8. montar cache de HuggingFace
  -v $(pwd):/workspace/repo \                   # 9. montar el repo del host
  -w /workspace/repo \                          # 10. directorio de trabajo dentro del contenedor
  -e HF_HOME=/workspace/cache \                 # 11. variable de entorno
  pequellm:rocm \                               # 12. la imagen a usar
  python /workspace/repo/scripts/check_gpu.py   # 13. el comando a ejecutar adentro
```

Linea por linea:

| # | Bandera | Que hace |
|---|---|---|
| 1 | `--rm` | Borra el contenedor cuando termina. Sin esto, los contenedores muertos se acumulan. |
| 2 | `--device /dev/kfd` | Pasa el dispositivo de compute de AMD ROCm al contenedor. **Sin esto NO hay GPU.** |
| 3 | `--device /dev/dri` | Pasa los dispositivos de render (`renderD128`, `card0`). Tambien necesario. |
| 4 | `--group-add keep-groups` | Conserva los grupos suplementarios del usuario host (por si en algun host estas en `render` o `video`). |
| 5 | `--security-opt label=disable` | Le dice a podman que NO relabele los devices del host con SELinux (no podriamos relabelearlos aunque quisieramos). |
| 6 | `--shm-size 8g` | El DataLoader de PyTorch usa `/dev/shm` para pasar datos entre workers. El default (64 MB) es muy poco. |
| 7-9 | `-v origen:destino` | Monta un volumen o un directorio del host en una ruta del contenedor. Ver seccion 4. |
| 10 | `-w /workspace/repo` | Cuando el contenedor arranca, su CWD sera ese path. |
| 11 | `-e VAR=valor` | Setea una variable de entorno dentro del contenedor. |
| 12 | `pequellm:rocm` | El nombre de la **imagen** de la cual nace este contenedor. |
| 13 | `python ...` | El **comando** que corre dentro del contenedor. Cuando este comando termina, el contenedor termina. |

> No tienes que memorizar esto. `run.sh` lo hace por ti. Pero si algo falla, sabes que mirar.

---

## 4. Volumenes y bind-mounts: donde se guardan tus archivos

Hay tres formas en que el contenedor "ve" archivos:

### 4.1 El sistema de archivos del contenedor (efimero)

Todo lo que escribas en el contenedor (fuera de los mounts) **desaparece** cuando el contenedor termina. Es como un disco RAM. Util para temporales, inutil para checkpoints.

### 4.2 Bind mount (`-v /ruta/host:/ruta/container`)

Un bind mount es un **espejo**: la ruta del contenedor refleja en tiempo real una ruta del host. Si escribes adentro, aparece afuera, y viceversa.

```bash
-v /var/home/pequenllm-1/PequeLLM:/workspace/repo
```

Eso significa: `/workspace/repo` adentro del contenedor **es exactamente** `/var/home/pequenllm-1/PequeLLM` afuera. Si editas `Embeddings/emb_gpt2.py` en el host con vim, el contenedor ya lo ve cambiado.

`run.sh` usa `-v "$REPO_DIR:/workspace/repo"` para esto.

### 4.3 Named volume (`-v nombre:/ruta/container`)

Un named volume es un **espacio gestionado por podman**. No corresponde a una ruta especifica del host (bueno, si, pero no te debe importar). Lo creas con un nombre, y podman lo recuerda.

```bash
-v pequellm-data:/workspace/data
```

Esto monta el volumen llamado `pequellm-data` en `/workspace/data` adentro. Si `pequellm-data` no existe, lo creas con `podman volume create pequellm-data`.

¿Donde vive fisicamente?

```bash
podman volume inspect pequellm-data --format '{{.Mountpoint}}'
# /var/home/pequenllm-N/.local/share/containers/storage/volumes/pequellm-data/_data
```

¿Que hay adentro? Podes mirar:

```bash
ls -lh "$(podman volume inspect pequellm-data --format '{{.Mountpoint}}')/"
```

### Resumen visual

```
HOST                            CONTENEDOR
~/PequeLLM/         <----->     /workspace/repo     (bind mount, efectivo en tiempo real)
volumen pequellm-data <--->     /workspace/data     (persistente, gestionado por podman)
volumen pequellm-cache <-->     /workspace/cache    (persistente, gestionado por podman)
(nada)                          /tmp, /opt, etc.    (efimeros, mueren con el contenedor)
```

---

## 5. Imagenes: ¿de donde salen?

Una **imagen** es la "receta empacada". Hay dos formas de obtener una:

### 5.1 Pull desde un registry

```bash
podman pull docker.io/rocm/pytorch:latest
```

Esto descarga la imagen de Docker Hub. La primera vez es lento (la nuestra es ~30 GB). Despues queda cacheada localmente.

Para listar imagenes que ya tienes:

```bash
podman images
```

### 5.2 Build desde un Dockerfile

Un `Dockerfile` es una receta escrita en texto. Por ejemplo, el nuestro:

```dockerfile
FROM docker.io/rocm/pytorch:latest      # parto de esa imagen base

WORKDIR /workspace/repo                  # mi CWD por defecto

ENV HF_HOME=/workspace/cache             # variable de entorno

RUN pip install numpy tokenizers ...     # instalo deps adicionales

CMD ["bash", "-lc", "python ..."]        # comando por defecto
```

Construir la imagen:

```bash
podman build -t pequellm:rocm .
```

`-t pequellm:rocm` le pone nombre y tag. El `.` indica que el Dockerfile esta en el directorio actual. Cada linea del Dockerfile crea una "capa" cacheada — si solo cambia la ultima, las anteriores se reusan.

`run.sh build` hace exactamente esto.

---

## 6. Comandos basicos de podman que vas a usar

### Listar

```bash
podman images                    # imagenes que tengo
podman ps                        # contenedores CORRIENDO ahora
podman ps -a                     # contenedores corriendo + parados
podman volume ls                 # volumenes que tengo
```

### Inspeccionar

```bash
podman image inspect pequellm:rocm
podman volume inspect pequellm-data
```

### Limpiar

```bash
# Borrar un contenedor que se quedo zombie
podman rm <id-del-contenedor>
podman rm -f <id>                # forzado

# Borrar todos los contenedores parados
podman container prune

# Borrar una imagen
podman rmi pequellm:rocm

# Borrar un volumen (CUIDADO: esto borra tus datos del volumen)
podman volume rm pequellm-data
```

### Entrar a un contenedor para ver que pasa

```bash
./run.sh shell
# o, equivalente, mas largo:
podman run --rm -it \
    --device /dev/kfd --device /dev/dri \
    --group-add keep-groups --security-opt label=disable \
    -v pequellm-data:/workspace/data \
    -v "$(pwd):/workspace/repo" \
    -w /workspace/repo \
    pequellm:rocm bash
```

Adentro puedes navegar como en cualquier shell de Linux:

```bash
ls /workspace/data           # ¿que archivos tengo en el volumen?
python -c "import torch; print(torch.cuda.is_available())"
nvidia-smi                   # NO funciona; usar rocm-smi (si esta instalado)
exit                         # salir; el contenedor se borra (--rm)
```

### Ver lo que paso en un contenedor

```bash
# Si arrancaste un contenedor sin --rm:
podman ps -a                 # busca su ID
podman logs <id>             # imprime stdout/stderr
```

---

## 7. Mapeo: ¿que hace cada subcomando de `./run.sh`?

| Subcomando | Que ejecuta por debajo |
|---|---|
| `./run.sh build` | `podman build -t pequellm:rocm .` |
| `./run.sh smoke` | `podman run [...flags...] pequellm:rocm python /workspace/repo/scripts/check_gpu.py` |
| `./run.sh synth` | `podman run [...flags...] --network none pequellm:rocm python /workspace/repo/scripts/make_synthetic_bins.py` |
| `./run.sh prepare-data` | `podman run [...flags...] -w /workspace/data pequellm:rocm python /workspace/repo/Tokenizador/prepare_data/prepare_data.py` |
| `./run.sh train` | `podman run [...flags...] pequellm:rocm python /workspace/repo/Embeddings/emb_gpt2.py [args]` |
| `./run.sh shell` | `podman run [...flags...] pequellm:rocm bash` |

`[...flags...]` es siempre el bloque de la seccion 3.

---

## 8. Errores tipicos al empezar con contenedores

### "command not found: podman"

Estas en una shell que no tiene podman en el PATH. Si entraste a Distrobox por accidente, sal con `exit`. `podman` solo existe en la shell del host.

### "Error: short-name resolution enforced"

Le diste un nombre de imagen sin registry. Usa `docker.io/rocm/pytorch:latest` en vez de solo `rocm/pytorch:latest`.

### El contenedor termina inmediatamente sin error

Probablemente el comando que le pasaste termino al instante. Recuerda: el contenedor vive solo mientras corra el comando. Si quieres una shell interactiva, usa `bash` como comando y agrega `-it`.

### "no such file or directory" sobre algo que SI existe en tu host

El contenedor solo ve lo que montaste. Si te falta `-v`, el archivo no existe adentro. Verifica con `./run.sh shell` y `ls`.

### Cambios al codigo no se ven adentro

¿Estas seguro de que el archivo que editaste esta dentro del bind-mount? `-v "$(pwd):/workspace/repo"` solo monta el repo. Si editas un archivo en otra parte, el contenedor no lo ve.

### "permission denied" al escribir en `/workspace/data`

El volumen fue creado por otro usuario o con uid distinto. Solucion:

```bash
podman volume rm pequellm-data
# luego recrea con tu usuario:
./run.sh synth     # o ./run.sh prepare-data
```

### La imagen pesa muchisimo y no me cabe

`rocm/pytorch:latest` son ~30 GB. Eso es normal para imagenes con CUDA/ROCm. Si tu disco esta lleno:

```bash
podman system df         # cuanto usa podman en disco
podman system prune -a   # borra TODO lo no-usado (CUIDADO)
```

---

## 9. ¿Donde aprendo mas?

- **Documentacion oficial de podman**: https://docs.podman.io
- **Documentacion oficial de Docker** (los conceptos son iguales): https://docs.docker.com/get-started/
- **rocm/pytorch en Docker Hub**: https://hub.docker.com/r/rocm/pytorch
- **Para entender el "por que" del setup en este repo**: `CLAUDE.md` (en la raiz del repo, generado localmente).

---

## 10. Cheatsheet final (copia esto a una nota)

```bash
# Construir imagen del repo
./run.sh build

# Verificar que la GPU se ve
./run.sh smoke

# Generar datos sinteticos para probar
./run.sh synth

# Entrenar (default, ya con smoke incluido)
./run.sh

# Entrenar con argumentos custom
./run.sh train --max-iters 500 --batch-size 8

# Entrar al contenedor a explorar
./run.sh shell

# Ver que tengo
podman images
podman volume ls
podman ps -a

# Ver que hay en el volumen de datos
ls -lh "$(podman volume inspect pequellm-data --format '{{.Mountpoint}}')/"

# Empezar de cero (BORRA tus datos)
podman volume rm pequellm-data pequellm-cache
podman rmi localhost/pequellm:rocm
```
