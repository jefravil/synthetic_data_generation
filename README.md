# Synthetic Data Generation

Pipeline reutilizable para generar variaciones sintéticas de cualquier dataset de imágenes.
Útil para aumentar datasets pequeños de ML (enfermedades de plantas, defectos industriales, imágenes médicas, etc.).

---

## Requisitos

- Python 3.10+
- GPU NVIDIA con CUDA (recomendado ≥6 GB VRAM)
- 64 GB RAM recomendado si usas FLUX schnell con offload a CPU

---

## 1. Instalación

```bash
# Clona o descarga el proyecto y entra a la carpeta
cd synthetic_data_ws

# Crea y activa un entorno virtual (recomendado)
python -m venv synt_env

# Linux / macOS
source synt_env/bin/activate

# Windows
synt_env\Scripts\activate

# Instala PyTorch con CUDA (elige según tu versión de CUDA)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
# CPU only (sin GPU):
# pip install torch torchvision

# Instala el resto de dependencias
pip install -r requirements.txt

# Lanza Jupyter
jupyter lab
```

Se abre tu navegador en `http://localhost:8888`. Haz clic en `synthetic_data_generation.ipynb`.

---

## 2. Inicio rápido

```
1. Crea la carpeta   ./data/images/   y coloca tus imágenes ahí
2. Abre el notebook y edita SOLO la Celda 2 (CONFIG):
   - dataset_path     → ruta a tu carpeta de imágenes
   - img2img_prompt   → describe tu dominio (ej: "diseased cacao pod, macro photography")
   - img2img_backend  → "sdxl_lightning" (recomendado) | "flux_schnell" | "sd15_legacy"
3. Ejecuta celda por celda con Shift+Enter
```

---

## 3. Estructura del notebook

```
Celda 1 [Markdown]  ← Título y guía de decisión (solo lectura)
  ↓
Celda 2 [Code]  CONFIG  ← ⭐ EDITA AQUÍ (única celda que debes modificar)
  ↓
Celda 3 [Code]  Imports  ← Verifica GPU, CUDA, versiones
  ↓
Celda 4 [Code]  Dataset  ← Carga imágenes, muestra grilla de 16
  ↓
Celda 5 [Code]  Estrategia 1: Augmentación Clásica  ← Sin GPU, rápido
  ↓
Celda 6 [Code]  Estrategia 2: VAE Latent Perturbation  ← Encode→Perturb→Decode + PCA
  ↓
Celda 7 [Code]  Estrategia 3: img2img Diffusion  ← Máxima calidad (3 backends)
  ↓
Celda 8 [Code]  Evaluación + Export  ← SSIM, comparación visual, manifest.json
```

---

## 4. Las tres estrategias

| # | Estrategia | Diversidad | GPU necesaria | Velocidad |
|---|---|---|---|---|
| 1 | Augmentación Clásica | Baja | No | ~1-2 s/img |
| 2 | VAE Encode→Perturb→Decode | Media | 0.5–2 GB | ~2-5 s/img |
| 3 | img2img Diffusion | Alta | 4–8 GB | Varía por backend |

### Backends img2img disponibles

| Backend | VRAM pico | Velocidad | Cuándo usarlo |
|---|---|---|---|
| `sdxl_lightning` ⭐ | ~6–7 GB | ~3 s/img | Default recomendado |
| `flux_schnell` | ~7–8 GB (NF4) | ~10–30 s/img | Máxima calidad fotorealista |
| `sd15_legacy` | ~4 GB | ~5–10 s/img | Hardware limitado / fallback |

---

## 5. Cómo ejecutar celdas en Jupyter

**Opción A — Una por una (recomendado para aprender):**
1. Haz clic en una celda
2. Presiona `Shift + Enter`
3. Espera a que termine (el output aparece abajo)
4. Pasa a la siguiente

**Opción B — Todas de golpe:**
- Menú → `Run` → `Run All Cells`

**Indicadores de estado:**
- `[*]` a la izquierda → está procesando
- `[3]` → terminó (número = orden de ejecución)

---

## 6. Flujo recomendado

### Primera vez (prueba rápida sin GPU)

```
CONFIG:
  run_classical = True
  run_vae       = False
  run_img2img   = False

Ejecuta celdas 2 → 3 → 4 → 5
Resultado: variaciones en output/classical/
```

### Segunda vez (pipeline completo)

```
CONFIG:
  run_classical = True
  run_vae       = True
  run_img2img   = True
  img2img_backend = "sdxl_lightning"
  img2img_prompt  = "palabras de tu dominio"

Ejecuta todas las celdas en orden
```

> **Nota:** La primera ejecución descarga los modelos (~335 MB para VAE SD,
> ~6.5 GB para SDXL Lightning, ~24 GB para FLUX schnell). Solo ocurre una vez;
> quedan en caché en `~/.cache/huggingface/`.

---

## 7. Reproducibilidad (semillas)

Cada imagen generada tiene su semilla guardada en `output/manifest.json`:

```json
{
  "strategy": "classical",
  "source": "data/images/img001.jpg",
  "output": "output/classical/img001_aug00.png",
  "seed": 42
}
```

Para replicar exactamente el mismo dataset: usa el mismo `aug_seed` en CONFIG.
Para generar un dataset alternativo: cambia `aug_seed` a cualquier otro número.

---

## 8. Espacio latente (VAE) — ¿Qué extrae?

La Estrategia 2 extrae el **vector latente** del VAE (no features de CNN):

```python
z = vae.encode(x).latent_dist.mean  # shape: [1, 4, 64, 64] para SD VAE
```

El notebook incluye visualización **PCA 2D** (matplotlib) y **PCA 3D interactiva** (plotly).
Cada punto en el gráfico es una imagen; las imágenes similares aparecen cerca entre sí.

---

## 9. Estructura de salida

```
output/
├── classical/        ← Augmentación clásica
├── vae/              ← Perturbaciones latentes
├── img2img/          ← Difusión img2img
├── all_combined/     ← Symlinks a todo (úsalo como dataset final)
└── manifest.json     ← Metadata de cada imagen generada
```

---

## 10. Consejos

| Problema | Solución |
|---|---|
| `No se encontraron imágenes` | Verifica `dataset_path` en CONFIG |
| OOM en SDXL/FLUX | Reinicia kernel, ejecuta solo la celda img2img con GPU limpia |
| Lentísimo con FLUX | Usa `sdxl_lightning` o `sd15_legacy` |
| Import que falta | `pip install nombre_paquete` en terminal con el venv activo |
| Quiero replicar resultados | Usa el mismo `aug_seed` en CONFIG |
| Quiero más diversidad | Sube `num_variations_per_image` y/o `img2img_strengths` |

---

## 11. Dependencias principales

| Paquete | Uso |
|---|---|
| `torch` + `torchvision` | Cómputo GPU |
| `diffusers` | Pipelines de difusión (SDXL, FLUX, SD15) |
| `transformers` | Modelos de lenguaje (CLIP, T5) |
| `bitsandbytes` | Cuantización NF4 para FLUX |
| `peft` | Carga de LoRA (SDXL Lightning) |
| `albumentations` | Augmentación clásica |
| `Pillow` | Procesamiento de imágenes |
| `scikit-image` | Métricas SSIM |
| `plotly` | Visualización 3D interactiva |
