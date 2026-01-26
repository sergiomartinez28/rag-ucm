# Guía de Instalación y Uso - RAG-UCM

## 📋 Tabla de Contenidos

1. [Requisitos](#requisitos)
2. [Instalación](#instalación)
3. [Configuración](#configuración)
4. [Preparación de Datos](#preparación-de-datos)
5. [Uso del Sistema](#uso-del-sistema)
6. [Evaluación](#evaluación)
7. [Troubleshooting](#troubleshooting)

---

## 🔧 Requisitos

### Hardware mínimo
- **RAM**: 8GB (16GB recomendado)
- **Disco**: 10GB libres
- **CPU**: 4 cores recomendados
- **GPU**: Opcional (acelera generación, pero funciona en CPU)

### Software
- Python 3.10 o superior
- pip (gestor de paquetes Python)
- Git (opcional, para clonar el repositorio)

---

## 📦 Instalación

### Opción 1: Instalación local

```bash
# 1. Navegar al directorio del proyecto
cd "rag-ucm"

# 2. Crear entorno virtual
python -m venv venv

# 3. Activar entorno virtual
# En macOS/Linux:
source venv/bin/activate
# En Windows:
venv\Scripts\activate

# 4. Actualizar pip
pip install --upgrade pip

# 5. Instalar dependencias
pip install -r requirements.txt

# 6. Copiar configuración de ejemplo
cp .env.example .env
```

### Opción 2: Con Docker

```bash
# Construir imagen
docker build -t rag-ucm .

# Ejecutar contenedor
docker run -p 8501:8501 -v $(pwd)/data:/app/data rag-ucm
```

---

## ⚙️ Configuración

Edita el archivo `.env` con tus preferencias:

```bash
# Modelos (puedes cambiarlos según disponibilidad)
EMBEDDING_MODEL=BAAI/bge-m3
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
LLM_MODEL=meta-llama/Llama-3.2-3B-Instruct

# Alternativas de LLM:
# LLM_MODEL=microsoft/Phi-3-mini-4k-instruct
# LLM_MODEL=Qwen/Qwen2.5-3B-Instruct

# Parámetros de chunking
CHUNK_SIZE=600
CHUNK_OVERLAP=100

# Parámetros de recuperación
TOP_K_RETRIEVAL=20
TOP_K_RERANK=5
HYBRID_ALPHA=0.5  # 0=solo BM25, 1=solo embeddings

# Parámetros de generación
MAX_NEW_TOKENS=512
TEMPERATURE=0.3

# Verificación
ENABLE_VERIFICATION=true
VERIFICATION_THRESHOLD=0.7
```

---

## 📚 Preparación de Datos

### 1. Obtener documentos

Descarga documentos oficiales de la UCM:

- Normativas TFG/TFM de tu facultad
- Calendarios académicos
- Normativa de permanencia
- Guías de procedimientos

Colócalos en `data/raw/`

### 2. Construir índices

```bash
# Usando CLI
python cli.py build --path ./data/raw

# El proceso puede tardar varios minutos
# Verás el progreso en la terminal
```

### 3. Verificar índices

```bash
# Ver estadísticas
python cli.py stats
```

Deberías ver algo como:

```
📊 Estadísticas RAG-UCM

Índices:
  • Total chunks: 245
  • Vectores FAISS: 245
  • Modelo embeddings: bge-m3
  • Dimensión: 1024
  • Longitud promedio: 487 palabras
```

---

## 🚀 Uso del Sistema

### Interfaz Web (Streamlit)

```bash
streamlit run app.py
```

Abre tu navegador en http://localhost:8501

**Características:**
- Interfaz visual amigable
- Configuración en tiempo real
- Visualización de fuentes
- Métricas de verificación

### Línea de Comandos (CLI)

#### Hacer una pregunta

```bash
python cli.py ask "¿Cuál es el plazo para presentar el TFM?"
```

#### Con opciones avanzadas

```bash
python cli.py ask "¿Cuántos créditos puedo convalidar?" \
  --top-k 7 \
  --verbose
```

#### Modo interactivo

```bash
python cli.py interactive
```

Permite hacer múltiples preguntas en una sesión.

### Como librería Python

```python
from src.pipeline import RAGPipeline

# Inicializar
rag = RAGPipeline()

# Hacer pregunta
result = rag.query("¿Cuándo es el plazo del TFG?")

# Mostrar respuesta
print(result['answer'])

# Mostrar fuentes
for source in result['sources']:
    print(f"[{source['id']}] {source['title']}")

# Verificación
if 'verification' in result:
    print(f"Fidelidad: {result['verification']['fidelity_score']:.2%}")
```

---

## 📊 Evaluación

### Crear conjunto de evaluación

Crea un archivo `evaluation/questions.json`:

```json
[
  {
    "question": "¿Cuál es el plazo para presentar el TFM?",
    "expected_answer": "El plazo es...",
    "source_doc": "normativa_tfm_2024.pdf"
  },
  ...
]
```

### Ejecutar evaluación (TODO: implementar)

```bash
python scripts/evaluate.py --questions evaluation/questions.json
```

### Métricas RAGAS

El sistema incluye verificación automática con métricas de:
- **Fidelidad**: ¿La respuesta está respaldada por los documentos?
- **Relevancia**: ¿Los documentos recuperados son relevantes?
- **Completitud**: ¿La respuesta es completa?

---

## 🔧 Troubleshooting

### Problema: "No se encuentran índices"

**Solución**: Ejecuta `python cli.py build` primero.

### Problema: "Out of memory"

**Soluciones**:
1. Reduce `CHUNK_SIZE` en `.env`
2. Reduce `TOP_K_RETRIEVAL`
3. Usa un LLM más pequeño (Phi-3-mini)
4. Cierra otras aplicaciones

### Problema: "Modelo no encontrado"

**Solución**: Los modelos se descargan automáticamente de HuggingFace la primera vez. Asegúrate de tener conexión a internet.

### Problema: Respuestas lentas

**Soluciones**:
1. Si tienes GPU NVIDIA, instala `torch` con CUDA:
   ```bash
   pip install torch --index-url https://download.pytorch.org/whl/cu118
   ```
2. Reduce `TOP_K_RERANK` a 3
3. Usa un modelo más pequeño

### Problema: El sistema "alucina" (inventa información)

**Soluciones**:
1. Activa `ENABLE_VERIFICATION=true`
2. Reduce `TEMPERATURE` a 0.1-0.2
3. Aumenta `VERIFICATION_THRESHOLD`
4. Revisa que los documentos sean completos y claros

---

## 📝 Siguiente Pasos

1. **Expandir documentos**: Añade más normativas a `data/raw/`
2. **Fine-tuning**: Considera hacer fine-tuning del LLM con ejemplos UCM
3. **Evaluación formal**: Crea un conjunto de test con 100+ preguntas
4. **Despliegue**: Dockeriza y despliega en servidor interno

---

## 🆘 Soporte

Para dudas o problemas:
1. Revisa la documentación en `docs/`
2. Consulta el README principal
3. Contacta: sergma22@ucm.es

---

**¡Buena suerte con tu TFM! 🎓**
