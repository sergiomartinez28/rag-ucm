# RAG-UCM — Asistente Académico con Modelos Open Source

## 📌 Descripción

**Título**: "RAG ligero para asistencia académica en la Universidad Complutense de Madrid: recuperación semántica, generación explicable y control de alucinaciones con modelos open source"

RAG-UCM es un asistente inteligente basado en **Retrieval-Augmented Generation (RAG)** para responder preguntas sobre normativa académica de la Universidad Complutense de Madrid (UCM). 

### Características principales:
- 🔍 **Búsqueda híbrida**: Combina BM25 (búsqueda léxica) + embeddings semánticos
- 🎯 **Re-ranking inteligente**: Cross-encoder para máxima precisión
- ✅ **Verificación de fidelidad**: Control automático de alucinaciones
- 📚 **Citas obligatorias**: Siempre referencia las fuentes oficiales
- 🔓 **100% Open Source**: Sin dependencias comerciales

---

## 🎯 Objetivo e Hipótesis

### Objetivo
Desarrollar un asistente de preguntas y respuestas para estudiantes de la UCM que responda dudas prácticas (normativa TFG/TFM, matrículas, reconocimiento de créditos, becas, plazos administrativos…) citando siempre las fuentes oficiales, usando únicamente software y modelos open source y ejecutándose en hardware local/modesto.

### Hipótesis
Un sistema RAG "ligero", basado en búsqueda híbrida, re-ranking cruzado y verificación de fidelidad, puede ofrecer respuestas útiles y fieles a normativa universitaria sin necesidad de usar grandes modelos privados/comerciales.

---

## 🗂️ Estructura del Proyecto

```
rag-ucm/
├── data/
│   ├── raw/              # Documentos originales (PDFs/HTML)
│   └── processed/        # Texto limpio + chunks + metadatos
├── src/
│   ├── __init__.py
│   ├── preprocessor.py   # Limpieza y chunking de documentos
│   ├── indexer.py        # Indexación FAISS + BM25
│   ├── retrieval.py      # Búsqueda híbrida + re-ranking
│   ├── generator.py      # Generación de respuestas con LLM
│   ├── verifier.py       # Verificación de fidelidad
│   └── pipeline.py       # Pipeline completo RAG
├── notebooks/            # Análisis exploratorios y experimentos
├── tests/                # Tests unitarios
├── docs/                 # Documentación adicional
├── config/               # Configuraciones
├── app.py               # Interfaz Streamlit
├── cli.py               # Interfaz línea de comandos
├── requirements.txt     # Dependencias Python
├── Dockerfile           # Containerización
├── .env.example         # Variables de entorno
└── README.md           # Este archivo
```

---

## 🚀 Instalación

### Requisitos previos
- Python 3.10+
- 8GB RAM mínimo (16GB recomendado)
- 10GB espacio en disco

### Instalación local

```bash
# Clonar el repositorio
cd rag-ucm

# Crear entorno virtual
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt

# Configurar variables de entorno
cp .env.example .env
```

### Con Docker

```bash
docker build -t rag-ucm .
docker run -p 8501:8501 rag-ucm
```

---

## 📖 Uso

### Interfaz web (Streamlit)

```bash
streamlit run app.py
```

Abre tu navegador en `http://localhost:8501`

### CLI

```bash
python cli.py "¿Cuándo es el plazo para presentar el TFM?"
```

### Como librería

```python
from src.pipeline import RAGPipeline

# Inicializar el pipeline
rag = RAGPipeline()

# Hacer una pregunta
response = rag.query("¿Cuántos créditos puedo convalidar?")

print(response['answer'])
print(response['sources'])
```

---

## 🔬 Metodología Técnica

### 1. Colección de Documentos
- Normativas TFG/TFM por facultad
- Calendarios académicos y plazos
- Procedimientos de reconocimiento/convalidación
- Normativa de permanencia
- Tasas y precios públicos

### 2. Preprocesado
- Extracción de texto desde PDF/HTML
- Limpieza y normalización
- Chunking semántico (~500-800 tokens, solape ~100)
- Preservación de metadatos (título, facultad, fecha, URL)

### 3. Indexación
- **Embeddings**: `bge-m3` o `multilingual-e5-base`
- **Índice vectorial**: FAISS o Qdrant
- **Índice léxico**: BM25 (Whoosh/Elasticsearch)

### 4. Recuperación
1. Reformulación opcional de query (query expansion)
2. Búsqueda híbrida:
   - Similitud semántica (FAISS)
   - BM25 (términos exactos)
3. Fusión con Reciprocal Rank Fusion
4. Re-ranking con cross-encoder (`bge-reranker-v2-m3`)

### 5. Generación
- **LLM**: Llama-3.2-3B-Instruct / Phi-4-mini / Qwen2.5-3B-Instruct
- Prompt con instrucciones de citar fuentes
- Respuesta estructurada con referencias

### 6. Verificación de Fidelidad
- Evaluación automática de cada afirmación
- Detección de posibles alucinaciones
- Advertencias cuando la información no está respaldada

---

## 📊 Evaluación

### Métricas
- **RAGAS**: Fidelidad, relevancia, completitud
- **Precisión de citas**: % respuestas con referencias correctas
- **Latencia**: Tiempo de respuesta
- **Tasas de "no sé"**: Cuándo el sistema es prudente

### Comparación de enfoques
- BM25 solo
- Embeddings solo
- **Híbrido + reranker + verificador** (sistema final)

---

## 🎯 Alcance Funcional

### ✅ El sistema PUEDE:
- Responder preguntas sobre normativa UCM en lenguaje natural
- Recuperar fragmentos relevantes de documentos oficiales
- Generar respuestas claras en español con tono administrativo
- Incluir citas precisas de documentos originales
- Indicar cuando no tiene información suficiente

### ❌ El sistema NO:
- Da consejo legal personalizado
- Hace interpretaciones académicas subjetivas
- Sustituye a secretaría (siempre remite a la fuente)

---

## 🛠️ Tecnologías Utilizadas

### Modelos
- **Embeddings**: BGE-M3, Multilingual-E5
- **Re-ranking**: BGE-Reranker-v2-M3
- **Generación**: Llama-3.2-3B / Phi-4 / Qwen2.5-3B
- **Verificación**: Mismo LLM en modo crítico

### Librerías principales
- `transformers` - Modelos de HuggingFace
- `sentence-transformers` - Embeddings
- `faiss` - Búsqueda vectorial
- `rank-bm25` - Búsqueda léxica
- `langchain` - Orquestación RAG
- `streamlit` - Interfaz web
- `typer` - CLI

---

## 📅 Plan de Trabajo

### Fase 1: Definición del alcance ✅
- Selección de normativas UCM
- Definición de tipos de preguntas objetivo

### Fase 2: Adquisición y limpieza de datos 🔄
- Descarga de PDFs/HTML oficiales
- Conversión y limpieza
- Chunking e indexación

### Fase 3: Prototipo RAG básico
- Recuperación + generación básica
- Validación de coherencia y citas

### Fase 4: Mejora de recuperación
- Implementar búsqueda híbrida
- Añadir re-ranking
- Medir mejoras (recall@5, precisión)

### Fase 5: Verificación de fidelidad
- Implementar chequeo de alucinaciones
- Sistema de advertencias automáticas

### Fase 6: Evaluación
- Crear conjunto de ~100-150 preguntas
- Calcular métricas RAGAS
- Comparar con baselines

### Fase 7: Demo y memoria
- Streamlit + Docker
- Redacción memoria (≤20 páginas)

---

## 📝 Limitaciones

- Cobertura limitada a documentos públicos UCM incluidos
- Los modelos pequeños pueden tener límites de comprensión
- Requiere actualización periódica de normativas
- No sustituye consulta directa con secretaría

---

## 🔮 Trabajo Futuro

- Expandir a todas las facultades UCM
- Integración con sistemas de gestión académica
- Soporte multiidioma (inglés para estudiantes internacionales)
- Fine-tuning del LLM con lenguaje administrativo UCM
- Despliegue interno para secretarías

---

## 📄 Licencia

MIT License - Ver archivo `LICENSE` para más detalles

---

## 👤 Autor

**Sergio Martín**
- TFM - Máster [nombre del máster]
- Universidad Complutense de Madrid
- sergma22@ucm.es

---

## 🙏 Agradecimientos

- Universidad Complutense de Madrid por la disponibilidad de normativas públicas
- Comunidad open source de HuggingFace y LangChain
- [Nombre del tutor/a] por la supervisión del TFM
