# RAG-UCM — Asistente Académico con Modelos Open Source

## Descripción

**Título**: "RAG ligero para asistencia académica en la Universidad Complutense de Madrid"

RAG-UCM es un asistente inteligente basado en **Retrieval-Augmented Generation (RAG)** para responder preguntas sobre normativa académica de la Universidad Complutense de Madrid (UCM). 

### Características principales:
- **Búsqueda híbrida**: Combina BM25 (búsqueda léxica) + embeddings semánticos
- **Re-ranking inteligente**: Cross-encoder para máxima precisión
- **Verificación de fidelidad**: Control automático de alucinaciones
- **Citas obligatorias**: Siempre referencia las fuentes oficiales
- **100% Open Source**: Sin dependencias comerciales

---

## Objetivo e Hipótesis

### Objetivo
Desarrollar un asistente de preguntas y respuestas para estudiantes de la UCM que responda dudas prácticas (normativa TFG/TFM, matrículas, reconocimiento de créditos, becas, plazos administrativos…) citando siempre las fuentes oficiales, usando únicamente software y modelos open source y ejecutándose en hardware local/modesto.

### Hipótesis
Un sistema RAG "ligero", basado en búsqueda híbrida, re-ranking cruzado y verificación de fidelidad, puede ofrecer respuestas útiles y fieles a normativa universitaria sin necesidad de usar grandes modelos privados/comerciales.

---

## Estructura del Proyecto

```
rag-ucm/
├── app.py                  # Interfaz web Streamlit
├── cli.py                  # Interfaz línea de comandos
├── process_documents.py    # Script para indexar documentos
├── evaluate_rag.py         # Script de evaluación del sistema
├── requirements.txt        # Dependencias Python
├── pytest.ini              # Configuración de tests
├── README.md               # Este archivo
│
├── data/
│   ├── raw/                # Documentos originales (PDFs/HTML)
│   ├── processed/          # Índices FAISS y BM25
│   └── evaluation/         # Dataset y resultados de evaluación
│
├── src/
│   ├── __init__.py
│   ├── config.py           # Configuración centralizada (Pydantic)
│   ├── preprocessor.py     # Extracción y chunking de documentos
│   ├── indexer.py          # Indexación FAISS + BM25
│   ├── retrieval.py        # Búsqueda híbrida + re-ranking
│   ├── generator.py        # Generación de respuestas con LLM
│   ├── verifier.py         # Verificación de fidelidad
│   ├── pipeline.py         # Pipeline completo RAG
│   ├── prompt_loader.py    # Carga de prompts externos
│   ├── utils.py            # Utilidades (timing, memoria)
│   └── evaluator/          # Módulo de evaluación
│       ├── dataset_generator.py
│       ├── rag_evaluator.py
│       ├── llm_judge.py
│       └── metrics.py
│
├── tests/                  # Tests unitarios
│   ├── conftest.py         # Fixtures compartidos
│   ├── test_config.py
│   ├── test_preprocessor.py
│   ├── test_prompt_loader.py
│   ├── test_utils.py
│   └── test_metrics.py
│
├── prompts/                # Plantillas de prompts
│   ├── system_prompt.txt
│   ├── user_prompt.txt
│   ├── dataset_generator.txt
│   ├── judge_evaluation.txt
│   └── judge_user_prompt.txt
│
└── docs/
    └── INSTALLATION.md     # Guía detallada de instalación
```

---

## Instalación

### Requisitos previos
- Python 3.10+
- 8GB RAM mínimo (16GB recomendado)
- GPU opcional (acelera generación, funciona en CPU)
- 10GB espacio en disco

### Instalación

```bash
# Clonar el repositorio
git clone https://github.com/sergiomartinez28/rag-ucm.git
cd rag-ucm

# Crear entorno virtual
python -m venv .venv

# Activar entorno
# En Windows:
.venv\Scripts\activate
# En Linux/Mac:
source .venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

> 💡 **Configuración**: Todos los parámetros están en `src/config.py` con valores optimizados.

### Preparación de datos

```bash
# 1. Colocar documentos PDF/HTML en data/raw/

# 2. Procesar e indexar documentos
python process_documents.py
```

---

## Uso

### Interfaz web (Streamlit)

```bash
streamlit run app.py
```

Abre tu navegador en `http://localhost:8501`

### CLI

```bash
# Hacer una pregunta
python cli.py ask "¿Cuándo es el plazo para presentar el TFM?"

# Construir índices
python cli.py build --path ./data/raw

# Ver estadísticas
python cli.py stats
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

### Tests

```bash
# Ejecutar todos los tests
pytest

# Con cobertura
pytest --cov=src

# Tests específicos
pytest tests/test_config.py -v
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
- Extracción de texto desde PDF (PyMuPDF/pdfplumber) y HTML
- Limpieza y normalización
- Chunking semántico (~1000 tokens, solape 200)
- Preservación de metadatos (título, facultad, fecha, URL)

### 3. Indexación
- **Embeddings**: BAAI/bge-m3 (1024 dimensiones)
- **Índice vectorial**: FAISS (IndexFlatIP)
- **Índice léxico**: BM25 con tokenizador español

### 4. Recuperación
1. Búsqueda híbrida:
   - Similitud semántica (FAISS, top-10)
   - BM25 (términos exactos, top-10)
2. Fusión con Reciprocal Rank Fusion (alpha=0.45)
3. Re-ranking con cross-encoder (BAAI/bge-reranker-base)
4. Filtrado por umbral de score (min=0.5)
5. Top-3 documentos finales

### 5. Generación
- **LLM**: Qwen/Qwen2.5-3B-Instruct (cuantizado 4-bit)
- Cuantización automática para reducir ~50% uso de VRAM
- Prompt con instrucciones de citar fuentes
- Retry inteligente con contexto reducido si abstiene
- Máximo 100 tokens, temperatura 0.1

### 6. Verificación de Fidelidad
- Evaluación automática de cada afirmación
- Detección de posibles alucinaciones
- Advertencias cuando la información no está respaldada

---

## 📊 Evaluación

El sistema incluye un framework de evaluación completo con:

### Generación de Dataset
```bash
# Generar preguntas desde chunks (una vez)
python evaluate_rag.py generate --num-samples 100
```

### Ejecutar Evaluación
```bash
# Evaluar con dataset existente
python evaluate_rag.py evaluate

# Evaluación rápida (100 preguntas)
python evaluate_rag.py evaluate --limit 100
```

### Métricas
- **Precision@k**: Documento correcto en top-k resultados
- **Relevancia**: ¿La respuesta responde a la pregunta?
- **Fidelidad**: ¿La respuesta se basa en los documentos?
- **Precisión**: ¿La respuesta es correcta vs referencia?
- **Tasa de abstención**: Cuando el sistema dice "no sé"

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
- **Embeddings**: BAAI/bge-m3 (1024 dims, multilingüe)
- **Re-ranking**: BAAI/bge-reranker-base (cross-encoder)
- **Generación**: Qwen/Qwen2.5-3B-Instruct

### Librerías principales
- `transformers` - Modelos de HuggingFace
- `sentence-transformers` - Embeddings
- `faiss-cpu` - Búsqueda vectorial
- `rank-bm25` - Búsqueda léxica
- `pydantic` - Validación de configuración
- `streamlit` - Interfaz web
- `typer` + `rich` - CLI

---

## 📅 Estado del Proyecto

- ✅ **Fase 1**: Definición del alcance y selección de normativas
- ✅ **Fase 2**: Adquisición y limpieza de datos (PDFs/HTML)
- ✅ **Fase 3**: Prototipo RAG básico con recuperación + generación
- ✅ **Fase 4**: Búsqueda híbrida (BM25 + semántica) + re-ranking
- ✅ **Fase 5**: Verificación de fidelidad y control de abstenciones
- ✅ **Fase 6**: Evaluación con dataset de 449 preguntas
- ✅ **Fase 7**: Demo Streamlit + CLI

### Resultados de Evaluación

| Métrica | Valor |
|---------|-------|
| Overall Score | 0.72 |
| Precision | 0.62 |
| Fidelidad | 0.74 |
| Abstención | 0.0% |
| Tiempo retrieval | ~5s |
| Tiempo generación | ~50s |

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

## 👤 Autor

**Sergio Martínez Asenjo**
- TFM - Máster de Formación Permanente en
Big Data, Data Science e Inteligencia Artificial
- Universidad Complutense de Madrid
- sergma22@ucm.es

---

## 🙏 Agradecimientos

- Universidad Complutense de Madrid por la disponibilidad de normativas públicas
- Comunidad open source de HuggingFace
