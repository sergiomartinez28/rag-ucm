# Instrucciones para la carpeta de datos

## Estructura

```
data/
├── raw/              # Documentos originales (PDFs, HTML)
└── processed/        # Datos procesados e índices
    ├── faiss_index/  # Índice vectorial FAISS
    └── bm25_index/   # Índice léxico BM25
```

## Documentos a incluir en `data/raw/`

Para que el sistema funcione, necesitas añadir documentos oficiales de la UCM en formato PDF o HTML.

### Documentos recomendados:

#### 1. Normativas TFG/TFM
- Normativas de TFG/TFM por facultad
- Rúbricas de evaluación
- Calendarios de defensa

**Ejemplos de nombres de archivo:**
- `normativa_tfg_informatica_2024.pdf`
- `normativa_tfm_economicas_2024.pdf`

#### 2. Normativa de permanencia y reconocimiento
- Normativa de permanencia
- Procedimientos de reconocimiento de créditos
- Convalidaciones

#### 3. Calendarios y plazos
- Calendario académico oficial
- Plazos de matrícula
- Plazos de modificación de matrícula

#### 4. Tasas y precios
- Tasas académicas
- Precios públicos
- Normativa de becas

### Metadatos recomendados

Al procesar documentos, es útil incluir metadatos. Puedes crear un archivo `metadata.json` con información adicional:

```json
{
  "normativa_tfg_informatica_2024.pdf": {
    "title": "Normativa TFG Facultad de Informática",
    "faculty": "Informática",
    "year": "2024",
    "type": "TFG",
    "url": "https://www.fdi.ucm.es/..."
  },
  "calendario_academico_2024_2025.pdf": {
    "title": "Calendario Académico 2024-2025",
    "faculty": "General",
    "year": "2024",
    "type": "Calendario",
    "url": "https://www.ucm.es/..."
  }
}
```

## Fuentes oficiales UCM

Puedes descargar documentos de:

- **Web oficial UCM**: https://www.ucm.es/
- **Normativas por facultad**: Busca en la web de cada facultad
- **Secretaría virtual**: https://www.ucm.es/secretaria-virtual
- **Boletín Oficial UCM**: https://www.ucm.es/buc

## Procesamiento de documentos

Una vez añadidos los documentos en `data/raw/`, ejecuta:

```bash
# Construir índices
python cli.py build --path ./data/raw

# O desde Python
from src.pipeline import RAGPipeline
rag = RAGPipeline(load_existing=False)
rag.build_index()
```

## Notas importantes

⚠️ **Aviso legal**: Asegúrate de que tienes permiso para usar estos documentos. Todos los documentos deben ser de acceso público.

⚠️ **Privacidad**: No incluyas documentos con datos personales de estudiantes o profesores.

⚠️ **Actualización**: Los documentos deben actualizarse periódicamente cuando cambien las normativas.

## Estructura después del procesamiento

Después de ejecutar `build`, la estructura será:

```
data/
├── raw/
│   ├── normativa_tfg_informatica_2024.pdf
│   ├── normativa_tfm_economicas_2024.pdf
│   └── calendario_academico_2024_2025.pdf
└── processed/
    ├── faiss_index/
    │   ├── index.faiss       # Índice vectorial
    │   └── chunks.pkl        # Chunks con metadatos
    └── bm25_index/
        └── bm25.pkl          # Índice léxico
```
