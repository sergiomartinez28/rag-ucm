# RAG-UCM — Asistente de Consultas de Normativa Académica

## 📌 Descripción

RAG-UCM es un asistente inteligente basado en *Retrieval-Augmented Generation (RAG)* para responder a preguntas sobre normativa académica de la Universidad Complutense de Madrid (UCM). Integra búsqueda híbrida (BM25 + embeddings), re-ranking y verificación de fidelidad, generando respuestas claras que siempre citan las fuentes oficiales.

Este proyecto está diseñado con herramientas y modelos **open source**, y puede ejecutarse localmente sin necesidad de servicios comerciales.

---

## 🗂️ Estructura del proyecto

```

rag-ucm/
├── data/
│   ├── raw/           # Documentos descargados (PDFs/HTML)
│   ├── processed/     # Texto limpio + chunks
├── src/
│   ├── indexer.py
│   ├── retrieval.py
│   ├── generator.py
│   └── verifier.py
├── notebooks/         # Análisis y pruebas exploratorias
├── Dockerfile
├── README.md
└── LICENSE

````
