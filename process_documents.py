#!/usr/bin/env python
"""
Script para procesar documentos HTML de raw/ y crear índices
Usa el método build_index() del pipeline RAG
Uso: python process_documents.py
"""

from pathlib import Path
from src.pipeline import RAGPipeline

def main():
    """Procesa todos los documentos y crea índices"""
    
    print("\n" + "=" * 60)
    print("Iniciando procesamiento de documentos")
    print("=" * 60 + "\n")
    
    # Inicializar pipeline sin cargar índices existentes
    rag = RAGPipeline(load_existing=False)
    
    # Construir índices desde data/raw/
    rag.build_index(Path("./data/raw"))
    
    print("\n" + "=" * 60)
    print("✅ PROCESAMIENTO COMPLETADO")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
