"""
Pipeline completo RAG-UCM
Integra todos los componentes del sistema
"""

import os
import time
from pathlib import Path
from typing import Dict, List, Optional
from dotenv import load_dotenv
from loguru import logger

from .preprocessor import DocumentPreprocessor
from .indexer import DocumentIndexer
from .retrieval import HybridRetriever
from .generator import ResponseGenerator
from .verifier import FidelityVerifier


class RAGPipeline:
    """
    Pipeline completo del sistema RAG-UCM
    
    Flujo:
    1. Preprocesamiento (offline)
    2. Indexación (offline)
    3. Recuperación híbrida (online)
    4. Generación de respuesta (online)
    5. Verificación de fidelidad (online)
    """
    
    def __init__(
        self,
        config_path: Optional[Path] = None,
        load_existing: bool = True
    ):
        """
        Args:
            config_path: Ruta al archivo .env de configuración
            load_existing: Si True, intenta cargar índices existentes
        """
        # Cargar configuración
        if config_path:
            load_dotenv(config_path)
        else:
            load_dotenv()
        
        logger.info("Inicializando RAG-UCM Pipeline...")
        
        # Configuración desde variables de entorno
        # Configuración optimizada para modelos rápidos
        self.config = {
            'embedding_model': os.getenv('EMBEDDING_MODEL', 'BAAI/bge-m3'),
            'reranker_model': os.getenv('RERANKER_MODEL', 'BAAI/bge-reranker-base'),
            'reranker_type': os.getenv('RERANKER_TYPE', 'cross-encoder'),
            'llm_model': os.getenv('LLM_MODEL', 'Qwen/Qwen2.5-3B-Instruct'),
            'chunk_size': int(os.getenv('CHUNK_SIZE', 1000)),  
            'chunk_overlap': int(os.getenv('CHUNK_OVERLAP', 200)), 
            'top_k_retrieval': int(os.getenv('TOP_K_RETRIEVAL', 8)),  # Optimizado
            'top_k_rerank': int(os.getenv('TOP_K_RERANK', 3)),  # Top 3 documentos finales
            'hybrid_alpha': float(os.getenv('HYBRID_ALPHA', 0.6)),  # Más peso a semántico
            'max_new_tokens': int(os.getenv('MAX_NEW_TOKENS', 120)),  # Respuestas concisas
            'temperature': float(os.getenv('TEMPERATURE', 0.1)),  # Más determinístico
            'enable_verification': os.getenv('ENABLE_VERIFICATION', 'false').lower() == 'true',  # Desactivado para velocidad
            'verification_threshold': float(os.getenv('VERIFICATION_THRESHOLD', 0.7)),
            'data_raw_path': Path(os.getenv('DATA_RAW_PATH', './data/raw')),
            'data_processed_path': Path(os.getenv('DATA_PROCESSED_PATH', './data/processed')),
            'faiss_index_path': Path(os.getenv('FAISS_INDEX_PATH', './data/processed/faiss_index')),
            'bm25_index_path': Path(os.getenv('BM25_INDEX_PATH', './data/processed/bm25_index')),
        }
        
        # Inicializar componentes
        self.preprocessor = None
        self.indexer = None
        self.retriever = None
        self.generator = None
        self.verifier = None
        
        # Cargar índices si existen
        if load_existing:
            self._load_indices()
        
        logger.success("✓ RAG-UCM Pipeline inicializado")
    
    def _load_indices(self):
        """Intenta cargar índices existentes"""
        try:
            t_start = time.time()
            logger.info("Intentando cargar índices existentes...")
            
            # Inicializar indexador
            t0 = time.time()
            self.indexer = DocumentIndexer(
                embedding_model=self.config['embedding_model'],
                faiss_index_path=self.config['faiss_index_path'],
                bm25_index_path=self.config['bm25_index_path']
            )
            
            # Cargar índices
            self.indexer.load_indices()
            t_indexer = time.time() - t0
            
            # Inicializar retriever
            t0 = time.time()
            self.retriever = HybridRetriever(
                indexer=self.indexer,
                reranker_model=self.config['reranker_model'],
                reranker_type=self.config['reranker_type'],
                alpha=self.config['hybrid_alpha']
            )
            t_retriever = time.time() - t0
            
            # Inicializar generador
            t0 = time.time()
            self.generator = ResponseGenerator(
                model_name=self.config['llm_model'],
                max_new_tokens=self.config['max_new_tokens'],
                temperature=self.config['temperature']
            )
            t_generator = time.time() - t0
            
            # Inicializar verificador
            if self.config['enable_verification']:
                self.verifier = FidelityVerifier(
                    threshold=self.config['verification_threshold']
                )
            
            t_total = time.time() - t_start
            logger.info(f"⏱️  Carga: Indexer {t_indexer:.2f}s | Retriever {t_retriever:.2f}s | Generator {t_generator:.2f}s | Total {t_total:.2f}s")
            logger.success("✓ Índices y modelos cargados correctamente")
            
        except Exception as e:
            logger.warning(f"No se pudieron cargar índices existentes: {e}")
            logger.info("Necesitarás ejecutar build_index() primero")
    
    def build_index(self, documents_path: Optional[Path] = None):
        """
        Construye los índices desde cero
        
        Args:
            documents_path: Ruta a la carpeta con documentos (PDFs/HTML)
        """
        if documents_path is None:
            documents_path = self.config['data_raw_path']
        
        logger.info(f"Construyendo índices desde: {documents_path}")
        
        # Inicializar preprocessor
        self.preprocessor = DocumentPreprocessor(
            chunk_size=self.config['chunk_size'],
            chunk_overlap=self.config['chunk_overlap']
        )
        
        # Procesar todos los documentos
        all_chunks = []
        
        for doc_file in documents_path.glob('**/*.pdf'):
            try:
                logger.info(f"Procesando: {doc_file.name}")
                
                # Aquí deberías añadir metadatos específicos por documento
                # Por ejemplo, extrayéndolos del nombre del archivo o de un CSV
                metadata = {
                    'title': doc_file.stem,
                    # 'faculty': '...',
                    # 'year': '...',
                    # 'url': '...'
                }
                
                chunks = self.preprocessor.process_document(doc_file, metadata)
                all_chunks.extend(chunks)
                
            except Exception as e:
                logger.error(f"Error procesando {doc_file.name}: {e}")
        
        logger.info(f"Total de chunks procesados: {len(all_chunks)}")
        
        # Inicializar e indexar
        self.indexer = DocumentIndexer(
            embedding_model=self.config['embedding_model'],
            faiss_index_path=self.config['faiss_index_path'],
            bm25_index_path=self.config['bm25_index_path']
        )
        
        self.indexer.index_chunks(all_chunks)
        self.indexer.save_indices()
        
        # Inicializar retriever
        self.retriever = HybridRetriever(
            indexer=self.indexer,
            reranker_model=self.config['reranker_model'],
            reranker_type=self.config['reranker_type'],
            alpha=self.config['hybrid_alpha']
        )
        
        # Inicializar generador
        self.generator = ResponseGenerator(
            model_name=self.config['llm_model'],
            max_new_tokens=self.config['max_new_tokens'],
            temperature=self.config['temperature']
        )
        
        # Inicializar verificador
        if self.config['enable_verification']:
            self.verifier = FidelityVerifier(
                threshold=self.config['verification_threshold']
            )
        
        logger.success("✓ Índices construidos y guardados correctamente")
    
    def query(
        self,
        question: str,
        top_k: Optional[int] = None,
        include_verification: Optional[bool] = None
    ) -> Dict:
        """
        Responde a una pregunta usando el sistema RAG completo
        
        Args:
            question: Pregunta del usuario
            top_k: Número de documentos a recuperar (usa config por defecto)
            include_verification: Activar verificación (usa config por defecto)
        
        Returns:
            Dict con respuesta, fuentes, verificación, y metadata
        """
        if self.retriever is None or self.generator is None:
            raise RuntimeError(
                "El pipeline no está inicializado. "
                "Ejecuta build_index() o asegúrate de que existen índices."
            )
        
        logger.info(f"📝 Pregunta: {question}")
        
        # Parámetros
        if top_k is None:
            top_k = self.config['top_k_rerank']
        if include_verification is None:
            include_verification = self.config['enable_verification']
        
        # 1. Recuperación
        t0 = time.time()
        contexts = self.retriever.retrieve(
            query=question,
            top_k_retrieval=self.config['top_k_retrieval'],
            top_k_final=top_k
        )
        retrieval_time = time.time() - t0
        logger.info(f"⏱️  Recuperación: {retrieval_time:.2f}s")
        
        # 2. Generación
        t0 = time.time()
        response = self.generator.generate(question, contexts)
        generation_time = time.time() - t0
        logger.info(f"⏱️  Generación: {generation_time:.2f}s")
        
        # 3. Verificación (opcional)
        verification_time = 0
        if include_verification and self.verifier and contexts:
            t0 = time.time()
            verification = self.verifier.verify(
                answer=response['answer'],
                contexts=contexts
            )
            
            # Añadir verificación de citas
            citation_check = self.verifier.add_citation_check(
                answer=response['answer'],
                num_sources=len(contexts)
            )
            
            verification['citation_check'] = citation_check
            response['verification'] = verification
            
            # Añadir advertencia si es necesario
            if verification.get('warning'):
                response['warning'] = verification['warning']
            
            verification_time = time.time() - t0
            logger.info(f"⏱️  Verificación: {verification_time:.2f}s")
        
        # Tiempo total
        total_time = retrieval_time + generation_time + verification_time
        logger.info(f"⏱️  TOTAL: {total_time:.2f}s")
        
        # Añadir tiempos a la respuesta
        response['timing'] = {
            'retrieval': retrieval_time,
            'generation': generation_time,
            'verification': verification_time,
            'total': total_time
        }
        
        logger.success("✓ Respuesta generada y verificada")
        
        return response
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del sistema"""
        stats = {
            'config': self.config,
            'status': {}
        }
        
        if self.indexer:
            stats['index'] = self.indexer.get_stats()
        else:
            stats['index'] = {'status': 'No inicializado'}
        
        return stats


if __name__ == "__main__":
    # Ejemplo de uso
    
    # Inicializar pipeline (carga índices existentes si están disponibles)
    rag = RAGPipeline()
    
    # Si no hay índices, construirlos:
    # rag.build_index(Path("data/raw"))
    
    # Hacer una pregunta
    # result = rag.query("¿Cuál es el plazo para presentar el TFM?")
    # print(result['answer'])
    # print("\nFuentes:")
    # for source in result['sources']:
    #     print(f"[{source['id']}] {source['title']}")
    
    print("✓ Pipeline RAG-UCM listo para usar")
