"""
Pipeline completo RAG-UCM
Integra todos los componentes del sistema con arquitectura refactorizada
"""

from pathlib import Path
from typing import Dict, List, Optional
from loguru import logger

from .config import get_config, RAGConfig
from .utils import TimingContext, torch_memory_cleanup
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
        config: Optional[RAGConfig] = None,
        load_existing: bool = True
    ):
        """
        Args:
            config: Configuración personalizada (usa config global si es None)
            load_existing: Si True, intenta cargar índices existentes
        """
        logger.info("Inicializando RAG-UCM Pipeline...")
        
        # Cargar configuración
        self.config = config or get_config()
        
        # Inicializar componentes
        self.preprocessor: Optional[DocumentPreprocessor] = None
        self.indexer: Optional[DocumentIndexer] = None
        self.retriever: Optional[HybridRetriever] = None
        self.generator: Optional[ResponseGenerator] = None
        self.verifier: Optional[FidelityVerifier] = None
        
        # Cargar índices si existen
        if load_existing:
            try:
                self._load_indices()
            except Exception as e:
                logger.warning(f"No se pudieron cargar índices: {e}")
                logger.info("Ejecuta build_index() para crear índices")
        
        logger.success("✓ RAG-UCM Pipeline inicializado")
    
    def _load_indices(self) -> None:
        """Carga índices existentes y modelos"""
        with TimingContext("Carga completa de sistema"):
            # Inicializar y cargar indexador
            with TimingContext("Indexador"):
                self.indexer = DocumentIndexer(
                    embedding_model=self.config.models.embedding_model,
                    faiss_index_path=self.config.paths.faiss_index_path,
                    bm25_index_path=self.config.paths.bm25_index_path
                )
                self.indexer.load_indices()
            
            # Inicializar retriever
            with TimingContext("Retriever"):
                self.retriever = HybridRetriever(
                    indexer=self.indexer,
                    reranker_model=self.config.models.reranker_model,
                    reranker_type=self.config.models.reranker_type,
                    use_cross_encoder=self.config.models.use_cross_encoder,
                    alpha=self.config.retrieval.hybrid_alpha
                )
            
            # Inicializar generador
            with TimingContext("Generador"):
                self.generator = ResponseGenerator(
                    model_name=self.config.models.llm_model,
                    max_new_tokens=self.config.generation.max_new_tokens,
                    temperature=self.config.generation.temperature
                )
            
            # Inicializar verificador si está habilitado
            if self.config.verification.enable_verification:
                with TimingContext("Verificador"):
                    self.verifier = FidelityVerifier(
                        threshold=self.config.verification.verification_threshold
                    )
            
            logger.success("✓ Todos los componentes cargados")
    
    def build_index(self, documents_path: Optional[Path] = None) -> None:
        """
        Construye los índices desde cero
        
        Args:
            documents_path: Ruta a la carpeta con documentos (PDFs/HTML)
        
        Raises:
            RuntimeError: Si hay errores procesando documentos
        """
        if documents_path is None:
            documents_path = self.config.paths.data_raw_path
        
        logger.info(f"Construyendo índices desde: {documents_path}")
        
        with TimingContext("Construcción completa de índices"):
            # Inicializar preprocessor
            self.preprocessor = DocumentPreprocessor(
                chunk_size=self.config.chunking.chunk_size,
                chunk_overlap=self.config.chunking.chunk_overlap
            )
            
            # Procesar todos los documentos
            all_chunks = []
            pdf_files = list(documents_path.glob('**/*.pdf'))
            html_files = list(documents_path.glob('**/*.html'))
            all_files = pdf_files + html_files
            
            if not all_files:
                logger.warning(f"No se encontraron documentos (PDF/HTML) en {documents_path}")
                return
            
            logger.info(f"Encontrados {len(all_files)} documentos (PDFs: {len(pdf_files)}, HTML: {len(html_files)})")
            
            skipped_count = 0
            processed_count = 0
            
            for doc_file in all_files:
                try:
                    logger.info(f"Procesando: {doc_file.name}")
                    
                    # Metadatos básicos (en producción, cargar de metadata.json)
                    metadata = {
                        'title': doc_file.stem,
                        'filename': doc_file.name,
                    }
                    
                    chunks = self.preprocessor.process_document(doc_file, metadata)
                    
                    if chunks:
                        all_chunks.extend(chunks)
                        processed_count += 1
                    else:
                        skipped_count += 1
                    
                except Exception as e:
                    logger.error(f"Error procesando {doc_file.name}: {e}")
                    skipped_count += 1
                    continue
            
            logger.info(f"Documentos procesados: {processed_count}, omitidos: {skipped_count}")
            
            if not all_chunks:
                raise RuntimeError("No se generaron chunks para indexar")
            
            logger.info(f"Total de chunks procesados: {len(all_chunks)}")
            
            # Inicializar e indexar
            with TimingContext("Indexación"):
                self.indexer = DocumentIndexer(
                    embedding_model=self.config.models.embedding_model,
                    faiss_index_path=self.config.paths.faiss_index_path,
                    bm25_index_path=self.config.paths.bm25_index_path
                )
                
                self.indexer.index_chunks(all_chunks)
                self.indexer.save_indices()
            
            # Inicializar retriever
            self.retriever = HybridRetriever(
                indexer=self.indexer,
                reranker_model=self.config.models.reranker_model,
                reranker_type=self.config.models.reranker_type,
                use_cross_encoder=self.config.models.use_cross_encoder,
                alpha=self.config.retrieval.hybrid_alpha
            )
            
            # Inicializar generador
            self.generator = ResponseGenerator(
                model_name=self.config.models.llm_model,
                max_new_tokens=self.config.generation.max_new_tokens,
                temperature=self.config.generation.temperature
            )
            
            # Inicializar verificador
            if self.config.verification.enable_verification:
                self.verifier = FidelityVerifier(
                    threshold=self.config.verification.verification_threshold
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
        
        Raises:
            ValueError: Si la pregunta está vacía
            RuntimeError: Si el pipeline no está inicializado
        """
        # Validación de entrada
        if not question or not question.strip():
            raise ValueError("La pregunta no puede estar vacía")
        
        if self.retriever is None or self.generator is None:
            raise RuntimeError(
                "El pipeline no está inicializado. "
                "Ejecuta build_index() o asegúrate de que existen índices."
            )
        
        logger.info(f"📝 Pregunta: {question}")
        
        # Parámetros
        if top_k is None:
            top_k = self.config.retrieval.top_k_rerank
        if include_verification is None:
            include_verification = self.config.verification.enable_verification
        
        timing = {}
        
        # 1. Recuperación con umbral de calidad
        with TimingContext("Recuperación", log=False) as timer:
            contexts = self.retriever.retrieve_with_threshold(
                query=question,
                top_k_retrieval=self.config.retrieval.top_k_retrieval,
                top_k_final=top_k,
                min_score=self.config.retrieval.min_score_threshold
            )
        timing['retrieval'] = timer.elapsed
        logger.info(f"⏱️  Recuperación: {timer.elapsed:.2f}s")
        
        # 2. Abstención temprana si no hay contextos de calidad suficiente
        if len(contexts) == 0:
            logger.warning("No hay contextos que superen el umbral de calidad → Abstención")
            return {
                'answer': "No dispongo de información disponible sobre esta consulta en la normativa.",
                'sources': [],
                'contexts': [],
                'metadata': {
                    'abstention_reason': 'no_quality_contexts',
                    'min_score_threshold': self.config.retrieval.min_score_threshold
                },
                'timing': timing
            }
        
        # Si solo hay 1 contexto con score muy bajo, también abstenerse
        if len(contexts) == 1 and contexts[0][1] < (self.config.retrieval.min_score_threshold + 0.1):
            logger.warning(f"Solo 1 contexto con score bajo ({contexts[0][1]:.3f}) → Abstención")
            return {
                'answer': "No dispongo de información suficiente para responder con certeza a esta consulta.",
                'sources': [],
                'contexts': contexts,
                'metadata': {
                    'abstention_reason': 'single_low_quality_context',
                    'context_score': contexts[0][1]
                },
                'timing': timing
            }
        
        logger.info(f"✓ {len(contexts)} contextos de calidad para generación")
        
        # 3. Generación
        with TimingContext("Generación", log=False) as timer:
            response = self.generator.generate(question, contexts)
        timing['generation'] = timer.elapsed
        logger.info(f"⏱️  Generación: {timer.elapsed:.2f}s")
        
        # 4. Verificación (opcional)
        timing['verification'] = 0
        if include_verification and self.verifier and contexts:
            with TimingContext("Verificación", log=False) as timer:
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
            
            timing['verification'] = timer.elapsed
            logger.info(f"⏱️  Verificación: {timer.elapsed:.2f}s")
        
        # Tiempo total
        timing['total'] = sum(timing.values())
        response['timing'] = timing
        
        logger.success(f"✓ Respuesta generada - Tiempo total: {timing['total']:.2f}s")
        
        return response
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del sistema"""
        stats = {
            'config': self.config.to_dict(),
            'status': {
                'indexer': self.indexer is not None,
                'retriever': self.retriever is not None,
                'generator': self.generator is not None,
                'verifier': self.verifier is not None,
            }
        }
        
        if self.indexer:
            stats['index'] = self.indexer.get_stats()
        else:
            stats['index'] = {'status': 'No inicializado'}
        
        return stats
    
    def cleanup(self) -> None:
        """Libera recursos y memoria"""
        logger.info("Liberando recursos del pipeline...")
        
        with torch_memory_cleanup():
            self.generator = None
            self.retriever = None
            self.verifier = None
            self.indexer = None
            self.preprocessor = None
        
        logger.success("✓ Recursos liberados")
    
    def __enter__(self):
        """Context manager support"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Limpieza automática al salir del contexto"""
        self.cleanup()


if __name__ == "__main__":
    # Ejemplo de uso con context manager
    with RAGPipeline() as rag:
        # Hacer consultas
        # result = rag.query("¿Cuál es el plazo para presentar el TFM?")
        # print(result['answer'])
        pass
    # Recursos liberados automáticamente
    
    print("✓ Pipeline RAG-UCM listo para usar")
