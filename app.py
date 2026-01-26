"""
Interfaz web Streamlit para RAG-UCM
Ejecución: streamlit run app.py
"""

import streamlit as st
from pathlib import Path
from src.pipeline import RAGPipeline

# Configuración de la página
st.set_page_config(
    page_title="RAG-UCM | Asistente Académico",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #1f77b4;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    .warning-box {
        background-color: #fff3cd;
        border-left: 4px solid #ffc107;
        padding: 1rem;
        margin: 1rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_pipeline():
    """Carga el pipeline RAG (cachea en memoria)"""
    return RAGPipeline()


def main():
    # Header
    st.markdown('<div class="main-header">🎓 RAG-UCM</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="subtitle">Asistente de Consultas sobre Normativa Académica UCM</div>',
        unsafe_allow_html=True
    )
    
    # Sidebar
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Parámetros
        top_k = st.slider(
            "Documentos a recuperar",
            min_value=1,
            max_value=10,
            value=5,
            help="Número de fragmentos de documentos a usar para generar la respuesta"
        )
        
        enable_verification = st.checkbox(
            "Verificar fidelidad",
            value=True,
            help="Detecta posibles alucinaciones en la respuesta"
        )
        
        show_sources = st.checkbox(
            "Mostrar fuentes",
            value=True,
            help="Muestra los documentos consultados"
        )
        
        show_details = st.checkbox(
            "Mostrar detalles técnicos",
            value=False,
            help="Información detallada de verificación y metadata"
        )
        
        st.divider()
        
        # Información del sistema
        st.subheader("ℹ️ Acerca de")
        st.markdown("""
        Sistema RAG (Retrieval-Augmented Generation) para consultas 
        sobre normativa académica de la UCM.
        
        **Características:**
        - 🔍 Búsqueda híbrida (BM25 + embeddings)
        - 🎯 Re-ranking inteligente
        - ✅ Verificación de fidelidad
        - 📚 Citas a documentos oficiales
        - 🔓 100% Open Source
        """)
        
        # Estadísticas
        try:
            with st.spinner("Cargando sistema..."):
                rag = load_pipeline()
            
            st.divider()
            st.subheader("📊 Estadísticas")
            
            stats = rag.get_stats()
            if 'index' in stats and stats['index'].get('total_chunks'):
                idx = stats['index']
                st.metric("Documentos indexados", idx['total_chunks'])
                st.caption(f"Modelo: {idx['embedding_model'].split('/')[-1]}")
            else:
                st.warning("⚠️ No hay índices construidos")
        
        except Exception as e:
            st.error(f"Error cargando sistema: {str(e)}")
            return
    
    # Área principal
    st.divider()
    
    # Ejemplos de preguntas
    with st.expander("💡 Ejemplos de preguntas"):
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **TFG/TFM:**
            - ¿Cuál es el plazo para presentar el TFM?
            - ¿Cuántas convocatorias tengo para el TFG?
            - ¿Puedo cambiar el tema de mi TFM?
            """)
        
        with col2:
            st.markdown("""
            **Matrícula y reconocimiento:**
            - ¿Cómo convalido créditos de otra universidad?
            - ¿Cuándo es el periodo de matrícula?
            - ¿Cuántos créditos puedo reconocer?
            """)
    
    # Input de pregunta
    question = st.text_area(
        "🔍 Escribe tu pregunta:",
        height=100,
        placeholder="Ejemplo: ¿Cuándo es el plazo para presentar el TFM en la Facultad de Informática?"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        ask_button = st.button("🚀 Consultar", type="primary", use_container_width=True)
    
    with col2:
        clear_button = st.button("🗑️ Limpiar", use_container_width=True)
    
    if clear_button:
        st.rerun()
    
    # Procesar pregunta
    if ask_button and question.strip():
        try:
            with st.spinner("🔍 Buscando en la normativa..."):
                result = rag.query(
                    question=question,
                    top_k=top_k,
                    include_verification=enable_verification
                )
            
            # Mostrar respuesta
            st.success("✅ Respuesta generada")
            
            st.markdown("### 📝 Respuesta")
            st.markdown(result['answer'])
            
            # Advertencia si existe
            if 'warning' in result:
                st.markdown(
                    f'<div class="warning-box">⚠️ <strong>Atención:</strong><br>{result["warning"]}</div>',
                    unsafe_allow_html=True
                )
            
            # Fuentes
            if show_sources and result['sources']:
                st.markdown("### 📚 Fuentes Consultadas")
                
                for source in result['sources']:
                    with st.expander(
                        f"[{source['id']}] {source['title']} (relevancia: {source['score']:.3f})"
                    ):
                        st.text(source['text_preview'])
                        
                        # Metadata
                        meta = source['metadata']
                        if meta.get('faculty'):
                            st.caption(f"**Facultad:** {meta['faculty']}")
                        if meta.get('year'):
                            st.caption(f"**Año:** {meta['year']}")
                        if meta.get('url'):
                            st.caption(f"**URL:** {meta['url']}")
            
            # Detalles técnicos
            if show_details:
                st.markdown("### 🔍 Detalles Técnicos")
                
                col1, col2, col3 = st.columns(3)
                
                # Verificación
                if 'verification' in result:
                    ver = result['verification']
                    
                    with col1:
                        st.metric(
                            "Fidelidad",
                            f"{ver['fidelity_score']:.1%}",
                            delta="OK" if ver['is_faithful'] else "⚠️"
                        )
                    
                    with col2:
                        st.metric(
                            "Oraciones verificadas",
                            f"{ver['num_sentences'] - ver['num_unsupported']}/{ver['num_sentences']}"
                        )
                    
                    # Verificación de citas
                    if 'citation_check' in ver:
                        cit = ver['citation_check']
                        with col3:
                            st.metric(
                                "Cobertura de citas",
                                f"{cit['citation_coverage']:.1%}"
                            )
                
                # Metadata del modelo
                meta = result.get('metadata', {})
                
                st.caption(f"**Modelo:** {meta.get('model', 'N/A')}")
                st.caption(f"**Temperatura:** {meta.get('temperature', 'N/A')}")
                st.caption(f"**Contextos usados:** {meta.get('num_contexts', 0)}")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            st.exception(e)
    
    elif ask_button:
        st.warning("⚠️ Por favor, escribe una pregunta")
    
    # Footer
    st.divider()
    st.caption(
        "RAG-UCM v0.1.0 | Desarrollado como TFM | "
        "Universidad Complutense de Madrid | "
        "Sergio Martín © 2026"
    )


if __name__ == "__main__":
    main()
