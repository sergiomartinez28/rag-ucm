"""
Script para analizar y validar el dataset de evaluación
Identifica preguntas problemáticas o fuera del scope
"""

import json
from pathlib import Path
from typing import List, Dict
from collections import defaultdict

def load_results(path: Path) -> List[Dict]:
    """Carga los resultados de evaluación"""
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)

def analyze_dataset(results: List[Dict]) -> Dict:
    """Analiza el dataset para identificar problemas"""
    
    analysis = {
        'total': len(results),
        'out_of_scope': [],
        'empty_answers': [],
        'missing_references': [],
        'by_category': defaultdict(int),
        'by_type': defaultdict(int),
        'quality_issues': []
    }
    
    # Keywords para detectar preguntas out-of-scope (fuera de la UCM)
    out_of_scope_keywords = [
        'tratado', 'unión europea', 'directiva',
        'consejo', 'parlamento europeo',
        'reglamento ue', 'normativa europea'
    ]
    
    for item in results:
        q_id = item['id']
        question = item['question'].lower()
        ref_answer = item.get('reference_answer', '').strip()
        rag_answer = item.get('rag_answer', '').strip()
        
        # Contar por categoría y tipo
        analysis['by_category'][item.get('category', 'unknown')] += 1
        analysis['by_type'][item.get('question_type', 'unknown')] += 1
        
        # Detectar preguntas out-of-scope
        if any(kw in question for kw in out_of_scope_keywords):
            analysis['out_of_scope'].append({
                'id': q_id,
                'question': item['question'][:80] + '...'
            })
            analysis['quality_issues'].append(f"Q{q_id}: Out-of-scope (referencias europeas)")
        
        # Detectar respuestas RAG vacías
        if len(rag_answer) < 10:
            analysis['empty_answers'].append({
                'id': q_id,
                'question': item['question'][:80] + '...',
                'answer': rag_answer[:50] if rag_answer else '[VACÍO]'
            })
        
        # Detectar referencias faltantes
        if not ref_answer or len(ref_answer) < 5:
            analysis['missing_references'].append({
                'id': q_id,
                'question': item['question'][:80] + '...'
            })
            analysis['quality_issues'].append(f"Q{q_id}: Referencia incompleta o ausente")
        
        # Detectar inconsistencias
        if item.get('correct_doc_in_top_k') and len(rag_answer) < 10:
            analysis['quality_issues'].append(
                f"Q{q_id}: Doc correcto recuperado pero RAG no responde"
            )
    
    return analysis

def print_analysis(analysis: Dict):
    """Imprime un reporte del análisis"""
    print("\n" + "="*70)
    print("ANÁLISIS DEL DATASET DE EVALUACIÓN")
    print("="*70)
    
    print(f"\n📊 ESTADÍSTICAS GENERALES:")
    print(f"  Total de preguntas: {analysis['total']}")
    print(f"  Preguntas out-of-scope: {len(analysis['out_of_scope'])}")
    print(f"  Respuestas RAG vacías: {len(analysis['empty_answers'])}")
    print(f"  Referencias incompletas: {len(analysis['missing_references'])}")
    
    print(f"\n📂 DISTRIBUCIÓN POR CATEGORÍA:")
    for cat, count in sorted(analysis['by_category'].items()):
        print(f"  • {cat}: {count}")
    
    print(f"\n❓ DISTRIBUCIÓN POR TIPO:")
    for typ, count in sorted(analysis['by_type'].items()):
        print(f"  • {typ}: {count}")
    
    if analysis['out_of_scope']:
        print(f"\n⚠️  PREGUNTAS OUT-OF-SCOPE (deben removerse):")
        for item in analysis['out_of_scope']:
            print(f"  [{item['id']}] {item['question']}")
    
    if analysis['empty_answers']:
        print(f"\n❌ RESPUESTAS RAG VACÍAS ({len(analysis['empty_answers'])}):")
        for item in analysis['empty_answers'][:5]:  # Mostrar solo las primeras 5
            print(f"  [{item['id']}] {item['question']}")
            print(f"       Respuesta: {item['answer']}")
    
    if analysis['missing_references']:
        print(f"\n❓ REFERENCIAS INCOMPLETAS ({len(analysis['missing_references'])}):")
        for item in analysis['missing_references'][:5]:
            print(f"  [{item['id']}] {item['question']}")
    
    if analysis['quality_issues']:
        print(f"\n🔴 PROBLEMAS DE CALIDAD ({len(analysis['quality_issues'])}):")
        for issue in analysis['quality_issues'][:10]:
            print(f"  • {issue}")
    
    print("\n" + "="*70)
    print("RECOMENDACIONES:")
    print("="*70)
    print("1. REMOVER: Preguntas out-of-scope (referencias europeas)")
    print("2. VALIDAR: Respuestas de referencia vacías")
    print("3. REVISAR: Casos donde se recupera doc correcto pero no hay respuesta")
    print("="*70 + "\n")

def generate_clean_dataset(results: List[Dict], output_path: Path):
    """Genera un dataset limpio removiendo preguntas problemáticas"""
    
    out_of_scope_keywords = [
        'tratado', 'unión europea', 'directiva',
        'consejo', 'parlamento europeo',
        'reglamento ue', 'normativa europea'
    ]
    
    cleaned = []
    removed = []
    
    for item in results:
        question = item['question'].lower()
        
        # Remover si es out-of-scope
        if any(kw in question for kw in out_of_scope_keywords):
            removed.append(item['id'])
            continue
        
        # Remover si no tiene referencia
        ref_answer = item.get('reference_answer', '').strip()
        if not ref_answer or len(ref_answer) < 5:
            removed.append(item['id'])
            continue
        
        cleaned.append(item)
    
    # Guardar dataset limpio
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    
    print(f"\n✅ Dataset limpio guardado en: {output_path}")
    print(f"  Original: {len(results)} preguntas")
    print(f"  Limpio: {len(cleaned)} preguntas")
    print(f"  Removidas: {len(removed)} preguntas (IDs: {removed})")

if __name__ == "__main__":
    # Cargar y analizar
    results_path = Path("data/evaluation/results.json")
    
    if results_path.exists():
        results = load_results(results_path)
        analysis = analyze_dataset(results)
        print_analysis(analysis)
        
        # Generar dataset limpio
        clean_path = Path("data/evaluation/dataset_clean.json")
        generate_clean_dataset(results, clean_path)
    else:
        print(f"❌ No se encontró: {results_path}")
