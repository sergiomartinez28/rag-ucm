"""
Calculador de métricas agregadas para la evaluación RAG
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict

from loguru import logger


@dataclass
class AggregatedMetrics:
    """Métricas agregadas de la evaluación"""
    # Retrieval
    precision_at_k: float
    mrr: float  # Mean Reciprocal Rank
    
    # Generation (promedios) - 3 métricas simples
    avg_relevancia: float    # ¿Responde a la pregunta?
    avg_fidelidad: float     # ¿Está basado en documentos?
    avg_precision: float     # ¿Es correcto?
    avg_overall: float
    
    # Tiempo
    avg_retrieval_time: float
    avg_generation_time: float
    avg_total_time: float
    
    # Conteos
    total_questions: int
    questions_with_correct_doc: int
    
    # Por categoría
    metrics_by_category: Dict[str, Dict]
    metrics_by_question_type: Dict[str, Dict]


class MetricsCalculator:
    """
    Calcula métricas agregadas desde los resultados de evaluación
    """
    
    def calculate_all_metrics(
        self,
        results_path: Path,
        scores_path: Path,
        output_path: Optional[Path] = None
    ) -> AggregatedMetrics:
        """
        Calcula todas las métricas agregadas
        
        Args:
            results_path: Ruta a resultados de evaluación RAG
            scores_path: Ruta a puntuaciones del LLM Juez
            output_path: Ruta para guardar métricas (opcional)
        
        Returns:
            AggregatedMetrics
        """
        logger.info("Calculando métricas agregadas...")
        
        with open(results_path, 'r', encoding='utf-8') as f:
            results = json.load(f)
        
        with open(scores_path, 'r', encoding='utf-8') as f:
            scores = json.load(f)
        
        # Crear lookup de scores por ID
        scores_by_id = {s['id']: s for s in scores}
        
        n = len(results)
        
        # Métricas de retrieval
        correct_docs = sum(1 for r in results if r['correct_doc_in_top_k'])
        precision_at_k = correct_docs / n if n > 0 else 0
        
        mrr_sum = sum(1/r['correct_doc_rank'] for r in results if r['correct_doc_rank'] > 0)
        mrr = mrr_sum / n if n > 0 else 0
        
        # Métricas de tiempo
        avg_retrieval_time = sum(r['retrieval_time'] for r in results) / n if n > 0 else 0
        avg_generation_time = sum(r['generation_time'] for r in results) / n if n > 0 else 0
        avg_total_time = sum(r['total_time'] for r in results) / n if n > 0 else 0
        
        # Métricas de generación (del LLM Juez) - 3 métricas simples
        avg_relevancia = sum(s.get('relevancia', 3) for s in scores) / n if n > 0 else 0
        avg_fidelidad = sum(s.get('fidelidad', 3) for s in scores) / n if n > 0 else 0
        avg_precision = sum(s.get('precision', 3) for s in scores) / n if n > 0 else 0
        avg_overall = sum(s['overall_score'] for s in scores) / n if n > 0 else 0
        
        # Métricas por categoría
        metrics_by_category = self._calculate_by_group(results, scores_by_id, 'category')
        metrics_by_question_type = self._calculate_by_group(results, scores_by_id, 'question_type')
        
        metrics = AggregatedMetrics(
            precision_at_k=precision_at_k,
            mrr=mrr,
            avg_relevancia=avg_relevancia,
            avg_fidelidad=avg_fidelidad,
            avg_precision=avg_precision,
            avg_overall=avg_overall,
            avg_retrieval_time=avg_retrieval_time,
            avg_generation_time=avg_generation_time,
            avg_total_time=avg_total_time,
            total_questions=n,
            questions_with_correct_doc=correct_docs,
            metrics_by_category=metrics_by_category,
            metrics_by_question_type=metrics_by_question_type
        )
        
        if output_path:
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(asdict(metrics), f, ensure_ascii=False, indent=2)
            logger.success(f"✓ Métricas guardadas en {output_path}")
        
        return metrics
    
    def _calculate_by_group(
        self,
        results: List[Dict],
        scores_by_id: Dict[int, Dict],
        group_key: str
    ) -> Dict[str, Dict]:
        """Calcula métricas agrupadas"""
        groups = defaultdict(list)
        
        for r in results:
            group = r.get(group_key, 'unknown')
            score = scores_by_id.get(r['id'], {})
            groups[group].append({
                'result': r,
                'score': score
            })
        
        metrics = {}
        for group, items in groups.items():
            n = len(items)
            if n == 0:
                continue
            
            metrics[group] = {
                'count': n,
                'precision_at_k': sum(1 for i in items if i['result']['correct_doc_in_top_k']) / n,
                'avg_relevancia': sum(i['score'].get('relevancia', 0) for i in items) / n,
                'avg_fidelidad': sum(i['score'].get('fidelidad', 0) for i in items) / n,
                'avg_precision': sum(i['score'].get('precision', 0) for i in items) / n,
                'avg_overall': sum(i['score'].get('overall_score', 0) for i in items) / n,
                'avg_time': sum(i['result']['total_time'] for i in items) / n,
            }
        
        return dict(metrics)
    
    def print_report(self, metrics: AggregatedMetrics) -> None:
        """Imprime un reporte formateado de las métricas"""
        print("\n" + "="*70)
        print("            EVALUACIÓN RAG-UCM - RESUMEN EJECUTIVO")
        print("="*70)
        
        print(f"\n📊 DATASET: {metrics.total_questions} preguntas evaluadas")
        
        print("\n┌" + "─"*68 + "┐")
        print("│                    MÉTRICAS DE RETRIEVAL                           │")
        print("├" + "─"*68 + "┤")
        print(f"│  Precision@K:        {metrics.precision_at_k:.2%} ({metrics.questions_with_correct_doc}/{metrics.total_questions} docs correctos)".ljust(69) + "│")
        print(f"│  MRR:                {metrics.mrr:.3f}".ljust(69) + "│")
        print(f"│  Avg Retrieval Time: {metrics.avg_retrieval_time:.2f}s".ljust(69) + "│")
        print("└" + "─"*68 + "┘")
        
        print("\n┌" + "─"*68 + "┐")
        print("│                   MÉTRICAS DE GENERACIÓN                           │")
        print("├" + "─"*68 + "┤")
        stars = lambda x: "⭐" * int(round(x))
        print(f"│  Relevancia:        {metrics.avg_relevancia:.2f}/5  {stars(metrics.avg_relevancia)}".ljust(69) + "│")
        print(f"│  Fidelidad:         {metrics.avg_fidelidad:.2f}/5  {stars(metrics.avg_fidelidad)}".ljust(69) + "│")
        print(f"│  Precisión:         {metrics.avg_precision:.2f}/5  {stars(metrics.avg_precision)}".ljust(69) + "│")
        print("├" + "─"*68 + "┤")
        print(f"│  Overall Score:     {metrics.avg_overall:.2f}/5  {stars(metrics.avg_overall)}".ljust(69) + "│")
        print("└" + "─"*68 + "┘")
        
        print("\n┌" + "─"*68 + "┐")
        print("│                    MÉTRICAS DE TIEMPO                              │")
        print("├" + "─"*68 + "┤")
        print(f"│  Retrieval:          {metrics.avg_retrieval_time:.2f}s".ljust(69) + "│")
        print(f"│  Generation:         {metrics.avg_generation_time:.2f}s".ljust(69) + "│")
        print(f"│  Total:              {metrics.avg_total_time:.2f}s".ljust(69) + "│")
        print("└" + "─"*68 + "┘")
        
        if metrics.metrics_by_category:
            print("\n📈 RENDIMIENTO POR CATEGORÍA:")
            for cat, m in metrics.metrics_by_category.items():
                print(f"  • {cat}: {m['avg_overall']:.2f}/5 (n={m['count']})")
        
        if metrics.metrics_by_question_type:
            print("\n📈 RENDIMIENTO POR TIPO DE PREGUNTA:")
            for qtype, m in metrics.metrics_by_question_type.items():
                print(f"  • {qtype}: {m['avg_overall']:.2f}/5 (n={m['count']})")
        
        print("\n" + "="*70)


if __name__ == "__main__":
    from pathlib import Path
    
    calculator = MetricsCalculator()
    
    metrics = calculator.calculate_all_metrics(
        results_path=Path("data/evaluation/results_test.json"),
        scores_path=Path("data/evaluation/metrics_test.json"),
        output_path=Path("data/evaluation/aggregated_metrics.json")
    )
    
    calculator.print_report(metrics)
