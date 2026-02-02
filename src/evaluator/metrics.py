"""
Calculador de métricas agregadas para la evaluación RAG
"""

import json
import re
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
    precision_at_k_chunk: float  # Nuevo: chunk-level precision
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
    questions_with_correct_chunk: int  # Nuevo
    
    # Métricas automáticas (sin LLM juez)
    abstention_rate: float          # % de abstenciones
    suspicious_short_rate: float    # % de respuestas sospechosamente cortas
    factual_accuracy: float         # Accuracy en preguntas factuales (números/fechas)
    abstention_when_reference_exists: float  # % de abstenciones incorrectas
    
    # Por categoría
    metrics_by_category: Dict[str, Dict]
    metrics_by_question_type: Dict[str, Dict]


class MetricsCalculator:
    """
    Calcula métricas agregadas desde los resultados de evaluación
    """
    
    ABSTENTION_PATTERNS = [
        "no dispongo de información",
        "no dispongo de la información", 
        "no tengo información",
        "no encuentro información",
        "no hay información",
        "no puedo responder",
        "no es posible responder",
        "información no disponible",
        "sin información disponible",
        "no se especifica",
        "no se menciona"
    ]
    
    def _is_abstention(self, answer: str) -> bool:
        """Detecta si la respuesta es una abstención"""
        answer_lower = answer.lower().strip()
        return any(pattern in answer_lower for pattern in self.ABSTENTION_PATTERNS)
    
    def _is_suspicious_short(self, answer: str) -> bool:
        """Detecta respuestas sospechosamente cortas"""
        clean = answer.strip()
        if len(clean) <= 2:
            return True
        if clean.isdigit():
            return True
        return False
    
    def _extract_factual_data(self, text: str) -> set:
        """Extrae números, fechas y datos factuales de un texto"""
        facts = set()
        
        # Números (incluyendo decimales y porcentajes)
        numbers = re.findall(r'\b\d+(?:[.,]\d+)?%?\b', text)
        facts.update(numbers)
        
        # Fechas en formato DD/MM/YYYY o similar
        dates = re.findall(r'\b\d{1,2}[/-]\d{1,2}[/-]\d{2,4}\b', text)
        facts.update(dates)
        
        # Plazos comunes: "X días", "X meses", "X años", "X créditos"
        plazos = re.findall(r'\b(\d+)\s*(?:días?|meses?|años?|créditos?)\b', text.lower())
        facts.update(plazos)
        
        return facts
    
    def _check_factual_accuracy(self, rag_answer: str, reference: str, question_type: str) -> Optional[bool]:
        """
        Verifica si los datos factuales de la referencia aparecen en la respuesta.
        Solo aplica a preguntas factuales.
        Returns: True si coincide, False si no, None si no aplica.
        """
        if question_type != 'factual':
            return None
        
        # Extraer datos factuales de la referencia
        ref_facts = self._extract_factual_data(reference)
        if not ref_facts:
            return None  # No hay datos factuales para verificar
        
        # Si es abstención, es incorrecto
        if self._is_abstention(rag_answer):
            return False
        
        # Extraer datos de la respuesta
        answer_facts = self._extract_factual_data(rag_answer)
        
        # Verificar si al menos uno de los datos de referencia aparece
        intersection = ref_facts & answer_facts
        return len(intersection) > 0
    
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
        
        # Métricas de retrieval (documento y chunk)
        correct_docs = sum(1 for r in results if r['correct_doc_in_top_k'])
        correct_chunks = sum(1 for r in results if r.get('correct_chunk_in_top_k', False))
        precision_at_k = correct_docs / n if n > 0 else 0
        precision_at_k_chunk = correct_chunks / n if n > 0 else 0
        
        mrr_sum = sum(1/r['correct_doc_rank'] for r in results if r['correct_doc_rank'] > 0)
        mrr = mrr_sum / n if n > 0 else 0
        
        # Métricas de tiempo
        avg_retrieval_time = sum(r['retrieval_time'] for r in results) / n if n > 0 else 0
        avg_generation_time = sum(r['generation_time'] for r in results) / n if n > 0 else 0
        avg_total_time = sum(r['total_time'] for r in results) / n if n > 0 else 0
        
        # Métricas de generación (del LLM Juez) - 3 métricas simples
        n_scores = len(scores)
        avg_relevancia = sum(s.get('relevancia', 0.0) for s in scores) / n_scores if n_scores > 0 else 0.0
        avg_fidelidad = sum(s.get('fidelidad', 0.0) for s in scores) / n_scores if n_scores > 0 else 0.0
        avg_precision = sum(s.get('precision', 0.0) for s in scores) / n_scores if n_scores > 0 else 0.0
        avg_overall = sum(s.get('overall_score', 0.0) for s in scores) / n_scores if n_scores > 0 else 0.0
        
        # === MÉTRICAS AUTOMÁTICAS (sin LLM juez) ===
        abstentions = 0
        suspicious_short = 0
        abstention_with_ref = 0  # Abstenciones cuando hay referencia (error grave)
        factual_correct = 0
        factual_total = 0
        
        for r in results:
            answer = r.get('rag_answer', '')
            reference = r.get('reference_answer', '')
            question_type = r.get('question_type', '')
            
            # Abstención
            is_abstention = self._is_abstention(answer)
            if is_abstention:
                abstentions += 1
                # ¿Abstención cuando había referencia? (error grave)
                if len(reference.strip()) > 10:
                    abstention_with_ref += 1
            
            # Respuesta sospechosamente corta
            if self._is_suspicious_short(answer):
                suspicious_short += 1
            
            # Accuracy factual
            factual_check = self._check_factual_accuracy(answer, reference, question_type)
            if factual_check is not None:
                factual_total += 1
                if factual_check:
                    factual_correct += 1
        
        abstention_rate = abstentions / n if n > 0 else 0
        suspicious_short_rate = suspicious_short / n if n > 0 else 0
        abstention_when_reference_exists = abstention_with_ref / n if n > 0 else 0
        factual_accuracy = factual_correct / factual_total if factual_total > 0 else 0
        
        # Log de métricas automáticas
        logger.info(f"Métricas automáticas:")
        logger.info(f"  - Abstention rate: {abstention_rate:.1%} ({abstentions}/{n})")
        logger.info(f"  - Suspicious short: {suspicious_short_rate:.1%} ({suspicious_short}/{n})")
        logger.info(f"  - Abstención incorrecta: {abstention_when_reference_exists:.1%} ({abstention_with_ref}/{n})")
        logger.info(f"  - Factual accuracy: {factual_accuracy:.1%} ({factual_correct}/{factual_total})")
        
        # Métricas por categoría
        metrics_by_category = self._calculate_by_group(results, scores_by_id, 'category')
        metrics_by_question_type = self._calculate_by_group(results, scores_by_id, 'question_type')
        
        metrics = AggregatedMetrics(
            precision_at_k=precision_at_k,
            precision_at_k_chunk=precision_at_k_chunk,
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
            questions_with_correct_chunk=correct_chunks,
            abstention_rate=abstention_rate,
            suspicious_short_rate=suspicious_short_rate,
            factual_accuracy=factual_accuracy,
            abstention_when_reference_exists=abstention_when_reference_exists,
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
        print(f"│  Precision@K (doc):   {metrics.precision_at_k:.2%} ({metrics.questions_with_correct_doc}/{metrics.total_questions})".ljust(69) + "│")
        print(f"│  Precision@K (chunk): {metrics.precision_at_k_chunk:.2%} ({metrics.questions_with_correct_chunk}/{metrics.total_questions})".ljust(69) + "│")
        print(f"│  MRR:                 {metrics.mrr:.3f}".ljust(69) + "│")
        print(f"│  Avg Retrieval Time:  {metrics.avg_retrieval_time:.2f}s".ljust(69) + "│")
        print("└" + "─"*68 + "┘")
        
        print("\n┌" + "─"*68 + "┐")
        print("│                   MÉTRICAS DE GENERACIÓN                           │")
        print("├" + "─"*68 + "┤")
        stars = lambda x: "★" * int(round(x * 5)) + "☆" * (5 - int(round(x * 5)))
        print(f"│  Relevancia:        {metrics.avg_relevancia:.2f}/1  {stars(metrics.avg_relevancia)}".ljust(69) + "│")
        print(f"│  Fidelidad:         {metrics.avg_fidelidad:.2f}/1  {stars(metrics.avg_fidelidad)}".ljust(69) + "│")
        print(f"│  Precision:         {metrics.avg_precision:.2f}/1  {stars(metrics.avg_precision)}".ljust(69) + "│")
        print("├" + "─"*68 + "┤")
        print(f"│  Overall Score:     {metrics.avg_overall:.2f}/1  {stars(metrics.avg_overall)}".ljust(69) + "│")
        print("└" + "─"*68 + "┘")
        
        # Nueva sección: Métricas automáticas (señales de alerta)
        print("\n┌" + "─"*68 + "┐")
        print("│              MÉTRICAS AUTOMÁTICAS (SIN LLM JUEZ)                   │")
        print("├" + "─"*68 + "┤")
        # Abstención
        abstention_pct = metrics.abstention_rate * 100
        abstention_icon = "🔴" if abstention_pct > 50 else ("🟡" if abstention_pct > 20 else "🟢")
        print(f"│  {abstention_icon} Abstención:        {abstention_pct:.1f}%".ljust(69) + "│")
        # Abstención incorrecta
        bad_abs_pct = metrics.abstention_when_reference_exists * 100
        bad_abs_icon = "🔴" if bad_abs_pct > 30 else ("🟡" if bad_abs_pct > 10 else "🟢")
        print(f"│  {bad_abs_icon} Abstención errónea: {bad_abs_pct:.1f}% (cuando hay referencia)".ljust(69) + "│")
        # Respuestas sospechosas
        short_pct = metrics.suspicious_short_rate * 100
        short_icon = "🔴" if short_pct > 10 else ("🟡" if short_pct > 5 else "🟢")
        print(f"│  {short_icon} Respuestas cortas:  {short_pct:.1f}%".ljust(69) + "│")
        # Accuracy factual
        fact_pct = metrics.factual_accuracy * 100
        fact_icon = "🟢" if fact_pct > 70 else ("🟡" if fact_pct > 50 else "🔴")
        print(f"│  {fact_icon} Factual accuracy:   {fact_pct:.1f}%".ljust(69) + "│")
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
            for cat, m in sorted(metrics.metrics_by_category.items(), key=lambda x: -x[1]['avg_overall']):
                print(f"  • {cat}: {m['avg_overall']:.2f}/1 (n={m['count']})")
        
        if metrics.metrics_by_question_type:
            print("\n📈 RENDIMIENTO POR TIPO DE PREGUNTA:")
            for qtype, m in sorted(metrics.metrics_by_question_type.items(), key=lambda x: -x[1]['avg_overall']):
                print(f"  • {qtype}: {m['avg_overall']:.2f}/1 (n={m['count']})")
        
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
