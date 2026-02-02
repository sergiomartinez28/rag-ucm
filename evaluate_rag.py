#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Script principal para evaluar el RAG

Comandos:
  python evaluate_rag.py generate --num-samples 50    # Generar dataset (una vez)
  python evaluate_rag.py evaluate                     # Evaluar con dataset existente
  python evaluate_rag.py full --num-samples 50        # Generar + Evaluar
"""

import argparse
import sys
from pathlib import Path
import time
import json

# Añadir el directorio raíz al path
sys.path.insert(0, str(Path(__file__).parent))

from loguru import logger


def cmd_generate(args):
    """Genera el dataset de evaluación (Q&A ground truth)"""
    import torch
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    chunks_path = Path("data/processed/faiss_index/chunks.pkl")
    dataset_path = output_dir / "dataset.json"
    
    print("\n" + "="*70)
    print("         📝 GENERADOR DE DATASET - Ground Truth")
    print("="*70)
    print(f"\n📋 Configuración:")
    print(f"   • Chunks a muestrear: {args.num_samples}")
    print(f"   • Modelo generador: {args.generator_model}")
    print(f"   • Seed: {args.seed}")
    print(f"   • Output: {dataset_path}")
    
    # Verificar si ya existe dataset
    if dataset_path.exists() and not args.force:
        with open(dataset_path, 'r', encoding='utf-8') as f:
            existing = json.load(f)
        print(f"\n⚠️  Ya existe un dataset con {len(existing)} preguntas.")
        print(f"   Usa --force para regenerar o 'evaluate' para usar el existente.")
        return
    
    start_time = time.time()
    
    print("\n" + "-"*70)
    print("🔄 Generando preguntas desde chunks...")
    print("-"*70)
    
    from src.evaluator.dataset_generator import DatasetGenerator
    
    generator = DatasetGenerator(
        model_name=args.generator_model
    )
    
    dataset = generator.generate_dataset(
        chunks_path=chunks_path,
        output_path=dataset_path,
        num_samples=args.num_samples,
        random_seed=args.seed,
        max_workers=args.workers
    )
    
    # Liberar memoria
    del generator
    torch.cuda.empty_cache()
    
    elapsed = time.time() - start_time
    print(f"\n" + "="*70)
    print(f"✅ DATASET GENERADO EXITOSAMENTE")
    print(f"="*70)
    print(f"   • Total preguntas: {len(dataset)}")
    print(f"   • Archivo: {dataset_path}")
    print(f"   • Tiempo: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    print(f"\n💡 Ahora puedes ejecutar: python evaluate_rag.py evaluate")


def cmd_evaluate(args):
    """Ejecuta la evaluación usando un dataset existente"""
    import torch
    
    output_dir = Path(args.output_dir)
    
    # Intentar cargar dataset limpio primero
    if args.clean:
        dataset_path = output_dir / "dataset_clean.json"
        if not dataset_path.exists():
            print(f"\n⚠️  No se encontró dataset limpio, usando el dataset regular")
            dataset_path = output_dir / "dataset.json"
    else:
        dataset_path = output_dir / "dataset.json"
    
    results_path = output_dir / "results.json"
    scores_path = output_dir / "scores.json"
    metrics_path = output_dir / "aggregated_metrics.json"
    
    # Verificar que existe el dataset
    if not dataset_path.exists():
        print(f"\n❌ ERROR: No existe dataset en {dataset_path}")
        print(f"   Primero genera uno con: python evaluate_rag.py generate --num-samples 50")
        sys.exit(1)
    
    # Cargar dataset para mostrar info
    with open(dataset_path, 'r', encoding='utf-8') as f:
        dataset = json.load(f)
    
    # Aplicar límite si se especifica
    if args.limit and args.limit < len(dataset):
        dataset = dataset[:args.limit]
        # Guardar versión limitada temporalmente
        limited_dataset_path = output_dir / "dataset_limited.json"
        with open(limited_dataset_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        dataset_path = limited_dataset_path
        print(f"\n⚡ Modo prueba: usando {args.limit} preguntas")
    
    print("\n" + "="*70)
    print("         🎓 EVALUADOR RAG-UCM")
    print("="*70)
    print(f"\n📋 Configuración:")
    print(f"   • Dataset: {len(dataset)} preguntas")
    print(f"   • Modelo RAG: Qwen/Qwen2.5-3B-Instruct")
    print(f"   • Modelo Juez: {args.judge_model}")
    print(f"   • Top-K: {args.top_k}")
    
    total_start = time.time()
    
    # =========================================================================
    # FASE 1: Ejecución del RAG
    # =========================================================================
    if not args.skip_rag:
        print("\n" + "-"*70)
        print("🔍 FASE 1: Ejecutando RAG con preguntas del dataset...")
        print("-"*70)
        
        from src.evaluator.rag_evaluator import RAGEvaluator
        
        evaluator = RAGEvaluator()
        
        results = evaluator.run_evaluation(
            dataset_path=dataset_path,
            output_path=results_path,
            top_k=args.top_k,
            max_workers=args.rag_workers
        )
        
        print(f"\n✅ RAG ejecutado: {len(results)} respuestas generadas")
        
        # Liberar memoria
        del evaluator
        torch.cuda.empty_cache()
    else:
        print("\n⏭️  Saltando ejecución RAG (usando resultados existentes)")
    
    # =========================================================================
    # FASE 2: Evaluación con LLM Juez
    # =========================================================================
    if not args.skip_judge:
        print("\n" + "-"*70)
        print("⚖️  FASE 2: Evaluando respuestas con LLM Juez...")
        print("-"*70)
        
        from src.evaluator.llm_judge import LLMJudge
        
        judge = LLMJudge(
            model_name=args.judge_model
        )
        
        scores = judge.evaluate_all(
            results_path=results_path,
            output_path=scores_path,
            max_workers=args.judge_workers
        )
        
        print(f"\n✅ Evaluación completada: {len(scores)} puntuaciones")
        
        # Liberar memoria
        del judge
        torch.cuda.empty_cache()
    else:
        print("\n⏭️  Saltando evaluación LLM Juez")
    
    # =========================================================================
    # FASE 3: Cálculo de Métricas Agregadas
    # =========================================================================
    print("\n" + "-"*70)
    print("📊 FASE 3: Calculando métricas agregadas...")
    print("-"*70)
    
    from src.evaluator.metrics import MetricsCalculator
    
    calculator = MetricsCalculator()
    
    metrics = calculator.calculate_all_metrics(
        results_path=results_path,
        scores_path=scores_path,
        output_path=metrics_path
    )
    
    # Imprimir reporte final
    calculator.print_report(metrics)
    
    total_time = time.time() - total_start
    print(f"\n⏱️  Tiempo total de evaluación: {total_time:.1f}s ({total_time/60:.1f} min)")
    
    # Guardar siempre con fecha
    from datetime import datetime
    import shutil
    
    date_str = datetime.now().strftime("%Y-%m-%d_%H%M")
    history_dir = output_dir / "history" / date_str
    history_dir.mkdir(parents=True, exist_ok=True)
    
    for file in [results_path, scores_path, metrics_path]:
        if file.exists():
            shutil.copy(file, history_dir / file.name)
    
    print(f"\n📁 Resultados guardados en:")
    print(f"   • Actual: {output_dir}")
    print(f"   • Historial: {history_dir}")
    
    return metrics


def cmd_full(args):
    """Ejecuta el pipeline completo: generar + evaluar"""
    # Usar el valor de --force que pasó el usuario (ya viene en args)
    cmd_generate(args)
    
    # Ahora evaluar
    cmd_evaluate(args)


def main():
    parser = argparse.ArgumentParser(
        description="🎓 Evaluador RAG-UCM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python evaluate_rag.py generate --num-samples 50    # Generar dataset (solo una vez)
  python evaluate_rag.py evaluate                     # Evaluar (siempre guarda historial)
  python evaluate_rag.py full --num-samples 50        # Pipeline completo
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Comandos disponibles')
    
    # -------------------------------------------------------------------------
    # Comando: generate
    # -------------------------------------------------------------------------
    gen_parser = subparsers.add_parser(
        'generate',
        help='Genera el dataset de evaluación (Q&A ground truth)'
    )
    gen_parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=50,
        help='Número de chunks a muestrear (default: 50, genera ~100 preguntas)'
    )
    gen_parser.add_argument(
        '--generator-model',
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help='Modelo para generar ground truth (default: Qwen2.5-3B)'
    )
    gen_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad (default: 42)'
    )
    gen_parser.add_argument(
        '--output-dir',
        type=str,
        default="data/evaluation",
        help='Directorio para guardar dataset'
    )
    gen_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Forzar regeneración aunque exista dataset'
    )
    gen_parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Número de workers paralelos para generación (default: 4)'
    )
    gen_parser.set_defaults(func=cmd_generate)
    
    # -------------------------------------------------------------------------
    # Comando: evaluate
    # -------------------------------------------------------------------------
    eval_parser = subparsers.add_parser(
        'evaluate',
        help='Ejecuta la evaluación usando el dataset existente'
    )
    eval_parser.add_argument(
        '--judge-model',
        type=str,
        default="microsoft/Phi-3-mini-4k-instruct",
        help='Modelo juez para evaluar respuestas (default: Phi-3-mini-4k-instruct)'
    )
    eval_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Número de documentos a recuperar (default: 5)'
    )
    eval_parser.add_argument(
        '--skip-rag',
        action='store_true',
        help='Saltar ejecución RAG (usar resultados existentes)'
    )
    eval_parser.add_argument(
        '--skip-judge',
        action='store_true',
        help='Saltar evaluación con LLM Juez'
    )
    eval_parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limitar evaluación a N preguntas (para pruebas rápidas)'
    )
    eval_parser.add_argument(
        '--clean',
        action='store_true',
        help='Usar dataset limpio sin preguntas out-of-scope'
    )
    eval_parser.add_argument(
        '--output-dir',
        type=str,
        default="data/evaluation",
        help='Directorio con dataset y resultados'
    )
    eval_parser.add_argument(
        '--rag-workers',
        type=int,
        default=2,
        help='Workers paralelos para evaluación RAG (default: 2, cuidado con GPU)'
    )
    eval_parser.add_argument(
        '--judge-workers',
        type=int,
        default=3,
        help='Workers paralelos para LLM Juez (default: 3)'
    )
    eval_parser.set_defaults(func=cmd_evaluate)
    
    # -------------------------------------------------------------------------
    # Comando: full (generar + evaluar)
    # -------------------------------------------------------------------------
    full_parser = subparsers.add_parser(
        'full',
        help='Pipeline completo: generar dataset + evaluar'
    )
    full_parser.add_argument(
        '--num-samples', '-n',
        type=int,
        default=50,
        help='Número de chunks a muestrear (default: 50)'
    )
    full_parser.add_argument(
        '--force', '-f',
        action='store_true',
        help='Regenerar dataset aunque ya exista'
    )
    full_parser.add_argument(
        '--generator-model',
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help='Modelo para generar ground truth'
    )
    full_parser.add_argument(
        '--judge-model',
        type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help='Modelo juez para evaluar'
    )
    full_parser.add_argument(
        '--top-k',
        type=int,
        default=5,
        help='Documentos a recuperar (default: 5)'
    )
    full_parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Semilla para reproducibilidad'
    )
    full_parser.add_argument(
        '--output-dir',
        type=str,
        default="data/evaluation",
        help='Directorio de salida'
    )
    full_parser.add_argument(
        '--skip-rag',
        action='store_true',
        help='Saltar ejecución RAG'
    )
    full_parser.add_argument(
        '--skip-judge',
        action='store_true',
        help='Saltar LLM Juez'
    )
    full_parser.add_argument(
        '--workers',
        type=int,
        default=4,
        help='Workers para generación (default: 4)'
    )
    full_parser.add_argument(
        '--rag-workers',
        type=int,
        default=2,
        help='Workers para evaluación RAG (default: 2)'
    )
    full_parser.add_argument(
        '--judge-workers',
        type=int,
        default=3,
        help='Workers para LLM Juez (default: 3)'
    )
    full_parser.add_argument(
        '--limit',
        type=int,
        default=None,
        help='Limitar evaluación a N preguntas (para pruebas)'
    )
    full_parser.add_argument(
        '--clean',
        action='store_true',
        help='Usar dataset limpio sin preguntas out-of-scope'
    )
    full_parser.set_defaults(func=cmd_full)
    
    # Parsear argumentos
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        print("\n❌ Debes especificar un comando: generate, evaluate, o full")
        sys.exit(1)
    
    # Ejecutar comando
    args.func(args)


if __name__ == "__main__":
    main()
