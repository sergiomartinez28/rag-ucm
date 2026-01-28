"""
Script para comparar resultados entre iteraciones
"""

import json
from pathlib import Path
from typing import Dict, Tuple

def load_metrics(history_folder: Path) -> Dict:
    """Carga las métricas de una carpeta de historia"""
    metrics_file = history_folder / "aggregated_metrics.json"
    if metrics_file.exists():
        with open(metrics_file, 'r', encoding='utf-8') as f:
            return json.load(f)
    return None

def compare_iterations(iter1_folder: str, iter2_folder: str):
    """Compara dos iteraciones"""
    
    path1 = Path("data/evaluation/history") / iter1_folder
    path2 = Path("data/evaluation/history") / iter2_folder
    
    metrics1 = load_metrics(path1)
    metrics2 = load_metrics(path2)
    
    if not metrics1 or not metrics2:
        print("❌ No se pudieron cargar las métricas")
        return
    
    print("\n" + "="*70)
    print("COMPARACIÓN DE ITERACIONES")
    print("="*70)
    
    print(f"\nIteración 1: {iter1_folder}")
    print(f"Iteración 2: {iter2_folder}")
    
    # Comparar métricas principales
    metrics_to_compare = [
        'avg_relevancia',
        'avg_fidelidad', 
        'avg_precision',
        'avg_overall'
    ]
    
    print("\n📊 CAMBIOS EN MÉTRICAS:")
    print("-" * 70)
    print(f"{'Métrica':<20} {'Iter1':<12} {'Iter2':<12} {'Cambio':<12} {'%':<8}")
    print("-" * 70)
    
    for metric in metrics_to_compare:
        val1 = metrics1.get(metric, 0)
        val2 = metrics2.get(metric, 0)
        change = val2 - val1
        pct = (change / val1 * 100) if val1 > 0 else 0
        
        symbol = "⬆️" if change > 0 else ("⬇️" if change < 0 else "→")
        
        print(f"{metric:<20} {val1:<12.3f} {val2:<12.3f} {change:+.3f}     {pct:+.1f}% {symbol}")
    
    # Por tipo de pregunta
    print("\n📈 CAMBIOS POR TIPO DE PREGUNTA:")
    print("-" * 70)
    
    for q_type in metrics1.get('metrics_by_question_type', {}).keys():
        m1 = metrics1['metrics_by_question_type'][q_type]['avg_overall']
        m2 = metrics2['metrics_by_question_type'][q_type]['avg_overall']
        change = m2 - m1
        
        symbol = "⬆️" if change > 0 else ("⬇️" if change < 0 else "→")
        print(f"{q_type:<20} {m1:.3f} → {m2:.3f}  ({change:+.3f}) {symbol}")
    
    # Resumen
    print("\n" + "="*70)
    overall1 = metrics1.get('avg_overall', 0)
    overall2 = metrics2.get('avg_overall', 0)
    improvement = overall2 - overall1
    
    if improvement > 0.05:
        print(f"✅ MEJORA SIGNIFICATIVA: +{improvement:.3f}")
    elif improvement > 0:
        print(f"⬆️ Mejora leve: +{improvement:.3f}")
    elif improvement < -0.05:
        print(f"❌ EMPEORAMIENTO: {improvement:.3f}")
    else:
        print(f"→ Sin cambios: {improvement:.3f}")
    
    print("="*70 + "\n")

if __name__ == "__main__":
    # Ejemplo: comparar últimas dos iteraciones
    import os
    
    history_dir = Path("data/evaluation/history")
    if history_dir.exists():
        folders = sorted([f.name for f in history_dir.iterdir() if f.is_dir()])
        if len(folders) >= 2:
            print(f"Comparando: {folders[-2]} vs {folders[-1]}")
            compare_iterations(folders[-2], folders[-1])
        else:
            print("❌ Se necesitan al menos 2 iteraciones")
    else:
        print("❌ No existe el directorio de historia")
