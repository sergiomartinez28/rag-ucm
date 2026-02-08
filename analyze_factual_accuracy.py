"""Análisis de cambios en factual accuracy"""
import json
import re
from pathlib import Path

def has_numbers(text):
    return bool(re.search(r'\d+', text))

def extract_numbers(text):
    return set(re.findall(r'\d+', text))

def check_factual_match(answer, reference):
    """Verifica si los números de la referencia están en la respuesta"""
    answer_nums = extract_numbers(answer)
    ref_nums = extract_numbers(reference)
    
    if not ref_nums:
        return None  # No aplica
    
    return ref_nums.issubset(answer_nums)

# Cargar resultados
old_path = Path('data/evaluation/history/2026-02-07_1426/results.json')
new_path = Path('data/evaluation/history/2026-02-07_1650/results.json')

with open(old_path, 'r', encoding='utf-8') as f:
    old_results = json.load(f)

with open(new_path, 'r', encoding='utf-8') as f:
    new_results = json.load(f)

print('=' * 80)
print('ANÁLISIS DE FACTUAL ACCURACY')
print('=' * 80)

# Estadísticas generales
old_fa_count = sum(1 for r in old_results if check_factual_match(r['rag_answer'], r['reference_answer']) == True)
new_fa_count = sum(1 for r in new_results if check_factual_match(r['rag_answer'], r['reference_answer']) == True)

total_factual = sum(1 for r in old_results if has_numbers(r['reference_answer']))

print(f'\nTotal preguntas con números en referencia: {total_factual}')
print(f'Factual accuracy ANTES: {old_fa_count}/{total_factual} = {old_fa_count/total_factual*100:.1f}%')
print(f'Factual accuracy AHORA: {new_fa_count}/{total_factual} = {new_fa_count/total_factual*100:.1f}%')
print(f'Cambio: {new_fa_count - old_fa_count} preguntas')

# Encontrar regresiones
print('\n' + '=' * 80)
print('CASOS DE REGRESIÓN (Antes correcto → Ahora incorrecto)')
print('=' * 80)

regressions = []
for old_r, new_r in zip(old_results, new_results):
    if old_r['id'] != new_r['id']:
        continue
    
    old_fa = check_factual_match(old_r['rag_answer'], old_r['reference_answer'])
    new_fa = check_factual_match(new_r['rag_answer'], new_r['reference_answer'])
    
    if old_fa == True and new_fa == False:
        regressions.append((old_r, new_r))

print(f'\nEncontradas {len(regressions)} regresiones\n')

for i, (old_r, new_r) in enumerate(regressions[:5], 1):
    print(f'\n--- CASO {i}: ID={old_r["id"]} ---')
    print(f'Pregunta: {old_r["question"]}')
    print(f'\nReferencia: {old_r["reference_answer"]}')
    print(f'Números en ref: {extract_numbers(old_r["reference_answer"])}')
    print(f'\nANTES: {old_r["rag_answer"]}')
    print(f'Números: {extract_numbers(old_r["rag_answer"])}')
    print(f'\nAHORA: {new_r["rag_answer"]}')
    print(f'Números: {extract_numbers(new_r["rag_answer"])}')
    print('-' * 80)

# Encontrar mejoras
improvements = []
for old_r, new_r in zip(old_results, new_results):
    if old_r['id'] != new_r['id']:
        continue
    
    old_fa = check_factual_match(old_r['rag_answer'], old_r['reference_answer'])
    new_fa = check_factual_match(new_r['rag_answer'], new_r['reference_answer'])
    
    if old_fa == False and new_fa == True:
        improvements.append((old_r, new_r))

print('\n' + '=' * 80)
print(f'MEJORAS: {len(improvements)} casos (Antes incorrecto → Ahora correcto)')
print('=' * 80)

print(f'\nBalance neto: {len(improvements)} mejoras - {len(regressions)} regresiones = {len(improvements) - len(regressions)}')
