#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Interfaz CLI para RAG-UCM
Uso: python cli.py "¿Cuál es el plazo para el TFM?"
"""

import io
import sys
from pathlib import Path
import typer

# Forzar UTF-8 en Windows
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from src.pipeline import RAGPipeline

app = typer.Typer(help="🎓 RAG-UCM - Asistente Académico UCM")
console = Console()


@app.command()
def ask(
    question: str = typer.Argument(..., help="Pregunta sobre normativa UCM"),
    top_k: int = typer.Option(5, "--top-k", "-k", help="Número de documentos a recuperar"),
    no_verification: bool = typer.Option(False, "--no-verification", help="Desactivar verificación de fidelidad"),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Mostrar información detallada")
):
    """
    Hacer una pregunta al asistente académico
    
    Ejemplo:
        python cli.py ask "¿Cuándo es el plazo para presentar el TFM?"
    """
    try:
        with console.status("[bold blue]Cargando sistema RAG-UCM...", spinner="dots"):
            rag = RAGPipeline()
        
        console.print("\n[bold green]✓[/bold green] Sistema cargado\n")
        
        # Hacer la pregunta
        with console.status("[bold blue]Buscando en la normativa...", spinner="dots"):
            result = rag.query(
                question=question,
                top_k=top_k,
                include_verification=not no_verification
            )
        
        # Mostrar respuesta
        console.print(Panel(
            result['answer'],
            title="📝 Respuesta",
            border_style="green",
            padding=(1, 2)
        ))
        
        # Mostrar advertencia si existe
        if 'warning' in result:
            console.print(Panel(
                result['warning'],
                title="⚠️  Advertencia",
                border_style="yellow",
                padding=(1, 2)
            ))
        
        # Mostrar fuentes
        if result['sources']:
            console.print("\n[bold]📚 Fuentes consultadas:[/bold]\n")
            
            table = Table(show_header=True, header_style="bold cyan")
            table.add_column("#", style="dim", width=3)
            table.add_column("Documento")
            table.add_column("Relevancia", justify="right", width=12)
            
            for source in result['sources']:
                table.add_row(
                    f"[{source['id']}]",
                    source['title'],
                    f"{source['score']:.3f}"
                )
            
            console.print(table)
        
        # Información verbose
        if verbose:
            console.print("\n[bold]🔍 Información detallada:[/bold]")
            
            # Verificación
            if 'verification' in result:
                ver = result['verification']
                console.print(f"  • Fidelidad: {ver['fidelity_score']:.2%}")
                console.print(f"  • Oraciones verificadas: {ver['num_sentences'] - ver['num_unsupported']}/{ver['num_sentences']}")
            
            # Metadata
            meta = result.get('metadata', {})
            console.print(f"  • Modelo: {meta.get('model', 'N/A')}")
            console.print(f"  • Temperatura: {meta.get('temperature', 'N/A')}")
            console.print(f"  • Contextos usados: {meta.get('num_contexts', 0)}")
    
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def build(
    documents_path: Path = typer.Option(
        "./data/raw",
        "--path",
        "-p",
        help="Ruta a la carpeta con documentos PDF/HTML"
    )
):
    """
    Construir índices desde documentos
    
    Ejemplo:
        python cli.py build --path ./data/raw
    """
    try:
        console.print("[bold blue]🔨 Construyendo índices...[/bold blue]\n")
        
        rag = RAGPipeline(load_existing=False)
        rag.build_index(documents_path)
        
        console.print("\n[bold green]✓ Índices construidos correctamente[/bold green]")
        
        # Mostrar estadísticas
        stats = rag.get_stats()
        if 'index' in stats and stats['index'].get('total_chunks'):
            console.print(f"\n📊 Estadísticas:")
            console.print(f"  • Total chunks: {stats['index']['total_chunks']}")
            console.print(f"  • Vectores FAISS: {stats['index']['faiss_vectors']}")
            console.print(f"  • Dimensión embeddings: {stats['index']['embedding_dim']}")
            console.print(f"  • Longitud promedio: {stats['index']['avg_chunk_length']:.0f} palabras")
    
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def stats():
    """
    Mostrar estadísticas del sistema
    """
    try:
        rag = RAGPipeline()
        stats = rag.get_stats()
        
        console.print("\n[bold]📊 Estadísticas RAG-UCM[/bold]\n")
        
        # Índices
        if 'index' in stats:
            idx = stats['index']
            if idx.get('total_chunks'):
                console.print("[bold cyan]Índices:[/bold cyan]")
                console.print(f"  • Total chunks: {idx['total_chunks']}")
                console.print(f"  • Vectores FAISS: {idx['faiss_vectors']}")
                console.print(f"  • Modelo embeddings: {idx['embedding_model']}")
                console.print(f"  • Dimensión: {idx['embedding_dim']}")
                console.print(f"  • Longitud promedio: {idx['avg_chunk_length']:.0f} palabras")
            else:
                console.print("[yellow]No hay índices construidos[/yellow]")
        
        # Configuración
        console.print("\n[bold cyan]Configuración:[/bold cyan]")
        config = stats.get('config', {})
        console.print(f"  • Chunk size: {config.get('chunk_size')}")
        console.print(f"  • Chunk overlap: {config.get('chunk_overlap')}")
        console.print(f"  • Top-k retrieval: {config.get('top_k_retrieval')}")
        console.print(f"  • Top-k rerank: {config.get('top_k_rerank')}")
        console.print(f"  • Hybrid alpha: {config.get('hybrid_alpha')}")
        console.print(f"  • Verificación: {'✓' if config.get('enable_verification') else '✗'}")
    
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


@app.command()
def interactive():
    """
    Modo interactivo (chat continuo)
    """
    console.print("\n[bold blue]🎓 RAG-UCM - Modo Interactivo[/bold blue]")
    console.print("Escribe 'salir' o 'exit' para terminar\n")
    
    try:
        with console.status("[bold blue]Cargando...", spinner="dots"):
            rag = RAGPipeline()
        
        console.print("[bold green]✓ Sistema listo[/bold green]\n")
        
        while True:
            question = console.input("[bold cyan]Pregunta:[/bold cyan] ")
            
            if question.lower() in ['salir', 'exit', 'quit']:
                console.print("\n[bold]👋 ¡Hasta pronto![/bold]")
                break
            
            if not question.strip():
                continue
            
            try:
                with console.status("[bold blue]Buscando...", spinner="dots"):
                    result = rag.query(question)
                
                console.print(f"\n[bold green]Respuesta:[/bold green] {result['answer']}\n")
                
                if 'warning' in result:
                    console.print(f"[yellow]{result['warning']}[/yellow]\n")
            
            except Exception as e:
                console.print(f"[red]Error: {str(e)}[/red]\n")
    
    except Exception as e:
        console.print(f"[bold red]❌ Error:[/bold red] {str(e)}")
        raise typer.Exit(code=1)


if __name__ == "__main__":
    app()
