# app_rag_strategies_fixed.py
from typing import List, Tuple, Any, Optional, Dict
import os
import re
import json
import uuid
import time
import shutil
import random
import warnings
import numpy as np

warnings.filterwarnings("ignore")

from config import OPENAI_API_KEY
from langchain_core.documents import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    TokenTextSplitter
)
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.vectorstores import Chroma
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.chains import RetrievalQA

import gradio as gr

# ============================= Директрорії =============================
UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_data"
MANIFEST_DIR = "manifests"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

# ============================= LLM / Embeddings =============================
def get_llm():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не знайдено")
    from langchain_openai import ChatOpenAI
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=512)

def get_embeddings():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не знайдено")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")

# ============================= Loading & Visualization =============================
def document_loader(file_path: str) -> List[Document]:
    try:
        loader = PyPDFLoader(file_path)
        return loader.load()
    except Exception as e:
        raise ValueError(f"Помилка завантаження PDF: {str(e)}")

def get_color(seed, opacity=1.0):
    random.seed(seed)
    hue = random.randint(0, 360)
    return f"hsla({hue}, 90%, 60%, {opacity})"

def visualize_chunks_on_page_v2(page_document: Document, page_chunks: List[Document]) -> str:
    """
    Повертає HTML, де чанки на сторінці позначені кольоровими підкресленнями,
    що дозволяє візуалізувати перекриття.
    """
    page_text = page_document.page_content
    
    # Створюємо список "змін кольору" для кожної позиції символу
    char_styles = [[] for _ in range(len(page_text))]

    # Для кожного чанка додаємо його колір до відповідних символів
    for i, chunk in enumerate(page_chunks):
        chunk_text = chunk.page_content
        start_idx = page_text.find(chunk_text)
        if start_idx == -1:
            continue
        
        end_idx = start_idx + len(chunk_text)
        color = get_color(i)
        
        for j in range(start_idx, end_idx):
            char_styles[j].append(color)

    # Будуємо HTML, об'єднуючи символи з однаковим стилем
    html_output = ""
    last_style = []
    current_span = ""

    for i, char in enumerate(page_text):
        current_style = sorted(char_styles[i])
        
        if current_style != last_style:
            # Закриваємо попередній span і відкриваємо новий
            if current_span:
                # Створюємо стиль з кількох підкреслень
                text_decoration_lines = " ".join(["underline"] * len(last_style))
                text_decoration_colors = " ".join(last_style)
                style = (f'text-decoration: {text_decoration_lines}; '
                         f'text-decoration-color: {text_decoration_colors}; '
                         f'text-decoration-style: wavy;') if last_style else ""
                
                # Замінюємо переноси рядків на <br> і екрануємо HTML
                safe_text = current_span.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('\n', '<br>')
                html_output += f'<span style="{style}">{safe_text}</span>'
            
            current_span = char
            last_style = current_style
        else:
            current_span += char
    
    # Додаємо останній span
    if current_span:
        text_decoration_lines = " ".join(["underline"]*len(last_style))
        text_decoration_colors = " ".join(last_style)
        style = (f'text-decoration: {text_decoration_lines}; '
                 f'text-decoration-color: {text_decoration_colors}; '
                 f'text-decoration-style: wavy;') if last_style else ""
        safe_text = current_span.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('\n', '<br>')
        html_output += f'<span style="{style}">{safe_text}</span>'

    # Додаємо легенду
    legend_html = "<div><b>Легенда чанків:</b></div>"
    for i in range(len(page_chunks)):
        color = get_color(i)
        legend_html += f'<div><span style="display:inline-block; width: 20px; height: 10px; background:{color}; margin-right: 5px;"></span> - Чанк {i}</div>'

    return f'<div style="font-family: monospace; line-height: 1.8;">{legend_html}<hr>{html_output}</div>'
# ============================= Persist helpers =============================
def ensure_persisted_copy(tmp_path: str) -> str:
    if not os.path.exists(tmp_path): raise FileNotFoundError(f"Файл не знайдено: {tmp_path}")
    _, ext = os.path.splitext(tmp_path)
    stable_name = f"{uuid.uuid4().hex}{ext or '.pdf'}"
    stable_path = os.path.join(UPLOAD_DIR, stable_name)
    try:
        shutil.copy2(tmp_path, stable_path)
        return stable_path
    except Exception as e:
        raise IOError(f"Помилка копіювання файлу: {str(e)}")

def file_info(path: str) -> Dict[str, Any]:
    try:
        return {
            "path": os.path.abspath(path),
            "basename": os.path.basename(path),
            "size_bytes": os.path.getsize(path),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path)))
        }
    except Exception:
        return {}

# ============================= Metadata helpers =============================
SECTION_HINT_RE = re.compile(r"^\s*(?:[A-ZА-ЯІЇЄ0-9][^\n]{0,80})$")

def enrich_metadata(docs: List[Document], source_name: Optional[str] = None) -> List[Document]:
    enriched = []
    for d in docs:
        meta = d.metadata or {}
        page_text = d.page_content[:2000]
        candidates = [ln.strip() for ln in page_text.splitlines()[:10] if ln.strip()]
        header = next((c for c in candidates if SECTION_HINT_RE.match(c) and len(c) >= 6), None)
        
        meta["source"] = source_name or meta.get("source", "uploaded.pdf")
        if header: meta["section_title"] = header
        
        enriched.append(Document(page_content=d.page_content, metadata=meta))
    return enriched

# ============================= Split strategies =============================
def split_baseline(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return splitter.split_documents(docs)

def split_sliding_window(docs: List[Document], window_size: int, step: int) -> List[Document]:
    text_splitter = TokenTextSplitter(chunk_size=window_size, chunk_overlap=window_size - step)
    out = []
    for d in docs:
        for i, p in enumerate(text_splitter.split_text(d.page_content)):
            m = dict(d.metadata, window_index=i)
            out.append(Document(page_content=p, metadata=m))
    return out

def split_semantic(docs: List[Document], embeddings) -> List[Document]:
    try:
        chunker = SemanticChunker(embeddings, breakpoint_threshold_type="percentile", breakpoint_threshold_amount=90)
        out = []
        for d in docs:
            for i, p in enumerate(chunker.split_text(d.page_content)):
                m = dict(d.metadata, semantic_idx=i)
                out.append(Document(page_content=p, metadata=m))
        return out
    except Exception:
        return split_baseline(docs, 800, 150)

# ============================= Vector DB =============================
def build_vectordb(chunks: List[Document]) -> Chroma:
    safe_chunks = filter_complex_metadata(chunks)
    vdb = Chroma.from_documents(documents=safe_chunks, embedding=get_embeddings(), persist_directory=CHROMA_DIR)
    vdb.persist()
    return vdb

# ============================= RAG step =============================
def make_retriever(vdb: Chroma, mode: str):
    search_kwargs = {"k": 4}
    if mode == "mmr":
        search_kwargs.update({"fetch_k": 20, "lambda_mult": 0.5})
    return vdb.as_retriever(search_type=mode, search_kwargs=search_kwargs)

def render_sources(sources: List[Document]) -> str:
    lines = []
    for d in sources:
        page = d.metadata.get("page", 'N/A')
        preview = (d.page_content or "").replace("\n", " ").strip()[:400]
        lines.append(f"• p.{page}: {preview}…")
    return "\n".join(lines)

# ============================= Inspection helpers =============================
def chunks_manifest(stable_pdf_path: str, strategy: str, retriever_mode: str, chunks: List[Document]) -> Dict[str, Any]:
    items = [{
        "idx": i,
        "page": d.metadata.get("page"),
        "section_title": d.metadata.get("section_title", ""),
        "chars": len(d.page_content),
        "preview": (d.page_content or "").replace("\n", " ")[:400]
    } for i, d in enumerate(chunks)]
    return {
        "saved_pdf": file_info(stable_pdf_path),
        "chroma_dir": os.path.abspath(CHROMA_DIR),
        "strategy": strategy, "retriever_mode": retriever_mode,
        "chunk_count": len(chunks), "chunks": items
    }

def get_chunk_statistics(chunks: List[Document]) -> dict:
    if not chunks: return {}
    sizes = [len(c.page_content) for c in chunks]
    return {
        "Кількість чанків": len(chunks),
        "Середній розмір (символи)": int(np.mean(sizes)),
        "Мін. / Макс. розмір": f"{np.min(sizes)} / {np.max(sizes)}",
        "Ст. відхилення розміру": round(np.std(sizes), 2),
    }

def manifest_to_dataframe(manifest: dict) -> list:
    rows = [[
        c.get("idx"), c.get("page"), c.get("chars"),
        c.get("section_title", ""), c.get("preview", "")
    ] for c in manifest.get("chunks", [])]
    return rows

def save_manifest_to_file(manifest: Dict[str, Any]) -> str:
    path = os.path.join(MANIFEST_DIR, f"manifest_{uuid.uuid4().hex}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    return path

# ============================= Gradio glue =============================
initial_state: Tuple[Any, Dict] = (None, {})
EMPTY_DF = [["ID", "Сторінка", "Символи", "Заголовок розділу", "Прев'ю"]]

def ask_with_retrieval(
    file, query, strategy, retriever_mode,
    chunk_size, chunk_overlap, window_size, window_step, enrich_meta, state
):
    retriever, cache = state or (None, {})
    cache = dict(cache or {})
    
    # Визначаємо, чи потрібна переіндексація
    params_changed = (
        cache.get("file") != (file.name if file else None) or
        cache.get("strategy") != strategy or cache.get("retriever_mode") != retriever_mode or
        cache.get("chunk_size") != chunk_size or cache.get("chunk_overlap") != chunk_overlap or
        cache.get("window_size") != window_size or cache.get("window_step") != window_step or
        cache.get("enrich") != enrich_meta
    )

    try:
        if params_changed:
            if not file:
                return "Завантажте PDF-файл.", (None, {}), {}, [], "Завантажте файл для візуалізації.", "", ""
            
            # 1. Індексація
            stable_path = ensure_persisted_copy(file.name)
            docs = document_loader(stable_path)
            
            
            
            if enrich_meta: docs = enrich_metadata(docs, os.path.basename(stable_path))

            if strategy == "baseline": chunks = split_baseline(docs, chunk_size, chunk_overlap)
            elif strategy == "sliding_window": chunks = split_sliding_window(docs, window_size, window_step)
            elif strategy == "semantic": chunks = split_semantic(docs, get_embeddings())
            else: chunks = split_baseline(docs, chunk_size, chunk_overlap)

            vdb = build_vectordb(chunks)
            retriever = make_retriever(vdb, retriever_mode)
            
            # 2. Підготовка даних для інспекції
            manifest = chunks_manifest(stable_path, strategy, retriever_mode, chunks)
            manifest_path = save_manifest_to_file(manifest)
            
            # 3. Кешування результатів
            cache = {
                "file": file.name, "stable_path": stable_path, "strategy": strategy,
                "retriever_mode": retriever_mode, "chunk_size": chunk_size, "chunk_overlap": chunk_overlap,
                "window_size": window_size, "window_step": window_step, "enrich": enrich_meta,
                "stats": get_chunk_statistics(chunks),
                "manifest_df": manifest_to_dataframe(manifest),
                "manifest_json_str": json.dumps(manifest, ensure_ascii=False, indent=2),
                "viz_html": visualize_chunks_on_page_v2(docs[0], [c for c in chunks if c.metadata.get("page")==0]) if docs else "",
            }
    except Exception as e:
        return f"Помилка індексації: {e}", (None, {}), {}, [], str(e), "", ""

    # Витягуємо дані для UI з кешу
    stats = cache.get("stats", {})
    df_data = cache.get("manifest_df", [])
    viz_html = cache.get("viz_html", "Немає даних для візуалізації.")
    stable_path_out = cache.get("stable_path", "")
    manifest_str = cache.get("manifest_json_str", "{}")
    
    if not retriever:
        return "Система не готова. Завантажте файл.", (retriever, cache), stats, df_data, viz_html, stable_path_out, manifest_str

    if not query.strip():
        return "Введіть запит.", (retriever, cache), stats, df_data, viz_html, stable_path_out, manifest_str

    try:
        qa = RetrievalQA.from_chain_type(llm=get_llm(), chain_type="stuff", retriever=retriever, return_source_documents=True)
        result = qa.invoke({"query": query})
        answer = result.get("result", "")
        sources = result.get("source_documents", [])
        src_text = render_sources(sources)
        full_response = f"{answer}\n\n**Джерела:**\n{src_text}"
        return full_response, (retriever, cache), stats, df_data, viz_html, stable_path_out, manifest_str
    except Exception as e:
        return f"Помилка QA: {e}", (retriever, cache), stats, df_data, viz_html, stable_path_out, manifest_str

# ============================= Gradio UI (ВИПРАВЛЕНА ВЕРСІЯ) =============================
with gr.Blocks(theme=gr.themes.Soft(), title="RAG: Інспектор стратегій") as demo:
    gr.Markdown("### Mini RAG з різними підходами до підготовки тексту + інспектор індексу/файлу")

    state = gr.State(value=initial_state)

    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="PDF", file_count="single", file_types=[".pdf"])
            query_in = gr.Textbox(label="Запит", placeholder="Постав питання за змістом PDF…", lines=3)
            with gr.Accordion("Налаштування", open=True):
                strategy = gr.Dropdown(["baseline", "sliding_window", "semantic"], value="baseline", label="Стратегія розбиття")
                retriever_mode = gr.Radio(["similarity", "mmr"], value="mmr", label="Режим ретрівера")
                chunk_size = gr.Slider(200, 2000, 800, step=50, label="chunk_size (baseline)")
                chunk_overlap = gr.Slider(0, 400, 150, step=10, label="chunk_overlap (baseline)")
                window_size = gr.Slider(200, 2000, 800, step=50, label="window_size (sliding)")
                window_step = gr.Slider(50, 1000, 300, step=10, label="window_step (sliding)")
                enrich_meta_flag = gr.Checkbox(value=True, label="Збагачувати метадані")
            go_btn = gr.Button("Запит", variant="primary")

        with gr.Column(scale=2):
            with gr.Row():
                out_answer = gr.Markdown(label="Відповідь та Джерела")
                stats_out = gr.JSON(label="Статистика по чанках")

            with gr.Tabs():
                with gr.Tab(label="Інспектор Чанків (Сторінка 0)"):
                    chunk_viz_out = gr.HTML(label="Візуалізація розбивки на першій сторінці")
                with gr.Tab(label="Маніфест (таблиця)"):
                    manifest_df_out = gr.DataFrame(headers=["ID", "Сторінка", "Символи", "Заголовок розділу", "Прев'ю"], wrap=True)
                with gr.Tab(label="Збережені файли/JSON"):
                    saved_pdf_path_box = gr.Textbox(label="Шлях збереженого PDF", interactive=False)
                    manifest_content_display = gr.Textbox(label="Повний маніфест у форматі JSON", lines=10, interactive=False)


    inputs = [file_in, query_in, strategy, retriever_mode, chunk_size, chunk_overlap, window_size, window_step, enrich_meta_flag, state]
    outputs = [out_answer, state, stats_out, manifest_df_out, chunk_viz_out, saved_pdf_path_box, manifest_content_display]

    # Ця логіка зв'язує кнопку з основною функцією
    go_btn.click(fn=ask_with_retrieval, inputs=inputs, outputs=outputs)
    # Цей рядок дозволяє автоматично запускати обробку при завантаженні файлу (без введення запиту)
    file_in.upload(fn=ask_with_retrieval, inputs=inputs, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)