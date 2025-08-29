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
import gc
import hashlib
import html

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
from functools import lru_cache

import gradio as gr

# ============================= Директрорії =============================
UPLOAD_DIR = "uploads"
CHROMA_DIR = "chroma_data"
MANIFEST_DIR = "manifests"
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

# ============================= Утиліти очищення/керування =============================
def safe_delete_directory(path: str, max_attempts: int = 3) -> bool:
    """Безпечне видалення директорії з кількома спробами"""
    for attempt in range(max_attempts):
        try:
            if os.path.exists(path):
                gc.collect()
                time.sleep(0.2)

                # Рекурсивно видаляємо
                for root, dirs, files in os.walk(path, topdown=False):
                    for file in files:
                        file_path = os.path.join(root, file)
                        try:
                            os.chmod(file_path, 0o777)
                            os.remove(file_path)
                        except (PermissionError, FileNotFoundError, OSError):
                            pass
                    for d in dirs:
                        dir_path = os.path.join(root, d)
                        try:
                            os.rmdir(dir_path)
                        except (OSError, PermissionError):
                            pass

                try:
                    os.rmdir(path)
                    return True
                except OSError:
                    pass
        except Exception:
            time.sleep(0.3)
    return False

def clear_chroma_cache() -> None:
    """Повне очищення кешу ChromaDB"""
    safe_delete_directory(CHROMA_DIR)
    os.makedirs(CHROMA_DIR, exist_ok=True)

def rotate_persist_dirs(base_dir: str, keep: int = 8) -> None:
    """Залишаємо лише останні N індексів (за часом модифікації)"""
    if not os.path.exists(base_dir):
        return
    dirs = [
        (d, os.path.getmtime(os.path.join(base_dir, d)))
        for d in os.listdir(base_dir)
        if os.path.isdir(os.path.join(base_dir, d))
    ]
    dirs.sort(key=lambda x: x[1], reverse=True)
    for d, _ in dirs[keep:]:
        safe_delete_directory(os.path.join(base_dir, d))

def cleanup_uploads(keep: int = 10) -> None:
    """Легка ротація завантажених копій PDF"""
    if not os.path.exists(UPLOAD_DIR):
        return
    files = [
        (f, os.path.getmtime(os.path.join(UPLOAD_DIR, f)))
        for f in os.listdir(UPLOAD_DIR)
        if os.path.isfile(os.path.join(UPLOAD_DIR, f))
    ]
    files.sort(key=lambda x: x[1], reverse=True)
    for f, _ in files[keep:]:
        try:
            os.remove(os.path.join(UPLOAD_DIR, f))
        except Exception:
            pass

# ============================= Хеші та шляхи =============================
def make_persist_dir(stable_path: str, strategy: str, retriever_mode: str, params_hash: str) -> str:
    """Детермінований шлях для індексу (у межах CHROMA_DIR)"""
    key = f"{stable_path}|{strategy}|{retriever_mode}|{params_hash}"
    h = hashlib.sha1(key.encode()).hexdigest()[:16]
    return os.path.join(CHROMA_DIR, f"idx_{h}")

def get_config_hash(strategy: str, retriever_mode: str, chunk_size: int,
                    chunk_overlap: int, window_size: int, window_step: int,
                    enrich_meta: bool, stable_path: str,
                    mmr_fetch_k: int, mmr_lambda: float) -> str:
    """Хеш конфігурації для розуміння, чи потрібна переіндексація"""
    try:
        st = os.stat(stable_path)
        file_part = f"{stable_path}_{st.st_size}_{st.st_mtime}"
    except Exception:
        file_part = stable_path
    cfg = f"{strategy}_{retriever_mode}_{chunk_size}_{chunk_overlap}_{window_size}_{window_step}_{enrich_meta}_{mmr_fetch_k}_{mmr_lambda}"
    return hashlib.md5(f"{cfg}_{file_part}".encode()).hexdigest()

# ============================= LLM / Embeddings =============================
@lru_cache(maxsize=1)
def get_llm():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не знайдено")
    from langchain_openai import ChatOpenAI
    # Більший ліміт токенів для довших відповідей
    return ChatOpenAI(model="gpt-4o-mini", temperature=0.2, max_tokens=1024)

@lru_cache(maxsize=1)
def get_embeddings():
    if not OPENAI_API_KEY:
        raise ValueError("OPENAI_API_KEY не знайдено")
    from langchain_openai import OpenAIEmbeddings
    return OpenAIEmbeddings(model="text-embedding-3-small")

# ============================= Завантаження та нормалізація =============================
def normalize_text(t: str) -> str:
    if not t:
        return t
    # склеїти перенесені по дефісу слова
    t = re.sub(r"-\s*\n\s*", "", t)
    # прибрати зайві пробіли перед переносами
    t = re.sub(r"[ \t]+\n", "\n", t)
    return t

def document_loader(file_path: str) -> List[Document]:
    try:
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Файл не знайдено: {file_path}")
        loader = PyPDFLoader(file_path)
        docs = loader.load()
        if not docs:
            raise ValueError("PDF файл порожній або не містить текстового контенту")
        for d in docs:
            d.page_content = normalize_text(d.page_content)
        return docs
    except Exception as e:
        raise ValueError(f"Помилка завантаження PDF: {str(e)}")

# ============================= Візуалізація чанків =============================
def get_color(seed: int, opacity: float = 1.0) -> str:
    rng = random.Random(seed)
    hue = rng.randint(0, 360)
    return f"hsla({hue}, 90%, 60%, {opacity})"

def visualize_chunks_on_page_v2(page_document: Document, page_chunks: List[Document]) -> str:
    """
    Повертає HTML, де чанки на сторінці позначені кольоровими підкресленнями.
    Використовує start_index, якщо доступний; інакше робить fallback через .find().
    """
    if not page_document or not page_document.page_content:
        return "<div>Немає даних для візуалізації</div>"

    page_text = page_document.page_content
    char_styles = [[] for _ in range(len(page_text))]

    for i, chunk in enumerate(page_chunks):
        if not chunk or not chunk.page_content:
            continue
        start_idx = chunk.metadata.get("start_index")
        if start_idx is None:
            # fallback — намагаємося знайти
            start_idx = page_text.find(chunk.page_content)
            if start_idx == -1:
                continue
        end_idx = min(start_idx + len(chunk.page_content), len(page_text))
        color = get_color(i)
        for j in range(start_idx, end_idx):
            char_styles[j].append(color)

    html_output = ""
    last_style: List[str] = []
    current_span = ""

    for i, ch in enumerate(page_text):
        current_style = sorted(char_styles[i])
        if current_style != last_style:
            if current_span:
                style = f"text-decoration: underline; text-decoration-color: {last_style[-1]}; text-decoration-style: wavy;" if last_style else ""
                safe_text = html.escape(current_span).replace("\n", "<br>")
                html_output += f'<span style="{style}">{safe_text}</span>'
            current_span = ch
            last_style = current_style
        else:
            current_span += ch

    if current_span:
        style = f"text-decoration: underline; text-decoration-color: {last_style[-1]}; text-decoration-style: wavy;" if last_style else ""
        safe_text = html.escape(current_span).replace("\n", "<br>")
        html_output += f'<span style="{style}">{safe_text}</span>'

    legend_html = "<div><b>Легенда чанків:</b></div>"
    for i in range(len(page_chunks)):
        color = get_color(i)
        legend_html += f'<div><span style="display:inline-block; width: 20px; height: 10px; background:{color}; margin-right: 5px;"></span> - Чанк {i}</div>'

    return f'<div style="font-family: monospace; line-height: 1.8;">{legend_html}<hr>{html_output}</div>'

# ============================= Persist helpers =============================
def ensure_persisted_copy(tmp_path: str) -> str:
    if not tmp_path or not os.path.exists(tmp_path):
        raise FileNotFoundError(f"Файл не знайдено: {tmp_path}")
    _, ext = os.path.splitext(tmp_path)
    stable_name = f"{uuid.uuid4().hex}{ext or '.pdf'}"
    stable_path = os.path.join(UPLOAD_DIR, stable_name)
    shutil.copy2(tmp_path, stable_path)
    return stable_path

def file_info(path: str) -> Dict[str, Any]:
    try:
        if not path or not os.path.exists(path):
            return {"error": "Файл не існує"}
        return {
            "path": os.path.abspath(path),
            "basename": os.path.basename(path),
            "size_bytes": os.path.getsize(path),
            "mtime": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(os.path.getmtime(path))),
        }
    except Exception as e:
        return {"error": str(e)}

# ============================= Metadata helpers =============================
SECTION_HINT_RE = re.compile(r"^\s*(?:[A-ZА-ЯІЇЄ0-9][^\n]{0,80})$")

def enrich_metadata(docs: List[Document], source_name: Optional[str] = None) -> List[Document]:
    enriched = []
    for d in docs:
        if not d or not d.page_content:
            continue
        meta = dict(d.metadata or {})
        page_text = d.page_content[:2000]
        candidates = [ln.strip() for ln in page_text.splitlines()[:10] if ln.strip()]
        header = next((c for c in candidates if SECTION_HINT_RE.match(c) and len(c) >= 6), None)

        meta["source"] = source_name or meta.get("source", "uploaded.pdf")
        if header:
            meta["section_title"] = header

        enriched.append(Document(page_content=d.page_content, metadata=meta))
    return enriched

# ============================= Split strategies =============================
def split_baseline(docs: List[Document], chunk_size: int, chunk_overlap: int) -> List[Document]:
    if not docs:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=max(chunk_size, 100),
        chunk_overlap=min(chunk_overlap, max(chunk_size - 1, 0)),
        add_start_index=True,   # ключ до стабільної візуалізації
    )
    return splitter.split_documents(docs)

def split_sliding_window(docs: List[Document], window_size: int, step: int) -> List[Document]:
    if not docs:
        return []
    step = max(step, 1)
    overlap = max(0, window_size - step)
    text_splitter = TokenTextSplitter(chunk_size=window_size, chunk_overlap=overlap)
    out: List[Document] = []
    for d in docs:
        if not d or not d.page_content:
            continue
        start = 0
        for i, p in enumerate(text_splitter.split_text(d.page_content)):
            m = dict(d.metadata or {}, window_index=i)
            # приблизний start_index для візуалізації (не ідеально, але краще ніж нічого)
            m["start_index"] = start
            out.append(Document(page_content=p, metadata=m))
            start += max(len(p) - overlap, 1)
    return out

def split_semantic(docs: List[Document], embeddings) -> List[Document]:
    if not docs:
        return []
    try:
        # Важливо: split_documents зберігає метадані сторінки
        chunker = SemanticChunker(
            embeddings,
            breakpoint_threshold_type="percentile",
            breakpoint_threshold_amount=90,
        )
        out = chunker.split_documents(docs)
        for i, d in enumerate(out):
            # SemanticChunker не гарантує start_index — для візуалізації fallback буде через .find
            d.metadata["semantic_idx"] = i
        return [d for d in out if d.page_content and d.page_content.strip()]
    except Exception:
        return split_baseline(docs, 800, 150)

# ============================= Vector DB =============================
def build_vectordb(chunks: List[Document], persist_dir: str) -> Chroma:
    if not chunks:
        raise ValueError("Немає чанків для індексації")

    # чистимо саме директорію цього індексу (щоб не змішувалося)
    safe_delete_directory(persist_dir)
    os.makedirs(persist_dir, exist_ok=True)

    safe_chunks = filter_complex_metadata(chunks)
    safe_chunks = [c for c in safe_chunks if c.page_content and c.page_content.strip()]
    if not safe_chunks:
        raise ValueError("Після фільтрації не залишилося валідних чанків")

    vdb = Chroma.from_documents(
        documents=safe_chunks,
        embedding=get_embeddings(),
        persist_directory=persist_dir,
    )
    vdb.persist()
    return vdb

# ============================= RAG helpers =============================
def make_retriever(vdb: Chroma, mode: str, mmr_fetch_k: int, mmr_lambda: float):
    search_kwargs: Dict[str, Any] = {"k": 4}
    if mode == "mmr":
        search_kwargs.update({"fetch_k": int(mmr_fetch_k), "lambda_mult": float(mmr_lambda)})
    return vdb.as_retriever(search_type=mode, search_kwargs=search_kwargs)

def render_sources(sources: List[Document]) -> str:
    if not sources:
        return "Джерела не знайдено"
    lines = []
    for d in sources:
        if not d or not d.page_content:
            continue
        page = d.metadata.get("page", 'N/A')
        preview = (d.page_content or "").replace("\n", " ").strip()[:400]
        lines.append(f"• p.{page}: {preview}…")
    return "\n".join(lines) if lines else "Джерела не знайдено"

# ============================= Інспектор/маніфест =============================
def chunks_manifest(stable_pdf_path: str, strategy: str, retriever_mode: str, chunks: List[Document]) -> Dict[str, Any]:
    items = []
    for i, d in enumerate(chunks):
        if not d:
            continue
        items.append({
            "idx": i,
            "page": d.metadata.get("page", 'N/A'),
            "section_title": d.metadata.get("section_title", ""),
            "chars": len(d.page_content or ""),
            "preview": (d.page_content or "").replace("\n", " ")[:400],
        })
    return {
        "saved_pdf": file_info(stable_pdf_path),
        "chroma_dir": os.path.abspath(CHROMA_DIR),
        "strategy": strategy,
        "retriever_mode": retriever_mode,
        "chunk_count": len(items),
        "chunks": items,
    }

def get_chunk_statistics(chunks: List[Document]) -> dict:
    if not chunks:
        return {"Кількість чанків": 0}
    valid = [c for c in chunks if c and c.page_content]
    if not valid:
        return {"Кількість чанків": 0, "Помилка": "Немає валідних чанків"}
    sizes = [len(c.page_content) for c in valid]
    return {
        "Кількість чанків": len(valid),
        "Середній розмір (символи)": int(np.mean(sizes)),
        "Мін. / Макс. розмір": f"{np.min(sizes)} / {np.max(sizes)}",
        "Ст. відхилення розміру": round(np.std(sizes), 2),
    }

def manifest_to_dataframe(manifest: dict) -> list:
    rows = []
    for c in manifest.get("chunks", []):
        rows.append([
            c.get("idx", "N/A"),
            c.get("page", "N/A"),
            c.get("chars", 0),
            c.get("section_title", ""),
            c.get("preview", ""),
        ])
    return rows

def save_manifest_to_file(manifest: Dict[str, Any]) -> str:
    try:
        path = os.path.join(MANIFEST_DIR, f"manifest_{uuid.uuid4().hex}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=2)
        return path
    except Exception:
        return ""

# ============================= Головна логіка =============================
initial_state: Tuple[Any, Dict] = (None, {})
EMPTY_DF = [["ID", "Сторінка", "Символи", "Заголовок розділу", "Прев'ю"]]

def ask_with_retrieval(
    file, query, strategy, retriever_mode,
    chunk_size, chunk_overlap, window_size, window_step, enrich_meta,
    mmr_fetch_k, mmr_lambda,
    state
):
    retriever, cache = state or (None, {})

    default_returns = (
        "Завантажте PDF-файл.",
        (None, {}),
        {},
        EMPTY_DF,
        "Завантажте файл для візуалізації.",
        "",
        ""
    )

    if not file or not getattr(file, "name", None):
        return default_returns

    try:
        # 1) Копіюємо у стабільний шлях
        stable_path = cache.get("stable_path")
        if not stable_path or cache.get("last_tmp") != file.name:
            stable_path = ensure_persisted_copy(file.name)
            cache = {}  # скидаємо старий кеш, якщо інший файл
        cache["last_tmp"] = file.name
        cache["stable_path"] = stable_path

        # 2) Обчислюємо хеш конфігурації вже на основі stable_path
        current_config_hash = get_config_hash(
            strategy, retriever_mode, int(chunk_size), int(chunk_overlap),
            int(window_size), int(window_step), bool(enrich_meta), stable_path,
            int(mmr_fetch_k), float(mmr_lambda)
        )

        cached_hash = cache.get("config_hash")
        needs_reindex = (cached_hash != current_config_hash)

        if needs_reindex:
            # трохи гігієни
            cleanup_uploads(keep=10)
            rotate_persist_dirs(CHROMA_DIR, keep=8)

            # 3) Лоад + метадані
            docs = document_loader(stable_path)
            if not docs:
                return ("Помилка: PDF файл порожній", (None, {}), {}, EMPTY_DF, "Файл порожній", "", "")

            if enrich_meta:
                docs = enrich_metadata(docs, os.path.basename(stable_path))

            # 4) Розбиття
            if strategy == "baseline":
                chunks = split_baseline(docs, int(chunk_size), int(chunk_overlap))
            elif strategy == "sliding_window":
                chunks = split_sliding_window(docs, int(window_size), int(window_step))
            elif strategy == "semantic":
                chunks = split_semantic(docs, get_embeddings())
            else:
                chunks = split_baseline(docs, 800, 150)

            if not chunks:
                return ("Помилка: не вдалося створити чанки", (None, {}), {}, EMPTY_DF, "Помилка розбиття", "", "")

            # 5) Побудова індексу
            persist_dir = make_persist_dir(stable_path, strategy, retriever_mode, current_config_hash)
            vdb = build_vectordb(chunks, persist_dir)
            retriever = make_retriever(vdb, retriever_mode, int(mmr_fetch_k), float(mmr_lambda))

            # 6) Маніфест + візуалізація
            manifest = chunks_manifest(stable_path, strategy, retriever_mode, chunks)
            _ = save_manifest_to_file(manifest)
            first_page_chunks = [c for c in chunks if c and c.metadata and c.metadata.get("page") == 0]
            viz_html = "Немає даних для візуалізації."
            if docs and first_page_chunks:
                viz_html = visualize_chunks_on_page_v2(docs[0], first_page_chunks)

            # 7) Новий кеш
            cache = {
                "config_hash": current_config_hash,
                "stable_path": stable_path,
                "strategy": strategy,
                "retriever_mode": retriever_mode,
                "chunk_size": int(chunk_size),
                "chunk_overlap": int(chunk_overlap),
                "window_size": int(window_size),
                "window_step": int(window_step),
                "enrich": bool(enrich_meta),
                "mmr_fetch_k": int(mmr_fetch_k),
                "mmr_lambda": float(mmr_lambda),
                "persist_dir": persist_dir,
                "stats": get_chunk_statistics(chunks),
                "manifest_df": manifest_to_dataframe(manifest),
                "manifest_json_str": json.dumps(manifest, ensure_ascii=False, indent=2),
                "viz_html": viz_html,
            }

    except Exception as e:
        err = f"Помилка індексації: {e}"
        return (err, (None, {}), {}, EMPTY_DF, str(e), "", "")

    # Дані для UI
    stats = cache.get("stats", {})
    df_data = cache.get("manifest_df", EMPTY_DF)
    viz_html = cache.get("viz_html", "Немає даних для візуалізації.")
    stable_path_out = cache.get("stable_path", "")
    manifest_str = cache.get("manifest_json_str", "{}")

    if not retriever:
        return ("Система не готова. Завантажте файл.", (retriever, cache), stats, df_data, viz_html, stable_path_out, manifest_str)

    if not query or not query.strip():
        return ("Введіть запит.", (retriever, cache), stats, df_data, viz_html, stable_path_out, manifest_str)

    try:
        qa = RetrievalQA.from_chain_type(
            llm=get_llm(),
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True
        )
        result = qa.invoke({"query": query})
        answer = result.get("result", "Немає відповіді")
        sources = result.get("source_documents", [])
        src_text = render_sources(sources)
        full_response = f"{answer}\n\n**Джерела:**\n{src_text}"
        return (full_response, (retriever, cache), stats, df_data, viz_html, stable_path_out, manifest_str)
    except Exception as e:
        err = f"Помилка QA: {e}"
        return (err, (retriever, cache), stats, df_data, viz_html, stable_path_out, manifest_str)

def on_clear():
    try:
        clear_chroma_cache()
        return ("Кеш ChromaDB очищено. Завантажте файл заново.",
                (None, {}),
                {},
                EMPTY_DF,
                "—",
                "",
                "")
    except Exception as e:
        return (f"Помилка очищення: {e}",
                (None, {}),
                {},
                EMPTY_DF,
                "—",
                "",
                "")

# ============================= Gradio UI =============================
with gr.Blocks(theme=gr.themes.Soft(), title="RAG: Інспектор стратегій") as demo:
    gr.Markdown("### Mini RAG з різними підходами до підготовки тексту + інспектор індексу/файлу")

    state = gr.State(value=initial_state)

    with gr.Row():
        with gr.Column(scale=1):
            file_in = gr.File(label="PDF", file_count="single", file_types=[".pdf"])
            query_in = gr.Textbox(label="Запит", placeholder="Постав питання за змістом PDF…", lines=3)

            with gr.Accordion("Налаштування", open=True):
                strategy = gr.Dropdown(
                    ["baseline", "sliding_window", "semantic"],
                    value="baseline",
                    label="Стратегія розбиття"
                )
                retriever_mode = gr.Radio(
                    ["similarity", "mmr"],
                    value="mmr",
                    label="Режим ретрівера"
                )
                chunk_size = gr.Slider(200, 2000, 800, step=50, label="chunk_size (baseline)")
                chunk_overlap = gr.Slider(0, 400, 150, step=10, label="chunk_overlap (baseline)")
                window_size = gr.Slider(200, 2000, 800, step=50, label="window_size (sliding)")
                window_step = gr.Slider(50, 1000, 300, step=10, label="window_step (sliding)")
                enrich_meta_flag = gr.Checkbox(value=True, label="Збагачувати метадані")
                mmr_fetch_k = gr.Slider(4, 50, 15, step=1, label="MMR fetch_k")
                mmr_lambda = gr.Slider(0.0, 1.0, 0.5, step=0.05, label="MMR lambda")

            go_btn = gr.Button("Запит", variant="primary")
            clear_btn = gr.Button("Очистити кеш", variant="secondary")

        with gr.Column(scale=2):
            with gr.Row():
                out_answer = gr.Markdown(label="Відповідь та Джерела")
                stats_out = gr.JSON(label="Статистика по чанках")

            with gr.Tabs():
                with gr.Tab(label="Інспектор Чанків (Сторінка 0)"):
                    chunk_viz_out = gr.HTML(label="Візуалізація розбивки на першій сторінці")
                with gr.Tab(label="Маніфест (таблиця)"):
                    # У Gradio 4 часто використовується col_names
                    manifest_df_out = gr.DataFrame(headers=["ID", "Сторінка", "Символи", "Заголовок розділу", "Прев'ю"], wrap=True)
                with gr.Tab(label="Збережені файли/JSON"):
                    saved_pdf_path_box = gr.Textbox(label="Шлях збереженого PDF", interactive=False)
                    manifest_content_display = gr.Textbox(label="Повний маніфест у форматі JSON", lines=10, interactive=False)

    inputs = [
        file_in, query_in, strategy, retriever_mode,
        chunk_size, chunk_overlap, window_size, window_step, enrich_meta_flag,
        mmr_fetch_k, mmr_lambda,
        state
    ]
    outputs = [
        out_answer, state, stats_out, manifest_df_out, chunk_viz_out,
        saved_pdf_path_box, manifest_content_display
    ]

    go_btn.click(fn=ask_with_retrieval, inputs=inputs, outputs=outputs)
    file_in.upload(fn=ask_with_retrieval, inputs=inputs, outputs=outputs)
    clear_btn.click(fn=on_clear, inputs=None, outputs=outputs)

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
