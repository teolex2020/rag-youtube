import gradio as gr
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import LlamaCpp
from langchain_core.prompts import PromptTemplate
from langchain.chains import RetrievalQA
import os
import tempfile

# --- ЕТАП 1: ЗАВАНТАЖЕННЯ (Loading) ---
def load_document(file_path):
    print(f"Завантаження документу: {file_path}")
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    return documents

# --- ЕТАП 2: РОЗБИВКА (Splitting) ---
def split_document(documents):
    print("Розбивка документу на чанки...")
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    print(f"Створено {len(chunks)} чанків.")
    return chunks

# --- ЕТАП 3: ВЕКТОРИЗАЦІЯ І ЗБЕРЕЖЕННЯ (Embedding & Storing) ---
def get_vector_store(chunks):
    print("Створення векторного сховища...")
    model_name = "sentence-transformers/all-MiniLM-L6-v2"
    model_kwargs = {'device': 'cpu'}
    embeddings = HuggingFaceEmbeddings(model_name=model_name, model_kwargs=model_kwargs)
    
    # Створюємо векторне сховище у тимчасовій директорії
    persist_directory = tempfile.mkdtemp()
    vector_store = Chroma.from_documents(
        documents=chunks, 
        embedding=embeddings,
        persist_directory=persist_directory
    )
    print("Векторне сховище створено.")
    return vector_store

# --- ЕТАП 4: ЗАПУСК ЛОКАЛЬНОЇ LLM ---
def get_local_llm():
    print("Завантаження локальної LLM...")
    # Шукаємо модель у поточній директорії
    possible_models = [
      "D:/drone/gemma-3n-E4B-it-Q6_K.gguf",
        "./mistral-7b-instruct-v0.2.Q4_K_M.gguf",
        "./llama-2-7b-chat.Q4_K_M.gguf",
        "./vicuna-7b-v1.5.Q4_K_M.gguf"
    ]
    
    model_path = None
    for path in possible_models:
        if os.path.exists(path):
            model_path = path
            break
    
    if not model_path:
        raise FileNotFoundError(
            "Модель не знайдено! Завантажте GGUF модель та покладіть її в папку зі скриптом.\n"
            "Рекомендовані моделі:\n"
            "- mistral-7b-instruct-v0.2.Q4_K_M.gguf\n"
            "- llama-2-7b-chat.Q4_K_M.gguf\n"
            "Завантажити можна з: https://huggingface.co/TheBloke"
        )
        
    llm = LlamaCpp(
        model_path=model_path,
        n_gpu_layers=0, 
        n_batch=512,
        n_ctx=4096,
        f16_kv=True,
        verbose=True,
        temperature=0.3,  
        max_tokens=1024,  
        repeat_penalty=1.1,  
        stop=["<end_of_turn>", "</s>", "<|endoftext|>"]  
    )
    print(f"Локальну LLM завантажено: {model_path}")
    return llm

# --- ЕТАП 5: ПОШУК ТА ГЕНЕРАЦІЯ (Retrieval & Generation) ---
def create_rag_chain(vector_store, llm):
    retriever = vector_store.as_retriever(search_type="similarity", search_kwargs={"k": 3})
    
    template = """<start_of_turn>user
Ти - корисний асистент, який відповідає на питання на основі наданого контексту.

КОНТЕКСТ:
{context}

ПИТАННЯ:
{question}

ІНСТРУКЦІЇ:
- Надай детальну та повну відповідь українською мовою
- Використовуй тільки інформацію з наданого контексту
- Якщо інформації недостатньо, скажи це чесно
- Відповідай розгорнуто та зрозуміло

ВІДПОВІДЬ:<end_of_turn>
<start_of_turn>model
"""
    
    prompt = PromptTemplate(template=template, input_variables=["context", "question"])
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt}
    )
    return qa_chain

# --- ІНТЕРФЕЙС З GRADIO ---
def setup_rag_pipeline(file):
    if file is None:
        return "⚠️ Будь ласка, завантажте PDF файл."
    
    try:
        file_path = file.name
        
        # Повний цикл RAG
        yield "📖 Завантаження документу..."
        documents = load_document(file_path)
        
        yield "✂️ Розбивка на чанки..."
        chunks = split_document(documents)
        
        yield "🧠 Створення векторного сховища..."
        vector_store = get_vector_store(chunks)
        
        yield "🤖 Завантаження локальної LLM..."
        llm = get_local_llm()
        
        yield "🔗 Створення RAG ланцюжка..."
        qa_chain = create_rag_chain(vector_store, llm)
        
        # Зберігаємо готовий ланцюжок
        global RAG_CHAIN
        RAG_CHAIN = qa_chain
        
        yield "✅ Система готова до ваших запитань!"
        
    except Exception as e:
        yield f"❌ Помилка: {str(e)}"

def get_answer(question):
    if not question.strip():
        return "⚠️ Будь ласка, введіть питання."
    
    if 'RAG_CHAIN' not in globals() or RAG_CHAIN is None:
        return "❌ Помилка: Система не готова. Будь ласка, спочатку завантажте PDF."
    
    try:
        print(f"Обробка питання: {question}")
        result = RAG_CHAIN.invoke({"query": question})
        
        answer = result['result'].strip()
        source_docs = result['source_documents']
        
        print(f"Отримана відповідь: {answer}")
        
        # Перевіряємо якість відповіді
        if len(answer) < 10 or answer.lower() in ['бн.', 'н/а', 'немає', 'не знаю']:
            # Пробуємо переформулювати питання
            reformulated_question = f"На основі наданого документу детально поясни: {question}. Надай повну інформацію з контексту."
            result = RAG_CHAIN.invoke({"query": reformulated_question})
            answer = result['result'].strip()
        
        # Форматуємо джерела
        sources_text = "\n\n📚 **Джерела:**\n"
        for i, doc in enumerate(source_docs, 1):
            page_num = doc.metadata.get('page', 'невідома')
            content_preview = doc.page_content[:150].replace('\n', ' ')
            sources_text += f"\n**{i}.** Сторінка {page_num}:\n"
            sources_text += f'"{content_preview}..."\n'
            
        return answer + sources_text
        
    except Exception as e:
        print(f"Помилка: {str(e)}")
        return f"❌ Помилка при обробці запиту: {str(e)}"

# Глобальна змінна для RAG ланцюжка
RAG_CHAIN = None

# --- GRADIO ІНТЕРФЕЙС ---
def create_interface():
    with gr.Blocks(theme=gr.themes.Soft(), title="RAG PDF Chat") as demo:
        gr.Markdown("""
        # 🤖 Чат з вашим PDF документом
        
        **Інструкції:**
        1. Завантажте PDF файл
        2. Дочекайтесь завершення обробки
        3. Задавайте питання про вміст документу
        
        *Використовує локальну LLM модель - дані не передаються в інтернет!*
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                pdf_upload = gr.File(
                    label="📄 Завантажте ваш PDF", 
                    file_types=[".pdf"],
                    height=100
                )
                
                status_text = gr.Textbox(
                    label="📊 Статус системи", 
                    interactive=False,
                    lines=2
                )
            
            with gr.Column(scale=2):
                question_input = gr.Textbox(
                    label="❓ Ваше питання",
                    placeholder="Наприклад: Про що цей документ?",
                    lines=2
                )
                
                submit_btn = gr.Button("🚀 Отримати відповідь", variant="primary")
        
        answer_output = gr.Textbox(
            label="💬 Відповідь", 
            lines=15, 
            interactive=False,
            show_copy_button=True
        )
        
        # Прив'язуємо події
        pdf_upload.upload(
            fn=setup_rag_pipeline, 
            inputs=pdf_upload, 
            outputs=status_text
        )
        
        submit_btn.click(
            fn=get_answer, 
            inputs=question_input, 
            outputs=answer_output
        )
        
        question_input.submit(
            fn=get_answer, 
            inputs=question_input, 
            outputs=answer_output
        )
        
        # Приклади питань
        gr.Examples(
            examples=[
                "Про що цей документ?",
                "Які основні висновки?",
                "Хто є автором?",
                "Коли був написаний цей документ?",
                "Які ключові термини згадуються?"
            ],
            inputs=question_input
        )
    
    return demo

# --- ЗАПУСК ПРОГРАМИ ---
if __name__ == "__main__":
    print("🚀 Запуск RAG PDF Chat...")
    print("📋 Перевірте, що у вас є:")
    print("   1. Встановлені всі залежності")
    print("   2. GGUF модель у папці зі скриптом")
    print("   3. Достатньо оперативної пам'яті (мін. 8GB)")
    
    demo = create_interface()
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )