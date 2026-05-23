from typing import Union
import os
import glob
import json

from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter
# Import OpenAI Callback to track tokens and cost
from langchain_community.callbacks import get_openai_callback

# Load background files (Removed 'sangeetha_resume.txt' from sources as requested)
with open('website/about_this_app.txt') as f: file_text_about = f.read()
with open('website/biographical.txt') as f: file_text_bio = f.read()
with open('website/interview_faq.txt') as f: file_text_faq = f.read()
with open('website/sangeetha_resume.md') as f: file_text_resume_md = f.read()

from langchain_community.document_loaders import PyPDFLoader

sources = [
    Document(page_content=file_text_about, metadata={"source": "About this app"}),
    Document(page_content=file_text_faq, metadata={"source": "Interview FAQ"}),
    Document(page_content=file_text_bio, metadata={"source": "Biographical info"}),
    Document(page_content=file_text_resume_md, metadata={"source": "Sangeetha's resume markdown"})
]

# Load PDFs from static folder
static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')
pdf_files = glob.glob(os.path.join(static_folder, '*.pdf'))
for pdf_file in pdf_files:
    try:
        loader = PyPDFLoader(pdf_file)
        pdf_docs = loader.load()
        sources.extend(pdf_docs)
    except Exception as e:
        print(f"Failed to load PDF {pdf_file}: {e}")

source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

# Defaulting to gpt-4o-mini
chain = load_qa_with_sources_chain(ChatOpenAI(model="gpt-4o-mini", temperature=0))


def string_answer(question: str, test: Union[str, None] = None) -> str:
    if test is not None:
        return test

    # Capture tokens spent during normal QA
    with get_openai_callback() as cb:
        output = chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )
        print(f"\n--- [Token Usage: QA Chain] ---")
        print(f"Tokens Spent: {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")
        print(f"Approximate Cost: ${cb.total_cost:.6f}")
        print(f"---------------------------------\n")
        
        return str(output["output_text"])


from .scraper import extract_job_description

def generate_pitch_for_job(url: str, require_h1b: bool = False) -> dict:
    job_description = extract_job_description(url)
    if not job_description:
        return {"error": "Could not extract job description from the provided URL."}
        
    # Preserved original behavior: no token reduction / dumps all files into the context
    background_text = "\n\n".join([doc.page_content for doc in sources])
    
    h1b_instructions = ""
    if require_h1b:
        h1b_instructions = "IMPORTANT: You MUST ONLY suggest companies that have a 'Known' or 'Likely' history of sponsoring H1B visas. Exclude companies that are 'Unlikely' or 'Possible'."
    
    prompt = f"""
You are Sangeetha. Use the following background information about yourself:
{background_text}

Read the following job description:
{job_description[:4000]}

First, answer these three questions convincingly based on your background and the job description:
1. Why should we hire you?
2. Why do you want to work here?
3. What can you bring to the table?

Second, act as a career advisor and suggest 3 to 5 other target companies that hire for similar roles and would be a good fit for Sangeetha's resume.
For each company, provide:
- "company_name": Name of the company.
- "role": A similar role title they typically hire for.
- "reason": Why they are a good fit based on the resume.
- "h1b_score": Score their likelihood of H1B sponsorship as one of [Known, Likely, Possible, Unlikely].

{h1b_instructions}

Respond strictly with a valid JSON object containing exactly these four keys: 
"hire_you", "want_to_work", "bring_to_table", "target_companies".

"target_companies" must be an array of objects with exactly these keys: "company_name", "role", "reason", "h1b_score".
"""
    # Enforce standard JSON formatting via LangChain model arguments
    llm = ChatOpenAI(
        model="gpt-4o-mini", 
        temperature=0.2,
        model_kwargs={"response_format": {"type": "json_object"}}
    )
    
    # Track tokens and cost of this generation
    with get_openai_callback() as cb:
        response = llm.invoke(prompt)
        print(f"\n--- [Token Usage: Job Pitch Generation] ---")
        print(f"Tokens Spent: {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")
        print(f"Approximate Cost: ${cb.total_cost:.6f}")
        print(f"---------------------------------------------\n")
    
    try:
        # Since JSON mode is strictly enforced, we directly parse the text safely
        clean_text = response.content.strip()
        return json.loads(clean_text)
    except Exception as e:
        return {
            "error": "Failed to parse response.", 
            "details": str(e), 
            "raw": response.content
        }

if __name__ == "__main__":
    print(string_answer("Where did Sangeetha attend school?"))