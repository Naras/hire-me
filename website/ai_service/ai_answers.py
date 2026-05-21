from typing import Union

from langchain.chains.qa_with_sources.loading import load_qa_with_sources_chain
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_text_splitters import CharacterTextSplitter

# from CustomLLM import *
# llm = CustomLLM() <<- use a custom LLM

with open('website/sangeetha_resume.txt') as f: file_text_resume = f.read()
with open('website/about_this_app.txt') as f: file_text_about = f.read()
with open('website/biographical.txt') as f: file_text_bio = f.read()
with open('website/interview_faq.txt') as f: file_text_faq = f.read()
with open('website/sangeetha_resume.md') as f: file_text_resume_md = f.read()


sources = [
            Document(page_content=file_text_about, metadata={"source": "About this app"}),
            Document(page_content=file_text_faq, metadata={"source": "Interview FAQ"}),
            Document(page_content=file_text_bio, metadata={"source": "Biographical info"}),
            Document(page_content=file_text_resume, metadata={"source": "Sangeetha's resume"}),
            Document(page_content=file_text_resume_md, metadata={"source": "Sangeetha's resume markdown"})
           ]

source_chunks = []
splitter = CharacterTextSplitter(separator=" ", chunk_size=1024, chunk_overlap=0)
for source in sources:
    for chunk in splitter.split_text(source.page_content):
        source_chunks.append(Document(page_content=chunk, metadata=source.metadata))

search_index = FAISS.from_documents(source_chunks, OpenAIEmbeddings())

# chain = load_qa_with_sources_chain(llm)
chain = load_qa_with_sources_chain(ChatOpenAI(model="gpt-4", temperature=0))


def string_answer(question, test: Union[str, None] = None) -> str:
    if test is not None:
        return test

    return str(
        chain(
            {
                "input_documents": search_index.similarity_search(question, k=4),
                "question": question,
            },
            return_only_outputs=True,
        )["output_text"]
    )

from .scraper import extract_job_description
import json
import re

def generate_pitch_for_job(url: str) -> dict:
    job_description = extract_job_description(url)
    if not job_description:
        return {"error": "Could not extract job description from the provided URL."}
        
    # Gather background info
    background_text = "\n\n".join([doc.page_content for doc in sources])
    
    prompt = f"""
You are Sangeetha. Use the following background information about yourself:
{background_text}

Read the following job description:
{job_description[:4000]}

Answer these three questions convincingly based on your background and the job description:
1. Why should we hire you?
2. Why do you want to work here?
3. What can you bring to the table?

Respond ONLY with a valid JSON object containing exactly these three keys: "hire_you", "want_to_work", "bring_to_table".
"""
    llm = ChatOpenAI(model="gpt-4", temperature=0.2)
    response = llm.invoke(prompt)
    
    try:
        text = response.content
        match = re.search(r'```(?:json)?\n?(.*?)\n?```', text, re.DOTALL)
        if match:
            text = match.group(1)
        return json.loads(text)
    except Exception as e:
        return {"error": "Failed to parse response.", "raw": response.content}

if __name__ == "__main__":
    print(string_answer("Where did Sangeetha attend school?"))
