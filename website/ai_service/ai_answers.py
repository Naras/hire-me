from datetime import datetime
# from zoneinfo import ZoneInfo
from typing import Union
import os, glob, json, csv

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
def logUsage(row):
    filename = 'token-usage-log.csv'
    header = ['Date and Time', 'Token Usage', 'Tokens Spent', 'Prompt', 'Completion', '$Cost est.']
    # Check if file exists to determine if we need a header
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f)
        # Write header ONLY if the file is newly created
        if not file_exists: writer.writerow(header)
        writer.writerow(row)
def logPitchAnswers(row=None):
    if row == None: return
    filename = 'pitch-me-results-answers-log.csv'
    header = ['Date and Time', 'url-job-description', 'ans_hire_you', 'ans_want_to_work', 'ans_bring_to_table']
    # Check if file exists to determine if we need a header
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        # Write header ONLY if the file is newly created
        if not file_exists: writer.writerow(header)
        writer.writerow(row)
def logPitchTargets(row=None):
    if row == None: return
    filename = 'pitch-me-results-targets-log.csv'
    header = ['Date and Time', 'url-job-description', 'Target', 'Role', 'Region', 'Reason', 'Hiring signal']
    # Check if file exists to determine if we need a header
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        # Write header ONLY if the file is newly created
        if not file_exists: writer.writerow(header)
        writer.writerow(row)
def logPitchFails(row=None):
    if row == None: return
    filename = 'pitch-me-results-errors-log.csv'
    header = ['Date and Time', 'url-job-description', 'Error']
    # Check if file exists to determine if we need a header
    file_exists = os.path.isfile(filename)
    with open(filename, mode='a', newline='') as f:
        writer = csv.writer(f, quoting=csv.QUOTE_MINIMAL)
        # Write header ONLY if the file is newly created
        if not file_exists: writer.writerow(header)
        writer.writerow(row)
def url_parsed_previously(url):
    try:
        with (open(os.path.join('pitch-me-results-answers-log.csv')) as ansf):
            reader = csv.reader(ansf)
            header1 = next(reader)
            for row in reader:
                # print(f'row {row}')
                date_time, url_job_desc, ans_hire_you, ans_want_to_work, ans_bring_table = row
                if url.strip() == url_job_desc.strip():
                    targets = []
                    with (open(os.path.join('pitch-me-results-targets-log.csv')) as trgf):
                        readerTrgf = csv.reader(trgf)
                        header2 = next(reader)
                        for rowTrg in readerTrgf:
                            # print(f'rowTarget {rowTrg}')
                            date_time, url_job_desc_trg, company, role, region, reason, hiring_signal = rowTrg
                            if url.strip() == url_job_desc_trg.strip(): targets.append({'company_name': company, 'role':role, 'region': region, 'reason': reason, 'hiring_signal': hiring_signal})
                    return True, ans_hire_you, ans_want_to_work, ans_bring_table, targets
        return False, None, None, None, None
    except Exception as e:
        print(f'url parse rtn {e}')
        raise SyntaxError(f'url parse rtn {e}')

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
        '''print(f"\n--- [Token Usage: QA Chain] ---")
        print(f"Tokens Spent: {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")
        print(f"Approximate Cost: ${cb.total_cost:.6f}")
        print(f"---------------------------------\n")'''

        logUsage([datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"), 'QA Chain', cb.total_tokens, cb.prompt_tokens, cb.completion_tokens, f'{cb.total_cost:.6f}'])
        # is datetime.now(ZoneInfo("Asia/Kolkata")) required/desirable?

        return str(output["output_text"])

from .scraper import extract_job_description
REGION_OPTIONS = {
    "us": "United States",
    "canada": "Canada",
    "australia": "Australia",
    "new_zealand": "New Zealand",
    "uk": "United Kingdom",
    "europe_english": "Non-English speaking Europe where English-only candidates are commonly hired",
}

def generate_pitch_for_job(url: str, selected_regions: list[str] | None = None, require_h1b: bool = False, always_prompt: bool=False) -> dict:
    if isinstance(selected_regions, bool):
        require_h1b = selected_regions
        selected_regions = None
    if not always_prompt: # fetch from saved previous results in log files
        try:
            url_parsed, ans_hire_you, ans_why_work, ans_bring_table, targets = url_parsed_previously(url)
            if url_parsed:
                clean_json = {'url':url, 'hire_you':ans_hire_you, 'want_to_work':ans_why_work, 'bring_to_table':ans_bring_table, 'target_companies':targets}
                # print(f'saved results with payload:\n{clean_json}')
                return clean_json
        except Exception as e:
            print(f'Failed parse {e}')
            raise ValueError(f'failed parse {e}')
    if always_prompt or not url_parsed:  # get fresh results from LLM service response
        job_description = extract_job_description(url)
        if not job_description:
            logPitchAnswers([datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"), url, 'Error: Could not extract job description', 'N/A', 'N/A', 'N/A'])
            return {"error": "Could not extract job description from the provided URL."}

        # Preserved original behavior: no token reduction / dumps all files into the context
        background_text = "\n\n".join([doc.page_content for doc in sources])

        selected_regions = selected_regions or ["us"]
        region_names = [REGION_OPTIONS[region] for region in selected_regions if region in REGION_OPTIONS]
        if not region_names:
            region_names = [REGION_OPTIONS["us"]]

        region_instructions = "\n".join([
            f"- {region_name}" for region_name in region_names
        ])

        h1b_instructions = "For United States recommendations, include the company's H1B sponsorship likelihood in hiring_signal as one of [Known H1B, Likely H1B, Possible H1B, Unlikely H1B]."
        if require_h1b and "us" in selected_regions:
            h1b_instructions += " IMPORTANT: For United States recommendations, ONLY suggest companies that have a Known H1B or Likely H1B sponsorship history."
        elif require_h1b:
            h1b_instructions += " The H1B filter only applies to United States recommendations; do not apply it to other selected regions."

        non_us_instructions = "For Canada, Australia, New Zealand, the United Kingdom, and Europe recommendations, do not score H1B. Instead, use hiring_signal to summarize English-speaking hiring fit, local work authorization considerations, and whether the company commonly hires internationally or relocates candidates."

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
    Only suggest companies in the selected target regions:
    {region_instructions}
    
    Include non-English speaking European markets when selected, such as the Netherlands, Germany, Sweden, Denmark, Finland, Switzerland, Spain, Portugal, or France, but only when English-only candidates are plausible for the role.
    For each company, provide:
    - "company_name": Name of the company.
    - "role": A similar role title they typically hire for.
    - "region": The selected target region or country where the role/company is relevant.
    - "reason": Why they are a good fit based on the resume.
    - "hiring_signal": Region-appropriate hiring signal.
    
    {h1b_instructions}
    {non_us_instructions}
    
    Respond strictly with a valid JSON object containing exactly these four keys: 
    "hire_you", "want_to_work", "bring_to_table", "target_companies".
    
    "target_companies" must be an array of objects with exactly these keys: "company_name", "role", "region", "reason", "hiring_signal".
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
            '''print(f"\n--- [Token Usage: Job Pitch Generation] ---")
            print(f"Tokens Spent: {cb.total_tokens} (Prompt: {cb.prompt_tokens}, Completion: {cb.completion_tokens})")
            print(f"Approximate Cost: ${cb.total_cost:.6f}")
            print(f"---------------------------------------------\n")'''
            logUsage([datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"), 'Job Pitch Generation',  cb.total_tokens, cb.prompt_tokens, cb.completion_tokens,f'{cb.total_cost:.6f}'])
        try:
            # Since JSON mode is strictly enforced, we directly parse the text safely
            clean_text = response.content.strip()
            clean_json = json.loads(clean_text)
            # print(f"Raw model response Json: {clean_json}")  # Debugging line to inspect the raw response
            rowAnswers = [datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"), url, clean_json.get('hire_you', 'N/A'), clean_json.get('want_to_work', 'N/A'), clean_json.get('bring_to_table', 'N/A')]
            logPitchAnswers(rowAnswers)
            if 'target_companies' in clean_json:
                for company in clean_json['target_companies']:
                    rowTargets = [datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"), url, company.get('company_name','N/A'), company.get('role','N/A'), company.get('region','N/A'),  company.get('reason','N/A'), company.get('hiring_signal','N/A')]
                    logPitchTargets(rowTargets)
            else: logPitchFails([datetime.now().strftime("%Y-%m-%d %H:%M:%S %Z"), url,'No target companies found'])
            return json.loads(clean_text)
        except Exception as e:
            return {
                "error": "Failed to parse response.",
                "details": str(e),
                "raw": response.content
            }

if __name__ == "__main__":
    print(string_answer("Where did Sangeetha attend school?"))
