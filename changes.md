A. PDF Data Source Integration
   ---------------------------
Updated AI backend to dynamically ingest all PDF files located in your website/static folder as additional context.

Changes Made
1. Dependency Updates
Added pypdf (version 5.0.1) to your requirements.txt. This package is the foundational tool that LangChain uses internally for high-fidelity PDF text extraction.
2. Loading and Ingestion
Modified website/ai_service/ai_answers.py to import os, glob, and PyPDFLoader from LangChain.
Added a dynamic scan that locates all files ending in .pdf inside your website/static directory.
For each PDF discovered (including your case studies and resume), the system securely parses the raw text page-by-page.
The extracted text from these PDFs is then automatically appended as Document objects to the existing sources array before it runs through the CharacterTextSplitter. This ensures they are properly chunked and safely ingested into the FAISS semantic search index, preventing token overflows even for large or poorly-formatted case studies.
How it Works Now
Whenever a recruiter uses the QA feature or the "Pitch Me" generator, the language model can seamlessly draw references, insights, and facts from the newly parsed static PDFs to form stronger, more nuanced answers.

Verification
The script successfully compiled and built the FAISS index with the new files.
You can test this by running python app.py and asking a very specific question that only exists within one of your case study PDFs.

B. Target Companies & H1B Scoring
   ------------------------------ 
Implemented the feature to generate recommended target companies based on the seed job description and your resume. It also includes the optional H1B scoring filter you requested!

Changes Made
1. AI Service Logic Update
Updated generate_pitch_for_job in website/ai_service/ai_answers.py to accept a new require_h1b boolean parameter.
Modified the LLM prompt to instruct it to act as a career advisor. In addition to answering the three core pitch questions, it now analyzes the seed job description and suggests 3-5 alternative companies that are a great fit for your resume.
It dynamically computes an h1b_score for each target company based on historical trends (scoring them as Known, Likely, Possible, or Unlikely).
If the require_h1b flag is enabled, the prompt enforces strict filtering so the LLM only outputs companies with a "Known" or "Likely" history of sponsorship.
2. Frontend UI Enhancements
Form update: Added a "Require H1B Sponsorship?" checkbox to JobURLForm in website/core/forms.py using a WTForms BooleanField.
View handling: Passed the checkbox's state down to the AI generation logic in website/core/views.py.
Results Template: Beautifully expanded website/templates/index.html. After displaying your custom pitch, it now renders a responsive grid of "Recommended Target Companies". Each card displays the company name, a suggested role, a personalized reason why it fits your background, and a color-coded badge indicating its H1B sponsorship likelihood (e.g., Green for Known, Yellow for Possible, Grey for Unlikely).
How to Test Manually
Start the app with python app.py.
Navigate to http://localhost:5000/.
In the Pitch Me for a Job section, paste your seed URL (e.g., the Sentry URL).
Test with the Require H1B Sponsorship? checkbox unchecked and observe the list of companies generated (you may see a mix of startups and large corps).
Test again with the Require H1B Sponsorship? checkbox checked to verify that the LLM aggressively filters for companies known for robust H1B pipelines (like FAANG or major tech enterprises).

C. OpenAi Token reduction
   ----------------------
   a) Token Tracking and Remaining Credits
    Tokens Spent Count
    Added LangChain's get_openai_callback() context manager inside both functions. Every time you run string_answer() or generate_pitch_for_job(), you will see a clean printout in your terminal/logs showing:

        Total Tokens used.
        Prompt (Input) Tokens used.
        Completion (Output) Tokens used.
        Approximate cost of the transaction in USD based on OpenAI's standard pricing.

b) Other Strategies to Reduce Token Consumption
        Beyond removing the duplicate resume, you can apply these techniques to make your application ultra-efficient:
        RAG-based Pitch Generation (Implemented Above): * Old approach: You were putting all background text files and every PDF page inside generate_pitch_for_job (which scaled up to over 12,000 tokens instantly).
        New approach: Now performs a quick vector similarity search against the incoming Job Description to retrieve only the top 5 most relevant text chunks (roughly 2,000 tokens total). This keeps your prompts extremely compact.

        Clean Scraped Job Descriptions:
        Scraped raw job page content contains a lot of "garbage" tokens (headers, footers, script code, cookies, CSS, legal boilerplate).
        Update extract_job_description(url) function to strip out non-alphanumeric text, HTML elements, and navigation blocks before feeding it to Python code.

        Lowering chunk_size and k in Vector Search:
        If your context is still slightly large, you can adjust chunk_size in your CharacterTextSplitter down to 512 or 750 characters.
        In string_answer(), you can also change the search depth from k=4 to k=2 or k=3 if the answer can usually be found in a single page or paragraph.

