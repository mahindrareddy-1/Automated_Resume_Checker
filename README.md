# Automated Resume Relevance Check System

**Problem:** Innomatics Research Labs screens resumes manually against 18-20 weekly job requirements.
**Solution:** A hybrid evaluation engine that parses resumes (PDF/DOCX/TXT) and job descriptions, scores
relevance 0-100, lists missing skills/projects/certifications, gives a High/Medium/Low verdict and
personalised feedback, with a searchable placement-team dashboard.

## Approach
1. **Parsing** - pypdf / DOCX XML extraction, whitespace + repeated header/footer cleanup.
2. **JD parsing** - role title, must-have vs good-to-have skills (section + keyword aware),
   qualifications, minimum experience, location; multi-role JD files are split.
3. **Hard match (60%)** - skills taxonomy with aliases + fuzzy matching, education, experience.
4. **Soft match (40%)** - TF-IDF-style cosine + JD-term recall; optional LLM fit score.
5. **Verdict** - High >= 70 (and >= 60% of must-haves), Medium >= 45, else Low.

## Run locally
    pip install -r requirements.txt
    flask --app api/index run --debug      # open http://localhost:5000

## Deploy
    npm i -g vercel
    vercel --prod
Optional env vars: `ANTHROPIC_API_KEY` (enables LLM reasoning), `LLM_MODEL`.

## Limitations
Results are stored in the browser (localStorage); requests are limited to ~4 MB per file on Vercel.