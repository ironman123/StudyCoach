# app/main.py
import uuid, os, logging, json, time
from google import genai
from google.genai import types
from PIL import Image
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from dotenv import load_dotenv

# --- Observability Imports ---
from prometheus_fastapi_instrumentator import Instrumentator
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.trace import Status, StatusCode

load_dotenv()

# --- 1. Logging Configuration (Human Readable) ---
# We use a custom formatter to make console logs clean and easy to read.
# The complex JSON data is NOT printed here; it is sent silently to Jaeger.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("study_coach")
formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s', datefmt='%H:%M:%S')
handler = logging.StreamHandler()
handler.setFormatter(formatter)
logger.handlers = [handler]
logger.propagate = False

# --- 2. Tracing Configuration (Machine Data) ---
# Setup OpenTelemetry to send traces to the Jaeger container on port 4317
trace.set_tracer_provider(TracerProvider())
otlp_exporter = OTLPSpanExporter(endpoint="http://localhost:4317", insecure=True)
span_processor = BatchSpanProcessor(otlp_exporter)
trace.get_tracer_provider().add_span_processor(span_processor)

tracer = trace.get_tracer("study_coach.tracer")

# --- App Initialization ---
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)
PROCESSED_DIR = "processed"
os.makedirs(PROCESSED_DIR, exist_ok=True)

try:
    client = genai.Client()
except Exception as e:
    logger.error(f"Error initializing Gemini client: {e}")
    # Initialize dummy for safety, though real app should fail hard
    pass

app = FastAPI(title="Personalized Study Coach")

# Enable Prometheus Metrics (exposed at /metrics)
Instrumentator().instrument(app).expose(app)

# Enable Auto-Instrumentation for FastAPI (Traces HTTP requests automatically)
FastAPIInstrumentor.instrument_app(app)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

JOB_STATUS = {}

class QueryRequest(BaseModel):
    user_id: str
    text: str
    session_id: str | None = None

# --- API Endpoints ---

@app.get("/")
def root():
    """
    Health Check Endpoint, to verify the API is responsive.
    """
    return {"message": "Study Coach API is running!", "observability": "active"}

@app.post("/upload")
async def upload_image(user_id: str, file: UploadFile = File(...), bg: BackgroundTasks = None):
    """
    Handles image uploads for study note generation.
    
    This endpoint performs the I/O operation (saving file) in the foreground
    but offloads the heavy AI processing to a background task to keep the API snappy.
    
    Args:
        user_id (str): The ID of the user submitting the image.
        file (UploadFile): The binary image file.
        bg (BackgroundTasks): FastAPI dependency for async task management.
    """

    # Observability: Start a manual trace span. 
    # This allows us to isolate and measure exactly how long the file upload/write takes
    # separate from the total request time.
    with tracer.start_as_current_span("handle_upload_request") as span:
        content = await file.read()
        file_id = str(uuid.uuid4())
        path = os.path.join(UPLOAD_DIR, f"{file_id}_{file.filename}")

        with open(path, "wb") as f:
            f.write(content)
        
        # Add attributes for debugging in Jaeger
        # Telemetry: Attach metadata to the trace.
        # This helps debugging later (e.g., "Are large files causing timeouts?").
        span.set_attribute("user.id", user_id)
        span.set_attribute("file.size", len(content))
        
        logger.info(f"📥 Received file upload: {file.filename} (User: {user_id})")

        # State Management: Update in-memory job status.
        # Note: For production, replace this global dict with Redis/Database for persistence.
        JOB_STATUS[file_id] = {"user_id": user_id, "status": "processing", "notes": None}
        
        # Async Processing: Hand off the heavy lifting (AI analysis) to the background.
        # This allows the user to get a "200 OK" response immediately without waiting for the LLM.
        if bg:
            bg.add_task(orchestrate_image_processing, user_id, path, file_id)
        else:
            # Fallback for testing environments without background task capability
            await orchestrate_image_processing(user_id, path, file_id)

        return {"status": "accepted", "job_id": file_id}

@app.post("/query")
async def query(q: QueryRequest):
    """
    Handles text-based study queries.
    
    This uses a 'Chain of Thought' approach where one agent generates notes
    and a second agent refines them.
    """

    # Observability: Track the duration of the entire text processing pipeline
    with tracer.start_as_current_span("process_text_query"):
        logger.info(f"💬 Processing text query for user {q.user_id}")

        # Step 1: Agent generates the core content
        notes = await llm_agent_generate_notes(q.text, q.session_id or str(uuid.uuid4()))
        # Step 2: Agent enriches content (e.g., defining bold terms, adding analogies)
        final_notes_with_terms = await key_term_explainer_agent(notes)
        return {"session_id": q.session_id or str(uuid.uuid4()), "notes": final_notes_with_terms}

@app.get("/job/{job_id}")
async def job_status(job_id: str):
    """
    Polling Endpoint.
    
    Frontend calls this repeatedly (e.g., every 2s) to check if the 
    background image processing is finished.
    """
    if job_id not in JOB_STATUS:
        raise HTTPException(status_code=404, detail="Job ID not found")
    
    job = JOB_STATUS[job_id]
    
    if job["status"] == "done":
        # Analytics: Only log success when the user actually retrieves the data.
        logger.info(f"👀 User retrieved completed job {job_id}")
        return {"job_id": job_id, "status": job["status"], "notes": job["notes"]}
    elif job["status"] == "error":
        # Return error details so the frontend can display a helpful message
        return {"job_id": job_id, "status": job["status"], "notes": job["notes"]}
    else:
        # Still processing
        return {"job_id": job_id, "status": job["status"]}

# --- Business Logic / Agents ---

async def orchestrate_image_processing(user_id: str, filepath: str, job_id: str):
    # This 'parent' span wraps the entire background workflow
    """
    The 'Conductor' function.
    
    It manages the lifecycle of a background job, chaining multiple AI agents together.
    It handles error propagation, state updates (processing -> done), and file cleanup.
    """

    # Tracing: Create a 'Parent Span'. All subsequent agent calls will be nested 
    # under this span in Jaeger/Trace view, showing the total time taken.
    with tracer.start_as_current_span("background_orchestration") as span:
        span.set_attribute("job.id", job_id)
        logger.info(f"⚙️  Job {job_id}: Orchestration started")
        
        try:
            # Phase 1: Vision (Pixel -> Text)
            # We extract raw text from the image using a vision-capable model.
            text = await ocr_agent(filepath)
            
            # Phase 2: Structuring (Text -> JSON)
            # We turn messy raw text into a clean, structured JSON object.
            notes = await llm_agent_generate_notes(text, job_id)
            
            # Phase 3: Enrichment (JSON -> Enhanced JSON)
            # We perform a second pass to define complex terms found in Phase 2.
            final_notes_with_terms = await key_term_explainer_agent(notes)
            
            # State Update: Mark job as complete in memory.
            # (In production, write this to Redis/Postgres).
            JOB_STATUS[job_id]["notes"] = final_notes_with_terms
            JOB_STATUS[job_id]["status"] = "done"
            
            # Persistence: Save the final result to a physical file for backup.
            filename = os.path.basename(filepath)
            base = os.path.splitext(filename)[0]
            out_path = os.path.join(PROCESSED_DIR, f"notes_{base}.txt")
            with open(out_path, "w", encoding='utf-8') as f:
                f.write(final_notes_with_terms)
            
            logger.info(f"✅ Job {job_id}: Completed successfully!")
            span.set_status(Status(StatusCode.OK))
            
        except Exception as e:
            # Error Handling: If ANY agent fails, we catch it here so the 
            # entire pipeline stops safely and reports the error.
            logger.error(f"❌ Job {job_id}: Failed with error: {e}")
            JOB_STATUS[job_id]["status"] = "error"
            JOB_STATUS[job_id]["notes"] = json.dumps({"error": f"Internal processing failed: {str(e)}"})
            span.record_exception(e)
            span.set_status(Status(StatusCode.ERROR))
        finally:
            # Cleanup: Always delete the temporary upload to save disk space,
            # even if the job failed.
            if os.path.exists(filepath):
                os.remove(filepath)

async def ocr_agent(filepath: str) -> str:
    """
    Agent 1: The Vision Specialist.
    Responsible solely for extracting meaning from visual data.
    """
    with tracer.start_as_current_span("agent_ocr") as span:
        logger.info("   ↳ 👁️  OCR Agent: Analyzing image...")
        global client
        span.set_attribute("file.path", filepath)
        
        try:
            img = Image.open(filepath)
        except Exception as e:
            logger.error(f"Could not open image file: {e}")
            return f"Error: Could not open image file: {e}"

        # Prompt Strategy: We ask for a "summary of meaning" rather than just 
        # raw text extraction to handle diagrams or messy handwriting better.
        prompt = (
            "Analyze this image and provide a detailed explanation. "
            "Describe what the image is of, and if there is any text, extract and summarize its meaning. "
            "Present the explanation using clear Markdown headings and bullet points."
        )

        try:
            start_time = time.time()

            # Configuration: Temperature 0.0 ensures the model is deterministic 
            # and factual, reducing hallucinations in OCR tasks.
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=[img, prompt],
                config=types.GenerateContentConfig(temperature=0.0)
            )
            duration = time.time() - start_time
            span.set_attribute("genai.latency", duration)
            logger.info(f"   ↳ 👁️  OCR Agent: Finished in {duration:.2f}s")
            return response.text
        except Exception as e:
            logger.error(f"Gemini API call failed during OCR: {e}")
            span.record_exception(e)
            return f"Error: Gemini API call failed: {str(e)}"

async def llm_agent_generate_notes(text: str, session_id: str) -> str:
    """
    Agent 2: The Structurer.
    Takes raw text and forces it into a specific JSON Schema.
    """
    with tracer.start_as_current_span("agent_generate_notes") as span:
        logger.info("   ↳ 📝 Notes Agent: Generating study structure...")
        global client

        # Schema Definition: This enforces the output format.
        # The LLM *must* return a JSON with title, summary, key_points, etc.
        notes_schema = types.Schema(
            type=types.Type.OBJECT,
            properties={
                "title": types.Schema(type=types.Type.STRING, description="A concise title."),
                "summary": types.Schema(type=types.Type.STRING, description="Short executive summary."),
                "key_points": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
                "key_terms": types.Schema(type=types.Type.ARRAY, items=types.Schema(type=types.Type.STRING)),
            },
            required=["title", "summary", "key_points"]
        )

        system_instruction = (
            "You are an expert online tutor and professional note-taker. Your task is to analyze the "
            "provided text and generate a set of perfect, comprehensive, and well-structured study "
            "notes. You **MUST** strictly adhere to the provided JSON schema for your entire output."
        )
        prompt = f"Generate perfect study notes from the following text, ensuring the output is a single, valid JSON object:\n\n---\n\n{text}"

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.3,
            response_mime_type="application/json",
            response_schema=notes_schema
        )

        try:
            response = client.models.generate_content(
                model='gemini-2.5-pro',
                contents=prompt,
                config=config,
            )
            return response.text
        except Exception as e:
            logger.error(f"Error in Notes Agent: {e}")
            span.record_exception(e)
            return json.dumps({"error": f"Failed to generate notes: {str(e)}"})

async def key_term_explainer_agent(notes: str) -> str:
    """
    Agent 3: The Deep-Dive Tutor.
    Takes the structured notes, finds the 'key_terms', and generates a glossary.
    """

    with tracer.start_as_current_span("agent_term_explainer") as span:
        logger.info("   ↳ 💡 Explainer Agent: Defining key terms...")
        global client

        # Schema Strategy: We define a NEW schema specifically for the glossary
        # to ensure the explanations are consistent (term, explanation, analogy).    
        system_instruction = (
            "You are an expert academic assistant. Your task is to analyze the provided notes (which are in JSON format), "
            "identify the key academic terms listed in the 'key_terms' array, and generate a glossary section. "
            "Format the output strictly as a Markdown section titled '💡 Key Term Explanations:' "
            "with each term in bold, followed by a concise, simple explanation with appropriate examples. "
            "Do not include any other commentary."
        )
        
        keypoint_schema = types.Schema(
            type=types.Type.ARRAY,
            items=types.Schema(
                type=types.Type.OBJECT,
                properties={
                    "term": types.Schema(type=types.Type.STRING),
                    "explanation": types.Schema(type=types.Type.STRING),
                    "example_or_analogy": types.Schema(type=types.Type.STRING)
                },
                required=["term", "explanation"]
            )
        )

        prompt = f"Analyze the key_terms array in the following notes JSON and generate a very detailed glossary, explaining every term such that it clears the concept for first timers and good for revision sessions:\n\n{notes}"

        config = types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=0.15,
            response_mime_type="application/json",
            response_schema=keypoint_schema
        )
        
        try:
            response = client.models.generate_content(
                model='gemini-2.5-pro', 
                contents=prompt,
                config=config
            )
            explanation_text = response.text

            # Data Assembly: We append the glossary JSON to the original Notes JSON.
            # Note: The frontend must handle parsing these two distinct JSON blocks.
            final_notes = notes + "\n\n---\n\n" + explanation_text
            
            logger.info("   ↳ 💡 Explainer Agent: Finished.")
            return final_notes
            
        except Exception as e:
            logger.error(f"Error in KeyTermExplainerAgent: {e}")
            span.record_exception(e)

            # Graceful Degradation: If this agent fails, return the notes we DO have
            # rather than failing the whole job.
            return notes + "\n\n---\n\n### ⚠️ Explanation Error\nCould not generate key term explanations."