from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Body
from typing import Any, Dict
from .tools import extract_and_clean_from_uploadfile, save_json_to_disk
from .agent import Agent

app = FastAPI(title="Ultimate MCQ Agent")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

agent = Agent()

@app.get('/')
async def health():
    return {"status": "ok", "description": "Ultimate MCQ Agent is running"}

@app.post('/agent/run')
async def run_agent(file: UploadFile, num_questions: int = Form(5), force_summary: bool = Form(False)):
    ok, payload = await extract_and_clean_from_uploadfile(file)
    if not ok:
        return {"error": payload}

    try:
        result = agent.decide_and_run(payload, num_questions=num_questions, force_summary=force_summary)
    except Exception as e:
        return {"error": str(e)}
    # Return the result without automatically saving to disk. If the client
    # wants to persist the result, they should call the /agent/save endpoint
    # (this prevents accidental writes and gives the user control).
    return {"filename": file.filename, "result": result}


@app.post('/agent/save')
async def save_agent_result(filename: str = Form(...), result: Dict[str, Any] = Body(...)):
    """Save a previously returned agent result to disk.

    Accepts a `filename` (string) and the `result` JSON body. Returns the
    saved path on success.
    """
    if not filename:
        raise HTTPException(status_code=400, detail="filename is required")

    try:
        out_path = save_json_to_disk(result, filename)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"failed to save file: {e}")

    return {"saved_to": out_path}
