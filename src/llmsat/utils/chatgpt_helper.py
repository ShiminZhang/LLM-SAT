import os
import time
from pathlib import Path
from typing import Optional
from llmsat.llmsat import setup_logging, get_logger 
import logging
setup_logging(level=logging.INFO)
logger = get_logger(__name__)

def _get_openai_client():
    """
    Initialize the OpenAI client using environment variables.
    Requires OPENAI_API_KEY. Optionally honors OPENAI_BASE_URL.
    """
    try:
        from openai import OpenAI
    except Exception as exc:
        raise RuntimeError("The 'openai' package is required. Please install it (e.g., pip install openai).") from exc
    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set in environment.")
    base_url = os.environ.get("OPENAI_BASE_URL")
    if base_url:
        return OpenAI(base_url=base_url)
    return OpenAI()

def get_response_from_chatgpt(prompt: str, system_message: Optional[str] = None, model: Optional[str] = None, temperature: float = 0.7) -> str:
    """
    Single-turn call to OpenAI Responses API.
    Returns the text output.
    """
    logger.info(f"Getting response from ChatGPT for prompt: {prompt}")
    client = _get_openai_client()
    chosen_model = model or os.environ.get("OPENAI_MODEL", "gpt-4o")
    messages = []
    if system_message:
        messages.append({"role": "system", "content": system_message})
    messages.append({"role": "user", "content": prompt})
    # Use Responses API (preferred for new SDK)
    resp = client.responses.create(
        model=chosen_model,
        input=messages,
        temperature=temperature,
    )
    logger.info(f"Response from ChatGPT: {resp}")
    # Try multiple extraction strategies for robustness across SDK versions
    text = getattr(resp, "output_text", None)
    if text:
        return text
    # Fallback: try to traverse top-level output structure
    try:
        outputs = getattr(resp, "output", None) or getattr(resp, "outputs", None)
        if outputs and isinstance(outputs, list):
            for item in outputs:
                content = item.get("content") if isinstance(item, dict) else None
                if isinstance(content, list):
                    for part in content:
                        if isinstance(part, dict) and part.get("type") == "output_text":
                            maybe = part.get("text")
                            if maybe:
                                return maybe
    except Exception:
        pass
    # Last resort: string conversion
    return str(resp)

def block_until_completion(batch_id: str, poll_interval_seconds: int = 60, timeout_seconds: int = 24 * 60 * 60) -> str:
    """
    Poll until the batch reaches a terminal state or timeout occurs.
    Returns the final status string.
    """
    logger.info(f"Blocking until completion for batch {batch_id}")
    client = _get_openai_client()
    start = time.time()
    while True:
        batch = client.batches.retrieve(batch_id)
        status = getattr(batch, "status", "unknown")
        if status in {"completed", "failed", "cancelled", "expired"}:
            return status
        if time.time() - start > timeout_seconds:
            raise TimeoutError(f"Waiting for batch {batch_id} timed out after {timeout_seconds} seconds.")
        time.sleep(poll_interval_seconds)

def submit_batch_input(file_path: str, block: bool = False, poll_interval_seconds: int = 60, timeout_seconds: int = 24 * 60 * 60) -> str:
    """
    Submit a prepared JSONL batch input file to OpenAI Batches API.
    Returns the created batch_id. If block=True, waits for completion before returning.
    """
    client = _get_openai_client()
    p = Path(file_path)
    if not p.exists():
        raise FileNotFoundError(f"Batch input file not found: {p}")
    with p.open("rb") as fh:
        uploaded = client.files.create(file=fh, purpose="batch")
    batch = client.batches.create(
        input_file_id=uploaded.id,
        endpoint="/v1/responses",
        completion_window="24h",
        metadata={"source_path": str(p)},
    )
    batch_id = getattr(batch, "id", None)
    logger.info(f"Batch created with ID: {batch_id}")
    if not batch_id:
        raise RuntimeError("Failed to create batch; no batch id returned.")
    if block:
        block_until_completion(batch_id, poll_interval_seconds=poll_interval_seconds, timeout_seconds=timeout_seconds)
    return batch_id

def download_batch_outputs(batch_id: str, output_path: Path) -> Path:
    """
    Download the completed batch output JSONL to the specified file path.
    If an error file exists, write it next to the output with '.errors.jsonl' suffix.
    Returns the path to the output file.
    """
    logger.info(f"Downloading batch outputs for batch {batch_id} to {output_path}")
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_openai_client()
    batch = client.batches.retrieve(batch_id)
    status = getattr(batch, "status", None)
    if status != "completed":
        logger.warning(f"Batch {batch_id} is not completed (current status: {status}). Attempting to retrieve error details.")
        error_file_id = getattr(batch, "error_file_id", None)
        error_path = output_path.with_suffix(".errors.jsonl")
        if error_file_id:
            try:
                err_resp = client.files.content(error_file_id)
                err_text = getattr(err_resp, "text", None)
                if err_text is None:
                    err_chunk = err_resp.read()
                    if isinstance(err_chunk, bytes):
                        error_path.write_bytes(err_chunk)
                    else:
                        error_path.write_text(str(err_chunk))
                else:
                    error_path.write_text(err_text)
                logger.info(f"Saved batch error details to {error_path}")
            except Exception as exc:
                logger.error(f"Failed to download error file for batch {batch_id}: {exc}")
        else:
            logger.warning(f"Batch {batch_id} has no error_file_id to download.")
        return error_path
    logger.info(f"Batch {batch_id} is completed (current status: {status}).")
    output_file_id = getattr(batch, "output_file_id", None)
    if not output_file_id:
        raise RuntimeError(f"Batch {batch_id} has no output_file_id.")
    logger.info(f"Batch {batch_id} has output_file_id: {output_file_id}")
    resp = client.files.content(output_file_id)
    # logger.info(f"Number of lines in Response from ChatGPT: {len(resp.splitlines())}")
    try:
        text = getattr(resp, "text", None)
        if text is None:
            chunk = resp.read()
            if isinstance(chunk, bytes):
                output_path.write_bytes(chunk)
            else:
                output_path.write_text(str(chunk))
        else:
            output_path.write_text(text)
    except Exception:
        try:
            chunk = resp.read()
            if isinstance(chunk, bytes):
                output_path.write_bytes(chunk)
            else:
                output_path.write_text(str(chunk))
        except Exception as exc:
            raise RuntimeError(f"Failed to download output for batch {batch_id}") from exc

    error_file_id = getattr(batch, "error_file_id", None)
    if error_file_id:
        err_resp = client.files.content(error_file_id)
        error_path = output_path.with_suffix(".errors.jsonl")
        try:
            err_text = getattr(err_resp, "text", None)
            if err_text is None:
                err_chunk = err_resp.read()
                if isinstance(err_chunk, bytes):
                    error_path.write_bytes(err_chunk)
                else:
                    error_path.write_text(str(err_chunk))
            else:
                error_path.write_text(err_text)
        except Exception:
            pass
    return output_path