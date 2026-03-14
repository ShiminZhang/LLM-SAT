import os
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from llmsat.llmsat import setup_logging, get_logger
import logging
import json

setup_logging(level=logging.INFO)
logger = get_logger(__name__)


def _get_gemini_client():
    """
    Initialize the Gemini client using environment variables.
    Requires GOOGLE_API_KEY.
    """
    try:
        from google import genai
    except ImportError as exc:
        raise RuntimeError(
            "The 'google-genai' package is required. Install with: pip install google-genai"
        ) from exc

    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise RuntimeError("GOOGLE_API_KEY is not set in environment.")

    return genai.Client(api_key=api_key)


def get_response_from_gemini(
    prompt: str,
    system_message: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    max_retries: int = 5,
    initial_delay: float = 20.0,
    backoff_factor: float = 2.0,
) -> str:
    """
    Single-turn call to Gemini API with exponential-backoff retry.

    Retryable errors (transient):
      - 503 UNAVAILABLE  (high demand / service temporarily down)
      - 429 RESOURCE_EXHAUSTED (rate limit / quota)
      - 500 INTERNAL     (transient server error)

    Non-retryable errors (bad request, auth, etc.) are re-raised immediately.

    Args:
        max_retries: Maximum number of retry attempts after the first failure.
        initial_delay: Seconds to wait before the first retry.
        backoff_factor: Multiplier applied to the delay after each retry.
    """
    try:
        from google.genai.errors import ServerError
    except ImportError:
        ServerError = Exception  # fallback: retry on any exception

    client = _get_gemini_client()
    chosen_model = model or os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

    config = {"temperature": temperature}
    if system_message:
        config["system_instruction"] = system_message

    RETRYABLE_CODES = {429, 500, 503}
    delay = initial_delay

    for attempt in range(max_retries + 1):
        try:
            response = client.models.generate_content(
                model=chosen_model,
                contents=prompt,
                config=config,
            )

            # Extract text from response
            if hasattr(response, "text"):
                return response.text

            # Fallback: traverse candidates structure
            if hasattr(response, "candidates") and response.candidates:
                for candidate in response.candidates:
                    if hasattr(candidate, "content") and candidate.content:
                        for part in candidate.content.parts:
                            if hasattr(part, "text") and part.text:
                                return part.text

            return str(response)

        except ServerError as e:
            status_code = getattr(e, "status_code", None)
            if status_code not in RETRYABLE_CODES or attempt == max_retries:
                logger.error(
                    f"Gemini API error (code={status_code}, attempt={attempt + 1}): {e}"
                )
                raise
            logger.warning(
                f"Gemini API transient error (code={status_code}, attempt={attempt + 1}/{max_retries + 1}). "
                f"Retrying in {delay:.0f}s... [{e}]"
            )
            time.sleep(delay)
            delay *= backoff_factor


def build_gemini_batch_request(
    prompt: str,
    system_message: Optional[str] = None,
    model: Optional[str] = None,
    temperature: float = 0.7,
    custom_id: str = "req-0001",
) -> Dict[str, Any]:
    """
    Build a single request object for Gemini batch API.
    Returns a dict in Gemini's batch JSONL format.
    """
    chosen_model = model or os.environ.get("GEMINI_MODEL", "gemini-3-flash-preview")

    request = {
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {"temperature": temperature},
    }

    if system_message:
        request["systemInstruction"] = {"parts": [{"text": system_message}]}

    return {"key": custom_id, "request": request, "model": f"models/{chosen_model}"}


def write_gemini_batch_jsonl(requests: List[Dict[str, Any]], output_path: Path) -> None:
    """
    Write batch requests to JSONL file for Gemini batch API.
    """
    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as f:
        for req in requests:
            f.write(json.dumps(req, ensure_ascii=False))
            f.write("\n")


def submit_batch_input(
    file_path: str,
    model: str = "gemini-3-flash-preview",
    block: bool = False,
    poll_interval_seconds: int = 60,
    timeout_seconds: int = 24 * 60 * 60,
) -> str:
    """
    Submit a prepared JSONL batch input file to Gemini Batch API.
    Returns the created batch job name. If block=True, waits for completion.
    """
    client = _get_gemini_client()
    p = Path(file_path)

    if not p.exists():
        raise FileNotFoundError(f"Batch input file not found: {p}")

    # Upload file to Gemini
    logger.info(f"Uploading batch input file: {p}")
    uploaded_file = client.files.upload(file=p)
    logger.info(f"File uploaded: {uploaded_file.name}")

    # Create batch job using file
    batch_job = client.batches.create(
        model=f"models/{model}",
        src=uploaded_file.name,
        config={"display_name": f"batch-{p.stem}"},
    )

    batch_name = batch_job.name
    logger.info(f"Batch job created: {batch_name}")

    if block:
        block_until_completion(
            batch_name,
            poll_interval_seconds=poll_interval_seconds,
            timeout_seconds=timeout_seconds,
        )

    return batch_name


def block_until_completion(
    batch_name: str, poll_interval_seconds: int = 60, timeout_seconds: int = 24 * 60 * 60
) -> str:
    """
    Poll until the batch job reaches a terminal state or timeout occurs.
    Returns the final state string.
    """
    logger.info(f"Waiting for batch job completion: {batch_name}")
    client = _get_gemini_client()
    start = time.time()

    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    while True:
        batch_job = client.batches.get(name=batch_name)
        state = str(batch_job.state) if hasattr(batch_job, "state") else "unknown"

        # Handle enum or string state
        state_str = state.split(".")[-1] if "." in state else state

        logger.info(f"Batch {batch_name} state: {state_str}")

        if state_str in terminal_states:
            return state_str

        if time.time() - start > timeout_seconds:
            raise TimeoutError(
                f"Waiting for batch {batch_name} timed out after {timeout_seconds} seconds."
            )

        time.sleep(poll_interval_seconds)


def wait_for_all_batches(
    batch_names: List[str],
    poll_interval_seconds: int = 30,
    timeout_seconds: int = 24 * 60 * 60,
) -> Dict[str, str]:
    """
    Poll all batch jobs in parallel until all reach terminal states.
    Returns a dict mapping batch_name -> final_state.

    This is more efficient than calling block_until_completion sequentially
    because all batches are checked in each polling cycle.
    """
    logger.info(f"Waiting for {len(batch_names)} batches to complete (parallel polling)")
    client = _get_gemini_client()
    start = time.time()

    terminal_states = {
        "JOB_STATE_SUCCEEDED",
        "JOB_STATE_FAILED",
        "JOB_STATE_CANCELLED",
        "JOB_STATE_EXPIRED",
    }

    pending = set(batch_names)
    results: Dict[str, str] = {}

    while pending:
        newly_completed = []

        for batch_name in pending:
            batch_job = client.batches.get(name=batch_name)
            state = str(batch_job.state) if hasattr(batch_job, "state") else "unknown"
            state_str = state.split(".")[-1] if "." in state else state

            if state_str in terminal_states:
                results[batch_name] = state_str
                newly_completed.append(batch_name)
                logger.info(f"Batch {batch_name} completed: {state_str}")

        for batch_name in newly_completed:
            pending.remove(batch_name)

        if pending:
            logger.info(
                f"Progress: {len(results)}/{len(batch_names)} complete, "
                f"{len(pending)} pending"
            )

            if time.time() - start > timeout_seconds:
                raise TimeoutError(
                    f"Waiting for batches timed out after {timeout_seconds} seconds. "
                    f"Pending: {pending}"
                )

            time.sleep(poll_interval_seconds)

    logger.info(f"All {len(batch_names)} batches completed")
    return results


def download_batch_outputs(batch_name: str, output_path: Path) -> Path:
    """
    Download the completed batch output JSONL to the specified file path.
    Returns the path to the output file.
    """
    logger.info(f"Downloading batch outputs for {batch_name} to {output_path}")

    if not isinstance(output_path, Path):
        output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    client = _get_gemini_client()
    batch_job = client.batches.get(name=batch_name)

    state = str(batch_job.state) if hasattr(batch_job, "state") else "unknown"
    state_str = state.split(".")[-1] if "." in state else state

    if state_str != "JOB_STATE_SUCCEEDED":
        raise RuntimeError(
            f"Batch {batch_name} is not succeeded (current state: {state_str})."
        )

    # Get output file name from batch job destination
    if hasattr(batch_job, "dest") and batch_job.dest:
        output_file_name = batch_job.dest.file_name
    else:
        raise RuntimeError(f"Batch {batch_name} has no output file destination.")

    logger.info(f"Downloading output file: {output_file_name}")

    # Download file content from Gemini
    # For file-based batches, download using client.files.download(file=...)
    try:
        file_content = client.files.download(file=output_file_name)
        if isinstance(file_content, bytes):
            output_path.write_bytes(file_content)
        else:
            output_path.write_text(str(file_content))
        logger.info(f"Downloaded batch output to {output_path}")
    except Exception as e:
        logger.warning(f"File download failed: {e}, trying inlined responses")
        # Alternative: read from inlined responses if available
        if (
            hasattr(batch_job, "dest")
            and batch_job.dest
            and hasattr(batch_job.dest, "inlined_responses")
            and batch_job.dest.inlined_responses
        ):
            responses = batch_job.dest.inlined_responses
            with output_path.open("w", encoding="utf-8") as f:
                for resp in responses:
                    # Convert response object to dict if needed
                    if hasattr(resp, "to_dict"):
                        resp_dict = resp.to_dict()
                    elif hasattr(resp, "__dict__"):
                        resp_dict = {"response": resp.__dict__}
                    else:
                        resp_dict = {"response": str(resp)}
                    f.write(json.dumps(resp_dict, ensure_ascii=False))
                    f.write("\n")
            logger.info(f"Wrote inlined responses to {output_path}")
        else:
            raise RuntimeError(
                f"Failed to download outputs for batch {batch_name}: {e}"
            )

    return output_path


def parse_gemini_batch_response(response_line: Dict[str, Any]) -> Dict[str, Any]:
    """
    Parse a single line from Gemini batch output JSONL.
    Returns a normalized response dict with 'key' and 'response' fields.
    """
    # Gemini batch output format:
    # {"key": "req-0001", "response": {...}} or
    # {"key": "req-0001", "error": {...}}

    if "error" in response_line:
        return response_line

    return response_line
