"""
inference.py — OpenEnv Customer Support Inference Script

MANDATORY environment variables (injected by validator):
    API_BASE_URL   — LLM proxy base URL
    API_KEY        — API key (injected by validator, also accepts HF_TOKEN)
    MODEL_NAME     — model identifier
    SERVER_BASE_URL — OpenEnv server URL (defaults to localhost)

STDOUT FORMAT (strict competition spec):
    [START] task=<n> env=<env> model=<model>
    [STEP]  step=<n> action=<str> reward=<0.00> done=<true|false> error=<msg|null>
    [END]   success=<true|false> steps=<n> score=<0.00> rewards=<r1,r2,...>
"""
import json
import os
import time
from typing import Any, Dict, List, Optional
import requests
from openai import OpenAI
SERVER_BASE_URL = os.environ.get("SERVER_BASE_URL", "http://localhost:7860")
MODEL_NAME      = os.environ.get("MODEL_NAME", "llama-3.1-8b-instant")
API_BASE_URL = os.environ["API_BASE_URL"]
API_KEY      = os.environ["API_KEY"]
)
TASKS = ["easy_classification", "refund_handling", "escalation_decision"]
client = OpenAI(
    api_key=API_KEY,
    base_url=API_BASE_URL,
)
def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)
def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    done_str  = "true" if done else "false"
    error_str = error if error else "null"
    action_clean = action.replace("\n", " ").replace("\r", "")[:120]
    print(
        f"[STEP] step={step} action={action_clean!r} "
        f"reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )
def log_end(success: bool, steps: int, score: float, rewards: List[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} "
        f"score={score:.2f} rewards={rewards_str}",
        flush=True,
    )
def server_reset(task: str) -> Dict[str, Any]:
    resp = requests.post(
        f"{SERVER_BASE_URL}/reset",
        params={"task": task},
        timeout=30,
    )
    resp.raise_for_status()
    return resp.json()
def server_step(action_payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = requests.post(
        f"{SERVER_BASE_URL}/step",
        json=action_payload,
        headers={"Content-Type": "application/json"},
        timeout=60,
    )
    resp.raise_for_status()
    return resp.json()
def call_llm(observation: Dict[str, Any], task_name: str) -> Dict[str, Any]:
    ticket_text   = observation.get("ticket_text", "")
    instructions  = observation.get("instructions", "")
    category_hint = observation.get("category_hint")
    metadata      = observation.get("metadata", {})
    hint_line = f"\nCategory hint: {category_hint}" if category_hint else ""
    system_prompt = (
        "You are an expert customer support AI agent. "
        "Always reply with a valid JSON object (no markdown fences) containing:\n"
        "  response_text      (string)  — your reply to the customer\n"
        "  predicted_category (string)  — one of: delivery, refund, complaint\n"
        "  should_escalate    (boolean) — true if this needs a human agent\n"
        "  confidence         (float)   — your confidence 0.0-1.0\n"
        "Output ONLY the JSON object, no extra text."
    )
    user_prompt = (
        f"Task: {task_name}\n"
        f"Instructions: {instructions}{hint_line}\n\n"
        f"Customer ticket:\n{ticket_text}\n\n"
        "Reply with the JSON action now."
    )
    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            temperature=0.3,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        return json.loads(content.strip())
    except Exception as exc:
        print(f"  [WARN] LLM call failed ({exc}), using fallback.", flush=True)
        return _rule_based_action(ticket_text, task_name, category_hint, metadata)
def _rule_based_action(
    ticket_text: str,
    task_name: str,
    category_hint: Optional[str],
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    text_lower = ticket_text.lower()
    if category_hint:
        category = category_hint
    elif any(w in text_lower for w in ["deliver", "ship", "package", "track", "arrive"]):
        category = "delivery"
    elif any(w in text_lower for w in ["refund", "return", "money back", "reimburse", "cancel"]):
        category = "refund"
    else:
        category = "complaint"
    escalate_words = ["damaged", "defective", "rude", "terrible", "lawsuit", "manager", "broken"]
    should_escalate = any(w in text_lower for w in escalate_words)
    if metadata.get("priority") == "high" and metadata.get("sentiment") == "negative":
        should_escalate = True
    responses = {
        "delivery": (
            "Thank you for reaching out. We sincerely apologize for the inconvenience "
            "with your delivery. We are investigating the status of your shipment right "
            "away and will keep you updated. Please feel free to contact us if you have "
            "any further questions."
        ),
        "refund": (
            "Thank you for contacting us. We understand your concern and sincerely "
            "apologize for the trouble. Your refund request has been received and is "
            "eligible for processing. You can expect the amount to be credited within "
            "5-7 business days. Please let us know if you need any further assistance."
        ),
        "complaint": (
            "We sincerely apologize for the experience you have had. Your feedback is "
            "very important to us and we take such matters seriously. We are looking "
            "into this issue immediately and will ensure it is resolved to your "
            "satisfaction. Thank you for bringing this to our attention."
        ),
    }

    return {
        "response_text": responses.get(category, responses["complaint"]),
        "predicted_category": category,
        "should_escalate": should_escalate,
        "confidence": 0.75,
    }
def run_episode(task_name: str) -> Dict[str, Any]:
    log_start(task=task_name, env="openenv-customer-support", model=MODEL_NAME)

    try:
        reset_data = server_reset(task_name)
    except Exception as exc:
        print(f"  [ERROR] reset failed: {exc}", flush=True)
        log_end(success=False, steps=0, score=0.0, rewards=[])
        return {"task": task_name, "success": False, "steps": 0, "score": 0.0, "rewards": []}
    observation = reset_data.get("observation", {})
    step_num    = 0
    rewards: List[float] = []
    done        = False
    error_msg   = None
    while not done:
        step_num += 1
        action_dict = call_llm(observation, task_name)

        try:
            result = server_step(action_dict)
        except Exception as exc:
            error_msg = str(exc)
            log_step(step_num, str(action_dict.get("response_text", ""))[:80], 0.0, True, error_msg)
            done = True
            break
        reward      = result.get("reward", 0.0)
        done        = result.get("done", True)
        error_msg   = result.get("error")
        observation = result.get("observation", observation)
        rewards.append(reward)
        log_step(
            step=step_num,
            action=str(action_dict.get("response_text", "")),
            reward=reward,
            done=done,
            error=error_msg,
        )
    final_score = round(sum(rewards) / len(rewards), 2) if rewards else 0.0
    success     = len(rewards) > 0 and error_msg is None
    log_end(success=success, steps=step_num, score=final_score, rewards=rewards)
    return {
        "task": task_name, "success": success,
        "steps": step_num, "score": final_score, "rewards": rewards,
    }
def main() -> None:
    print("=" * 60, flush=True)
    print("OpenEnv Customer Support — Inference Runner", flush=True)
    print(f"  SERVER    : {SERVER_BASE_URL}", flush=True)
    print(f"  API_BASE  : {API_BASE_URL}", flush=True)
    print(f"  MODEL     : {MODEL_NAME}", flush=True)
    print(f"  KEY_SOURCE: {'API_KEY' if os.environ.get('API_KEY') else 'HF_TOKEN'}", flush=True)
    print("=" * 60, flush=True)
    all_results = []
    for task in TASKS:
        print(flush=True)
        result = run_episode(task)
        all_results.append(result)
        time.sleep(0.5)
    print(flush=True)
    print("=" * 60, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 60, flush=True)
    total_score = 0.0
    for r in all_results:
        status = "PASS" if r["success"] else "FAIL"
        rewards_str = ",".join(f"{x:.2f}" for x in r["rewards"])
        print(
            f"  [{status}] task={r['task']} steps={r['steps']} "
            f"score={r['score']:.2f} rewards=[{rewards_str}]",
            flush=True,
        )
        total_score += r["score"]
    avg = round(total_score / len(all_results), 2)
    print(f"\n  Average score across all tasks: {avg:.2f}", flush=True)
    print("=" * 60, flush=True)
if __name__ == "__main__":
    main()
