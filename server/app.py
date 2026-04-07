import logging
import os
import sys
from contextlib import asynccontextmanager
from typing import Any, Dict
import uvicorn
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from env.environment import CustomerSupportEnv
from env.models import Action
from env.tasks import list_tasks
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("openenv-server")
_env: CustomerSupportEnv
@asynccontextmanager
async def lifespan(app: FastAPI):
    global _env
    _env = CustomerSupportEnv()
    logger.info("Environment started.")
    yield
    await _env.close()
    logger.info("Environment closed.")
app = FastAPI(
    title="OpenEnv Customer Support",
    description="Simulated customer support ticket system with tasks for classification, refunds, and escalation.",
    version="1.0.0",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.get("/")
async def root() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": "openenv-customer-support",
        "version": "1.0.0",
        "docs": "/docs",
    }
@app.get("/tasks")
async def get_tasks() -> Dict[str, Any]:
    return {"tasks": list_tasks()}
@app.post("/reset")
async def reset(
    task: str = Query(default="easy_classification")
) -> Dict[str, Any]:
    try:
        obs = await _env.reset(task_name=task)
        logger.info("Episode reset. task=%s episode_id=%s", task, obs.metadata.get("episode_id"))
        return {"observation": obs.model_dump()}
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Error in /reset")
        raise HTTPException(status_code=500, detail=str(exc))
@app.post("/step")
async def step(action: Action) -> Dict[str, Any]:
    try:
        result = await _env.step(action)
        logger.info(
            "Step done. step=%d reward=%.4f done=%s",
            result.info.get("step", "?"),
            result.reward,
            result.done,
        )
        return result.model_dump()
    except Exception as exc:
        logger.exception("Error in /step")
        raise HTTPException(status_code=500, detail=str(exc))
@app.get("/state")
async def state() -> Dict[str, Any]:
    return _env.state()
@app.post("/close")
async def close() -> Dict[str, str]:
    await _env.close()
    return {"status": "closed"}
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 7860))
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=port,
        reload=False,
        log_level="info",
    )
