from hashlib import sha256
import time
from fastapi import APIRouter, Depends, FastAPI, HTTPException, Request
import bittensor as bt
from loguru import logger
from common.protocol import ChatCompletionRequest
from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from neurons.validator import Validator


ALLOWED_SOURCE = ["5FWxwB3DbWvmV9WD2FfojafAw2juiw7MMbc2TQi82SBSgW6Q"]

app = FastAPI()
router = APIRouter()

async def verify_signature(request: Request):
    signature = request.headers.get("Hermes-Sign")
    signed_by = request.headers.get("Hermes-Signed-By")
    time_stamp = request.headers.get("Hermes-Timestamp")

    if not signature or not signed_by or not time_stamp:
        raise HTTPException(status_code=400, detail="Missing required signature headers")

    try:

        body = await request.body()
        body_hash = f"{sha256(body).hexdigest()}"
        logger.info(f"[API] Incoming request body sha256: {body_hash}, signature: {signature}, signed_by: {signed_by}, time_stamp: {time_stamp}")

        if signed_by not in ALLOWED_SOURCE:
            raise HTTPException(status_code=401, detail="Signer not the expected ss58 address")

        now = int(time.time())
        if abs(now - int(time_stamp)) > 300:
            raise HTTPException(status_code=401, detail="Request is too old")

        keypair =  bt.Keypair(ss58_address=signed_by)
        verified = keypair.verify(body_hash, bytes.fromhex(signature))

        if not verified:
            raise HTTPException(status_code=401, detail="Invalid signature")
    except HTTPException as he:
        raise he
        
    except Exception as e:
        import traceback
        logger.error(f"[API] Error verifying signature: {e} {traceback.format_exc()}")
        raise HTTPException(status_code=400, detail="Error verifying signature")

@router.post("/{cid_hash}/chat/completions")
async def chat(
    cid_hash: str, request: Request, body: ChatCompletionRequest, _: dict = Depends(verify_signature)
):
    v: "Validator" = request.app.state.validator
    return await v.forward_miner(cid_hash, body)


@app.get("/health")
def health():
    return {"status": "ok"}

app.include_router(router, prefix="/miners")
