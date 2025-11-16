import json
import bittensor as bt
from typing import Any, Optional, List
import fastapi
from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from loguru import logger
from langchain_core.messages import AnyMessage, SystemMessage, HumanMessage, AIMessage
from langgraph.graph import MessagesState

from agent.stats import ProjectUsageMetrics, TokenUsageMetrics
from common.sqlite_manager import SQLiteManager
import common.utils as utils


# ===============  openai ================
class ChatCompletionMessage(BaseModel):
    role: str = Field(..., description="Message role: system, user, or assistant")
    content: str = Field(..., description="Message content")

class ChatCompletionRequest(BaseModel):
    id: str = Field(default=None, description="Unique identifier for the request")
    model: str = Field(default="gpt-4o-mini", description="Model to use")
    messages: List[ChatCompletionMessage] = Field(..., description="List of messages")
    stream: bool = Field(default=False, description="Whether to stream responses")
    temperature: float = Field(default=0.0, description="Sampling temperature")
    max_tokens: Optional[int] = Field(default=None, description="Maximum tokens to generate")

class CapacitySynapse(bt.Synapse):
    time_elapsed: int = 0
    response: Optional[dict] = None
class BaseSynapse(bt.Synapse):
    id: str | None = None
    cid_hash: str | None = None
    status_code: int | None = 200
    error: str | None = None
    elapsed_time: float | None = 0.0
    block_height: int | None = 0

class CompletionMessagesMixin:
    """Mixin class for synapses that contain ChatCompletionRequest with messages."""
    
    id: str | None = None
    cid_hash: str | None = None
    block_height: int | None = 0
    elapsed_time: float | None = 0.0
    completion: ChatCompletionRequest | None = None
    
    def to_messages(self) -> list[AnyMessage]:
        """Convert ChatCompletionRequest messages to LangChain message types."""
        if not self.completion:
            return []
        messages = []
        for msg in self.completion.messages:
            if msg.role == "system":
                messages.append(SystemMessage(content=msg.content))
            elif msg.role == "user":
                messages.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                messages.append(AIMessage(content=msg.content))
        return messages
    
    def get_question(self) -> str | None:
        """Extract the last user question from completion messages."""
        if not self.completion:
            return None
        user_messages = [msg for msg in self.completion.messages if msg.role == "user"]
        if not user_messages:
            return None
        return user_messages[-1].content

class SyntheticNonStreamSynapse(BaseSynapse):
    question: str | None = None
    response: str | None = ''

    def get_question(self):
        return self.question

class OrganicStreamSynapse(CompletionMessagesMixin, bt.StreamingSynapse):
    status_code: int | None = 200
    error: str | None = None
    response: str | None = None

    async def process_streaming_response(self, clientResponse: "ClientResponse"):
        # logger.info(f"Processing streaming response: {clientResponse}")
        # logger.info(f"Streaming response success: {clientResponse.ok}, status={clientResponse.status}")

        axon_status_code = clientResponse.headers.get('bt_header_axon_status_code', '200')
        self.status_code = axon_status_code

        buffer = ""
        async for chunk in clientResponse.content.iter_any():
            text = chunk.decode("utf-8", errors="ignore")
            buffer += text
            # logger.info(f"Streaming response part: {text}")
            yield text
        self._buffer = buffer

    def extract_response_json(self, r: "ClientResponse") -> dict:
        return {}
    
    def deserialize(self):
        return ''

class OrganicNonStreamSynapse(CompletionMessagesMixin, BaseSynapse):
    response: str | None = ''

class StatsMiddleware(BaseHTTPMiddleware):
    def __init__(
        self,
        app,
        sqlite_manager: SQLiteManager,
        project_usage_metrics: ProjectUsageMetrics,
        token_usage_metrics: TokenUsageMetrics
    ):
        super().__init__(app)
        self.sqlite_manager = sqlite_manager
        self.project_usage_metrics = project_usage_metrics
        self.token_usage_metrics = token_usage_metrics
        self.allowed_path = [
            '/stats',
            '/stats/data',
            '/stats/token_stats',
            '/CapacitySynapse',
            '/SyntheticNonStreamSynapse',
            '/OrganicNonStreamSynapse',
            '/OrganicStreamSynapse'
        ]

    def handle_stats_html(self):
        with open(f"common/stats_miner.html", "r", encoding="utf-8") as f:
            html = f.read()
        return fastapi.Response(content=html, media_type="text/html")

    def handle_stats_data(self, since_id: int = 0):
        if since_id > 0:
            data = self.sqlite_manager.fetch_newer_than(since_id)
        else:
            data = self.sqlite_manager.fetch_all()

        return fastapi.Response(content=json.dumps({
            "data": data, 
            "usage": self.project_usage_metrics.stats(),
        }), media_type="application/json")
    
    def handle_token_stats(self, latest: str = '2h'):
        # Use utils method to parse time range
        cutoff_timestamp = utils.parse_time_range(latest)
        
        return fastapi.Response(content=json.dumps({
            "token_usage": self.token_usage_metrics.stats(since_timestamp=cutoff_timestamp),
            "time_range": latest if latest else "all",
        }), media_type="application/json")

    async def dispatch(
        self, request: "fastapi.Request", call_next: "RequestResponseEndpoint"
    ) -> fastapi.Response:
        path = request.url.path
        if path not in self.allowed_path:
            return fastapi.Response(status_code=404)

        if path == '/stats':
            return self.handle_stats_html()
        elif path == '/stats/data':
            return self.handle_stats_data(int(request.query_params.get("since_id", 0)))
        elif path == '/stats/token_stats':
            return self.handle_token_stats(request.query_params.get("latest", "2h"))
        return await call_next(request)
class ExtendedMessagesState(MessagesState):
    errored: bool = False
    graphql_agent_hit: bool
    intermediate_graphql_agent_input_token_usage: int
    intermediate_graphql_agent_input_cache_read_token_usage: int
    intermediate_graphql_agent_output_token_usage: int
    block_height: int
    tool_calls: list[str]

class BaseBoardResponse(BaseModel):
    code: int
    message: str

class MetaConfigResponse(BaseBoardResponse):
    data: dict[str, Any]