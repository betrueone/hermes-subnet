# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# Copyright © 2023 <your name>

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the “Software”), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

import asyncio
import json
import os
from pathlib import Path
import time
import bittensor as bt
from langchain_openai import ChatOpenAI
from loguru import logger
from loguru._logger import Logger
from bittensor.core.stream import StreamingSynapse
from agent.stats import Phase, ProjectUsageMetrics, TokenUsageMetrics
from common.prompt_template import get_miner_self_tool_prompt, fill_miner_self_tool_prompt
from langchain_core.messages import HumanMessage, SystemMessage
from common.table_formatter import table_formatter
from common.agent_manager import AgentManager
from common.enums import ErrorCode, RoleFlag
from common.logger import HermesLogger
from common.protocol import CapacitySynapse, OrganicNonStreamSynapse, OrganicStreamSynapse, StatsMiddleware, SyntheticNonStreamSynapse
from common.sqlite_manager import SQLiteManager
import common.utils as utils
from common.settings import settings
from hermes.base import BaseNeuron

ROLE = "miner"

settings.load_env_file(ROLE)
LOGGER_DIR = os.getenv("LOGGER_DIR", f"logs/{ROLE}")

WHITELISTED_VALIDATORS = [
    "5C5XL8kUqwzZ9WQBrppjTxtMkNZHUKtzEq8Ath4iDWm6RUEa",
    "5CDgbBhSpePngE1Ef3LTvfu3opMD2wEXn4NqfUfJXDm5Ks82",
    "5C9yALD6gjxhzyawXomy4E6ykcYm8bKXTJrGnQNa498Hsn82",
    "5CsvRJXuR955WojnGMdok1hbhffZyB4N5ocrv82f3p5A2zVp",
    "5FLoWCDovMPeH3Gv4syQSZ8TuKcMv6N27g8diDU8zJSeRv8m",
    "5G9hfkx9wGB1CLMT9WXkpHSAiYzjZb5o1Boyq4KAdDhjwrc5",
    "5E2LP6EnZ54m3wS8s1yPvD5c3xo71kQroBw7aUVK32TKeZ5u",
    "5Gn3dRM5C6KjZ6u46PcjU54cYsmyKRtsM8TQZpcn8s1CNEYm",
]

class Miner(BaseNeuron):

    @property
    def role(self) -> str:
        return ROLE

    def __init__(self, config_loguru: bool = True):
        if config_loguru:
            HermesLogger.configure_loguru(
                file=f"{LOGGER_DIR}/{self.role}.log",
                error_file=f"{LOGGER_DIR}/{self.role}_error.log"
            )
        super().__init__()

    async def start(self):
        try:
            super().start(flag=RoleFlag.MINER)

            self.project_usage_metrics = ProjectUsageMetrics()
            self.token_usage_metrics = TokenUsageMetrics()

            self.db_queue = asyncio.Queue()
            self.sqlite_manager = SQLiteManager(f".data/{self.role}.db")
            self.axon = bt.axon(
                wallet=self.settings.wallet,
                port=self.settings.port,
                ip=self.settings.external_ip,
                external_ip=self.settings.external_ip,
                external_port=self.settings.port
            )
            self.axon.app.add_middleware(
                StatsMiddleware,
                sqlite_manager=self.sqlite_manager,
                project_usage_metrics=self.project_usage_metrics,
                token_usage_metrics=self.token_usage_metrics,
            )

            def _reject_if_not_whitelisted(synapse) -> None:
                hotkey = getattr(synapse.dendrite, "hotkey", None)
                if not hotkey or hotkey not in WHITELISTED_VALIDATORS:
                    raise ValueError(
                        f"Rejecting request from non-whitelisted validator: {hotkey}"
                    )

            def verify_organic_stream(synapse: OrganicStreamSynapse) -> None:
                _reject_if_not_whitelisted(synapse)

            def verify_organic_non_stream(synapse: OrganicNonStreamSynapse) -> None:
                _reject_if_not_whitelisted(synapse)

            def verify_synthetic(synapse: SyntheticNonStreamSynapse) -> None:
                _reject_if_not_whitelisted(synapse)

            def verify_capacity(synapse: CapacitySynapse) -> None:
                _reject_if_not_whitelisted(synapse)

            self.axon.attach(
                forward_fn=self.forward_organic_stream,
                verify_fn=verify_organic_stream,
            )

            self.axon.attach(
                forward_fn=self.forward_organic_non_stream,
                verify_fn=verify_organic_non_stream,
            )

            self.axon.attach(
                forward_fn=self.forward_synthetic_non_stream,
                verify_fn=verify_synthetic,
            )

            self.axon.attach(
                forward_fn=self.forward_capacity,
                verify_fn=verify_capacity,
            )

            # self.axon.serve(netuid=self.settings.netuid, subtensor=self.settings.subtensor)

            self.axon.start()
            logger.info(f"Miner starting at block: {self.settings.subtensor.block}")
            logger.info(f"Axon serving on port: {self.settings.port}")
            logger.info(f"Axon created: {self.axon}")
            logger.info(f"Miner starting at block: {self.settings.subtensor.block}")
            logger.info(f"Stats at: http://{self.settings.external_ip}:{self.settings.port}/stats")

            self.agent_manager = None
            self._running_tasks = [
                asyncio.create_task(
                    self.refresh_agents()
                ),
                asyncio.create_task(
                    self.profile_tools_stats()
                ),
                asyncio.create_task(
                    self.db_writer()
                )
            ]

            await asyncio.gather(*self._running_tasks)
        except KeyboardInterrupt:
            logger.info("[Miner] Miner start process interrupted by user")
            # Cancel all running tasks
            if hasattr(self, '_running_tasks'):
                for task in self._running_tasks:
                    if not task.done():
                        task.cancel()
                # Wait for tasks to complete cancellation
                await asyncio.gather(*self._running_tasks, return_exceptions=True)
            logger.info("[Miner] All tasks cancelled successfully")
            raise  # Re-raise to allow graceful shutdown at higher level
        except Exception as e:
            logger.error(f"[Miner] Failed to start miner: {e}")
            raise

    async def db_writer(self):
        try:
            last_check_time = 0

            while True:
                if int(time.time()) - last_check_time > 60 * 10:
                    self.sqlite_manager.cleanup_old_records()
                    last_check_time = int(time.time())

                item = await self.db_queue.get()

                type = item.get("type")
                status_code = item.get("status_code")
                project_id = item.get("project_id")

                target = self.project_usage_metrics.synthetic_project_usage if type == 0 else self.project_usage_metrics.organic_project_usage
                target.incr(
                    project_id,
                    success=False if status_code != 200 else True
                )

                tool_hit = item.get("tool_hit")

                logger.info(f"[DB Writer] - Inserting request log for project {project_id} with status code {status_code}, type:{type}, tool_hit: {tool_hit}")

                if tool_hit and tool_hit != '[]':
                    tool_hit_list = json.loads(tool_hit)
                    target = self.project_usage_metrics.synthetic_tool_usage if type == 0 else self.project_usage_metrics.organic_tool_usage
                    for tool_name, count in tool_hit_list:
                        target.incr(tool_name, count)

                self.sqlite_manager.insert_request(**item)
                self.db_queue.task_done()
        except KeyboardInterrupt:
            logger.info("[Miner] DB writer interrupted by user")
            raise  # Re-raise to allow graceful shutdown
        except Exception as e:
            logger.error(f"[Miner] DB writer error: {e}")
            raise

    async def _handle_task(
            self,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse,
            log: Logger,
    ) -> SyntheticNonStreamSynapse | OrganicNonStreamSynapse:
        
        before = time.perf_counter()
        question = task.get_question()

        cid_hash = task.cid_hash
        graph, graphql_agent = self.agent_manager.get_miner_agent(cid_hash)

        if graphql_agent:
            response, _, _ = await graphql_agent.query_no_stream(
                question,
                prompt_cache_key=f"{cid_hash}_{before}",
                is_synthetic=True,
                block_height=task.block_height
            )
            messages = response.get('messages', [])
            result = messages[-1].content if messages else None
            if result:
                task.response = result
                task.status_code = ErrorCode.SUCCESS
                task.error = None
                task.elapsed_time = utils.fix_float(time.perf_counter() - before)

                print(f"🔥🔥🔥 task returning: {task}")

                self._schedule_postprocess(
                    answer=result,
                    usage_info={},
                    tool_hit=[],
                    graphql_agent_inner_tool_calls=[],
                    error=None,
                    status_code=ErrorCode.SUCCESS,
                    tag="GraphQL Agent",
                    task=task,
                    elapsed=task.elapsed_time,
                    log=log,
                    type=0,
                )
                return task

        if isinstance(task, SyntheticNonStreamSynapse):
            tag = "Synthetic"
            type = 0
            is_synthetic = True
            phase = Phase.MINER_SYNTHETIC
            messages = [
                SystemMessage(content=get_miner_self_tool_prompt(block_height=task.block_height, node_type=graphql_agent.config.node_type if graphql_agent else "unknown")),
                HumanMessage(content=question)
            ]

        elif isinstance(task, OrganicNonStreamSynapse):
            tag = "Organic"
            type = 1
            is_synthetic = False
            phase = Phase.MINER_ORGANIC_NONSTREAM
            messages = [SystemMessage(content=get_miner_self_tool_prompt(block_height=task.block_height, node_type=graphql_agent.config.node_type if graphql_agent else "unknown"))] + task.to_messages()

        else:
            raise ValueError("Unsupported task type")

        answer = None
        usage_info = {}
        tool_hit = []
        graphql_agent_inner_tool_calls = []
        response = None
        error = None
        status_code = ErrorCode.SUCCESS


        try:
            if not graph:
                log.warning(f"[{tag}] - {task.id} No agent found for project {cid_hash}")
                error = f"No agent found for project {cid_hash}"
                status_code = ErrorCode.AGENT_NOT_FOUND
            else:
                r = await graph.ainvoke({"messages": messages, "block_height": task.block_height})
                (
                    answer,
                    usage_info,
                    tool_hit,
                    graphql_agent_inner_tool_calls,
                    response,
                    error,
                    status_code
                ) = self.get_answer(phase, task, r)

        except Exception as e:
            log.error(f"handle task error {task.id} - {question}. {e}\n")
            error = str(e)
            status_code = ErrorCode.INTERNAL_SERVER_ERROR

        elapsed = utils.fix_float(time.perf_counter() - before)
        
        self._schedule_postprocess(
            answer=answer,
            usage_info=usage_info,
            tool_hit=tool_hit,
            graphql_agent_inner_tool_calls=graphql_agent_inner_tool_calls,
            error=error,
            status_code=status_code,
            tag=tag,
            task=task,
            elapsed=elapsed,
            log=log,
            type=type,
        )

        task.response = response
        task.error = error
        task.status_code = status_code.value
        task.usage_info = usage_info
        task.graphql_agent_inner_tool_calls = graphql_agent_inner_tool_calls
        task.miner_model_name = self.llm.model_name
        task.graphql_agent_model_name = graphql_agent.llm.model_name

        return task

    def get_answer(
        self,
        phase: Phase,
        task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse | OrganicStreamSynapse,
        r: dict
    ) -> tuple[str | None, dict, list, list[str], str | None, str | None, ErrorCode]:
        # logger.info(f"[{tag}] - {task.id} Agent response: {r}")
        
        usage_info = self.token_usage_metrics.parse(task.cid_hash, phase, r)
        self.token_usage_metrics.append(usage_info)

        # check tool stats
        tool_hit = utils.try_get_tool_hit(
            r.get('messages', []),
        )

        if r.get('graphql_agent_hit', False):
            tool_hit.append(("graphql_agent_tool", 1))

        graphql_agent_inner_tool_calls: list[str] = r.get('tool_calls', [])

        error = None
        status_code = ErrorCode.SUCCESS

        answer = None
        if r.get('error', None) is not None:
            error = r.get('error')
            status_code = ErrorCode.LLM_ERROR
        else:
            answer = r.get('messages')[-1].content or None
            if not answer:
                error = utils.try_get_invalid_tool_messages(r.get('messages', []))
                status_code = ErrorCode.TOOL_ERROR if error is not None else status_code

        response = answer if status_code == ErrorCode.SUCCESS else None
        
        return answer, usage_info, tool_hit, graphql_agent_inner_tool_calls, response, error, status_code
        
    def _schedule_postprocess(
            self,
            answer: str,
            usage_info: str,
            tool_hit: list,
            graphql_agent_inner_tool_calls: list,
            error: str,
            status_code: ErrorCode,
            tag: str,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse | OrganicStreamSynapse,
            elapsed: float,
            log: Logger,
            type: int,
    ) -> None:
        try:
            asyncio.create_task(
                self._post_process_async(
                    answer=answer,
                    usage_info=usage_info,
                    tool_hit=tool_hit,
                    graphql_agent_inner_tool_calls=graphql_agent_inner_tool_calls,
                    error=error,
                    status_code=status_code,
                    tag=tag,
                    task=task,
                    elapsed=elapsed,
                    log=log,
                    type=type,
                )
            )
        except RuntimeError:
            # If no running loop (should be rare), run synchronously.
            self._post_process_sync(
                answer=answer,
                usage_info=usage_info,
                tool_hit=tool_hit,
                graphql_agent_inner_tool_calls=graphql_agent_inner_tool_calls,
                error=error,
                status_code=status_code,
                tag=tag,
                task=task,
                elapsed=elapsed,
                log=log,
                type=type,
            )

    async def _post_process_async(
            self,
            answer: str,
            usage_info: str,
            tool_hit: list,
            graphql_agent_inner_tool_calls: list,
            error: str,
            status_code: ErrorCode,
            tag: str,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse | OrganicStreamSynapse,
            elapsed: float,
            log: Logger,
            type: int,
    ) -> None:
        self.put_db(
            type=type,
            answer=answer,
            usage_info=usage_info,
            tool_hit=tool_hit,
            error=error,
            status_code=status_code,
            elapsed=elapsed,
            task=task,
        )
        await asyncio.to_thread(
            self.print_table,
            answer=answer,
            usage_info=usage_info,
            tool_hit=tool_hit,
            graphql_agent_inner_tool_calls=graphql_agent_inner_tool_calls,
            error=error,
            status_code=status_code,
            tag=tag,
            task=task,
            elapsed=elapsed,
            log=log,
        )

    def _post_process_sync(
            self,
            answer: str,
            usage_info: str,
            tool_hit: list,
            graphql_agent_inner_tool_calls: list,
            error: str,
            status_code: ErrorCode,
            tag: str,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse | OrganicStreamSynapse,
            elapsed: float,
            log: Logger,
            type: int,
    ) -> None:
        self.put_db(
            type=type,
            answer=answer,
            usage_info=usage_info,
            tool_hit=tool_hit,
            error=error,
            status_code=status_code,
            elapsed=elapsed,
            task=task,
        )
        self.print_table(
            answer=answer,
            usage_info=usage_info,
            tool_hit=tool_hit,
            graphql_agent_inner_tool_calls=graphql_agent_inner_tool_calls,
            error=error,
            status_code=status_code,
            tag=tag,
            task=task,
            elapsed=elapsed,
            log=log,
        )

    def print_table(
            self,
            answer: str,
            usage_info: str,
            tool_hit: list,
            graphql_agent_inner_tool_calls: list,
            error: str,
            status_code: ErrorCode,
            tag: str,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse | OrganicStreamSynapse,
            elapsed: float,
            log: Logger,
    ):
        tool_hit_names = [t[0] for t in tool_hit]
        rows = [f"💬 Answer: {answer}\n"]
        if error:
            rows.append(f"⚠️ {status_code.value} | {error}\n")
        
        rows.append(f" 📊 Metrics Data: {usage_info}\n")

        if os.getenv("ENABLE_GRAPHQL_AGENT_TOOL_CALLS_LOG", "false").lower() == "true":
            rows.append(f" 📊 GraphQL Agent tools: {graphql_agent_inner_tool_calls}\n")

        if len(tool_hit_names) > 0:
            rows.append(f"🛠️ Tools Hit: {', '.join(tool_hit_names)}\n")
        rows.append(f"⏱️ Cost: {elapsed}s")
        
        status_icon = "✅" if status_code == ErrorCode.SUCCESS else "❌"
        output = table_formatter.create_single_column_table(
            f"🤖 {status_icon} {tag}: {task.get_question()} ({task.id})",
            rows,
            caption=task.cid_hash
        )
        log.info(f"\n{output}")

    def put_db(
            self,
            type: int,
            answer: str,
            usage_info: str,
            tool_hit: list,
            error: str | None,
            status_code: ErrorCode,
            elapsed: float,
            task: SyntheticNonStreamSynapse | OrganicNonStreamSynapse | OrganicStreamSynapse,
    ):
        response_data = answer if status_code == ErrorCode.SUCCESS else error
        
        self.db_queue.put_nowait({
            "type": type,
            "source": task.dendrite.hotkey,
            "task_id": task.id,
            "project_id": task.cid_hash,
            "cid": task.cid_hash,
            "request_data": task.get_question(),
            "response_data": response_data or '',
            "status_code": status_code.value,
            "tool_hit": json.dumps(tool_hit),
            "cost": elapsed,
            "token_usage_info": json.dumps(usage_info) if usage_info else ''
        })

    async def forward_synthetic_non_stream(self, task: SyntheticNonStreamSynapse) -> SyntheticNonStreamSynapse:
        print(f"forward_synthetic_non_stream: {task.get_question()}")
        log = logger.bind(source=task.dendrite.hotkey)
        await self._handle_task(task, log)
        return task

    async def forward_organic_stream(self, synapse: OrganicStreamSynapse) -> StreamingSynapse.BTStreamingResponse:
        print(f"forward_organic_stream: {synapse}")
        from starlette.types import Send
        log = logger.bind(source=synapse.dendrite.hotkey)
        log.info(f"\n🤖 [Miner] Received organic stream: {synapse.id}")

        messages = synapse.to_messages()
        graph, graphql_agent = self.agent_manager.get_miner_agent(synapse.cid_hash)

        if not graph:
            error_msg = f"Error: No agent found for project {synapse.cid_hash}"
            log.warning(f"[Miner] - {synapse.id} {error_msg}")
            async def error_streamer(send: Send):
                error_line = json.dumps({
                    "type": "data",
                    "data": error_msg
                }) + "\n"
                await send({
                    "type": "http.response.body",
                    "body": error_line.encode('utf-8'),
                    "more_body": False
                })
            
            return synapse.create_streaming_response(error_streamer)
        
        fill_miner_self_tool_prompt(messages, block_height=synapse.block_height, node_type=graphql_agent.config.node_type if graphql_agent else "unknown")

        async def token_streamer(send: Send):
            r = None
            tag = "Organic-S"
            phase = Phase.MINER_ORGANIC_STREAM
            before = time.perf_counter()
            async for event in graph.astream(
                {
                    "messages": messages,
                    "block_height": synapse.block_height
                },
                version="v2"
            ):
                for key, value in event.items():
                    if key == "final":
                        r = value
                        message = value.get("messages", [])[-1].content

                        if r.get('error', None) is not None:
                            message = r.get('error')
                        chunk_size = int(os.getenv("STREAM_CHUNK_SIZE", 256))
                        chunk_size = max(1, chunk_size)
                        chunk_delay = float(os.getenv("STREAM_CHUNK_DELAY", "0"))
                        idx = 0
                        while idx < len(message):
                            chunk = message[idx:idx + chunk_size]
                            # Send data chunks in JSONL format
                            data_line = json.dumps({
                                "type": "data",
                                "data": chunk
                            }) + "\n"
                            await send({
                                "type": "http.response.body",
                                "body": data_line.encode('utf-8'),
                                "more_body": True
                            })
                            if chunk_delay > 0:
                                await asyncio.sleep(chunk_delay)
                            idx += chunk_size
                
            elapsed = utils.fix_float(time.perf_counter() - before)
            synapse.elapsed_time = elapsed
            (
                answer,
                usage_info,
                tool_hit,
                graphql_agent_inner_tool_calls,
                response,
                error,
                status_code
            ) = self.get_answer(phase, synapse, r)

            # Send metadata in JSONL format
            metadata_line = json.dumps({
                "type": "meta",
                "data": {
                    "miner_model_name": self.llm.model_name,
                    "graphql_agent_model_name": graphql_agent.llm.model_name,
                    "elapsed": elapsed,
                    "status_code": status_code.value,
                    "error": error,
                    "graphql_agent_inner_tool_calls": graphql_agent_inner_tool_calls,
                    "usage_info": usage_info
                }
            }) + "\n"
            await send({
                "type": "http.response.body",
                "body": metadata_line.encode('utf-8'),
                "more_body": False
            })

            self._schedule_postprocess(
                answer=answer,
                usage_info=usage_info,
                tool_hit=tool_hit,
                graphql_agent_inner_tool_calls=graphql_agent_inner_tool_calls,
                error=error,
                status_code=status_code,
                tag=tag,
                task=synapse,
                elapsed=elapsed,
                log=log,
                type=2,
            )

        return synapse.create_streaming_response(token_streamer)

    async def forward_organic_non_stream(self, task: OrganicNonStreamSynapse) -> OrganicNonStreamSynapse:
        print(f"forward_organic_non_stream: {task}")
        log = logger.bind(source=task.dendrite.hotkey)
        await self._handle_task(task, log)
        return task
        
    async def forward_capacity(self, synapse: CapacitySynapse) -> CapacitySynapse:
        logger.debug(f"[Miner] Received capacity request")
        if not self.agent_manager:
            logger.warning(f"[Miner] No agent manager found")
            synapse.response = {
                "role": "miner",
                "capacity": {
                    "projects": []
                }
            }
            return synapse

        cid_hashs = self.agent_manager.get_miner_agent().keys()
        synapse.response = {
            "role": "miner",
            "capacity": {
                "projects": list(cid_hashs)
            }
        }
        return synapse

    async def invoke_graphql_agent(self, synapse: SyntheticNonStreamSynapse) -> str:
        _, _, graphql_agent = self.agent_manager.get_miner_agent(synapse.project_id)
        # For synthetic challenges, always attempt to answer without domain limitations
        response = await graphql_agent.query_no_stream(synapse.question, is_synthetic=True)
        answer = response.get('messages')[-1].content
        return answer

    async def invoke_miner_agent(self, synapse: SyntheticNonStreamSynapse) -> str:
        agent_graph, _, _ = self.agent_manager.get_miner_agent(synapse.project_id)
        response = await agent_graph.ainvoke(
            {"messages": [{"role": "user", "content": synapse.question}]}
        )
        answer = response.get('messages')[-1].content
        return answer

    async def refresh_agents(self, force_load=False):
        current_dir = Path(__file__).parent
        save_project_dir = current_dir.parent / "projects" / self.role

        model = os.environ.get("MINER_LLM_MODEL", "google/gemini-3-flash-preview")
        self.llm = ChatOpenAI(
            model=model,
            temperature=1
        )

        self.agent_manager = AgentManager(
            save_project_dir=Path(save_project_dir),
            llm_synthetic=self.llm,
        )

        mode = 'load' if force_load else os.getenv("PROJECT_PULL_MODE", "pull")
        # await self.agent_manager.start(mode == "pull", role="miner")

        refresh_agents_interval = int(os.getenv("REFRESH_AGENTS_INTERVAL", 60 * 5))  # seconds

        logger.info(f"[MINER] Using LLM model: {model} for miner self-owned agent")
        logger.info(f"[MINER] Using KEY: {utils.format_openai_key()}")

        silent = False
        while True:
            try:
                self.settings.reread()
                await self.agent_manager.start(mode == "pull", role="miner", silent=silent)
                silent = True
                mode = 'pull'  # after first load, always pull updates
            except Exception as e:
                logger.error(f"refresh_agents error: {e}")
            finally:
                await asyncio.sleep(refresh_agents_interval)

    async def profile_tools_stats(self):
        try:
            while True:
                await asyncio.sleep(60 * 1)
                logger.info(f"[MINER] usage stats: {json.dumps(self.project_usage_metrics.stats())}")
        except KeyboardInterrupt:
            logger.info("[Miner] Profile tools stats interrupted by user")
            raise  # Re-raise to allow graceful shutdown
        except Exception as e:
            logger.error(f"[Miner] Profile tools stats error: {e}")
            raise


if __name__ == "__main__":
    try:
        miner = Miner()
        asyncio.run(miner.start())

        # Keep the miner running
        while True:
            time.sleep(60 * 2)
    except KeyboardInterrupt:
        logger.info("[Miner] Received interrupt signal, shutting down gracefully...")
        # Additional cleanup can be added here if needed
        logger.info("[Miner] Shutdown complete")
    except Exception as e:
        logger.error(f"[Miner] Unexpected error: {e}")
        raise


