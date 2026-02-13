import asyncio
import json
import os
from collections import defaultdict
from pathlib import Path
import time
from langchain_openai import ChatOpenAI
from loguru import logger
from loguru._logger import Logger
from agent.stats import Phase
from common.prompt_template import get_miner_self_tool_prompt
from langchain_core.messages import HumanMessage, SystemMessage
from common.agent_manager import AgentManager
from common.enums import ErrorCode
import common.utils as utils

class SyntheticSynapse():
    id: str | None = None
    cid_hash: str | None = None
    block_height: int | None = 0
    question: str | None = None
    response: str | None = None
    graphql_agent_inner_tool_calls: list[str] | None = None

    def __init__(self, id: str, cid_hash: str, block_height: int, question: str):
        self.id = id
        self.cid_hash = cid_hash
        self.block_height = block_height
        self.question = question

class TestMiner():

    def __init__(self):
        self.agent_manager = None

    async def start(self):
        await self.refresh_agents()
    
    async def _handle_task(
            self,
            task: SyntheticSynapse,
            log: Logger,
    ) -> SyntheticSynapse:
        question = task.question
        cid_hash = task.cid_hash
        graph, graphql_agent = self.agent_manager.get_miner_agent(cid_hash)

        tag = "Synthetic"
        phase = Phase.MINER_SYNTHETIC
        messages = [
            SystemMessage(content=get_miner_self_tool_prompt(block_height=task.block_height, node_type=graphql_agent.config.node_type if graphql_agent else "unknown")),
            HumanMessage(content=question)
        ]

        answer = None
        usage_info = {}
        tool_hit = []
        graphql_agent_inner_tool_calls = []
        response = None
        error = None
        status_code = ErrorCode.SUCCESS
        elapsed = 0.0

        try:
            before = time.perf_counter()
            if not graph:
                log.warning(f"[{tag}] - {task.id} No agent found for project {cid_hash}")
                error = f"No agent found for project {cid_hash}"
                status_code = ErrorCode.AGENT_NOT_FOUND
                tool_hit = []
                graphql_agent_inner_tool_calls = []
                task.step_timings = []
            else:
                r, step_timings = await utils.run_graph_with_step_timings(
                    graph, {"messages": messages, "block_height": task.block_height}
                )
                task.step_timings = step_timings
                if r is None:
                    error = "Graph stream did not return final state"
                    status_code = ErrorCode.INTERNAL_SERVER_ERROR
                    tool_hit = []
                    graphql_agent_inner_tool_calls = []
                else:
                    (
                        answer,
                        tool_hit,
                        graphql_agent_inner_tool_calls,
                        response,
                        error,
                        status_code
                    ) = self.get_answer(phase, task, r)
            elapsed = time.perf_counter() - before
        except Exception as e:
            log.error(f"handle task error {task.id} - {question}. {e}\n")
            error = str(e)
            status_code = ErrorCode.INTERNAL_SERVER_ERROR
            tool_hit = []
            graphql_agent_inner_tool_calls = []
            elapsed = time.perf_counter() - before
            task.step_timings = getattr(task, "step_timings", None) or []

        task.response = response
        task.elapsed_time = elapsed
        task.error = error
        task.status_code = status_code.value
        task.tool_hit = tool_hit
        task.graphql_agent_inner_tool_calls = graphql_agent_inner_tool_calls
        task.miner_model_name = self.llm.model_name
        task.graphql_agent_model_name = graphql_agent.llm.model_name

        return task

    def get_answer(
        self,
        phase: Phase,
        task: SyntheticSynapse,
        r: dict
    ) -> tuple[str | None, list, list[str], str | None, str | None, ErrorCode]:
        # logger.info(f"[{tag}] - {task.id} Agent response: {r}")

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
        
        return answer, tool_hit, graphql_agent_inner_tool_calls, response, error, status_code
        
    async def forward_synthetic_non_stream(self, task: SyntheticSynapse) -> SyntheticSynapse:
        log = logger.bind(source="My hotkey")
        await self._handle_task(task, log)
        return task

    # async def forward_organic_stream(self, synapse: OrganicStreamSynapse) -> StreamingSynapse.BTStreamingResponse:
    #     print(f"forward_organic_stream: {synapse}")
    #     from starlette.types import Send
    #     log = logger.bind(source=synapse.dendrite.hotkey)
    #     log.info(f"\n🤖 [Miner] Received organic stream: {synapse.id}")

    #     messages = synapse.to_messages()
    #     graph, graphql_agent = self.agent_manager.get_miner_agent(synapse.cid_hash)

    #     if not graph:
    #         error_msg = f"Error: No agent found for project {synapse.cid_hash}"
    #         log.warning(f"[Miner] - {synapse.id} {error_msg}")
    #         async def error_streamer(send: Send):
    #             error_line = json.dumps({
    #                 "type": "data",
    #                 "data": error_msg
    #             }) + "\n"
    #             await send({
    #                 "type": "http.response.body",
    #                 "body": error_line.encode('utf-8'),
    #                 "more_body": False
    #             })
            
    #         return synapse.create_streaming_response(error_streamer)
        
    #     fill_miner_self_tool_prompt(messages, block_height=synapse.block_height, node_type=graphql_agent.config.node_type if graphql_agent else "unknown")

    #     async def token_streamer(send: Send):
    #         r = None
    #         tag = "Organic-S"
    #         phase = Phase.MINER_ORGANIC_STREAM
    #         before = time.perf_counter()
    #         async for event in graph.astream(
    #             {
    #                 "messages": messages,
    #                 "block_height": synapse.block_height
    #             },
    #             version="v2"
    #         ):
    #             for key, value in event.items():
    #                 if key == "final":
    #                     r = value
    #                     message = value.get("messages", [])[-1].content

    #                     if r.get('error', None) is not None:
    #                         message = r.get('error')
    #                     chunk_size = int(os.getenv("STREAM_CHUNK_SIZE", 256))
    #                     chunk_size = max(1, chunk_size)
    #                     chunk_delay = float(os.getenv("STREAM_CHUNK_DELAY", "0"))
    #                     idx = 0
    #                     while idx < len(message):
    #                         chunk = message[idx:idx + chunk_size]
    #                         # Send data chunks in JSONL format
    #                         data_line = json.dumps({
    #                             "type": "data",
    #                             "data": chunk
    #                         }) + "\n"
    #                         await send({
    #                             "type": "http.response.body",
    #                             "body": data_line.encode('utf-8'),
    #                             "more_body": True
    #                         })
    #                         if chunk_delay > 0:
    #                             await asyncio.sleep(chunk_delay)
    #                         idx += chunk_size
                
    #         elapsed = utils.fix_float(time.perf_counter() - before)
    #         synapse.elapsed_time = elapsed
    #         (
    #             answer,
    #             usage_info,
    #             tool_hit,
    #             graphql_agent_inner_tool_calls,
    #             response,
    #             error,
    #             status_code
    #         ) = self.get_answer(phase, synapse, r)

    #         # Send metadata in JSONL format
    #         metadata_line = json.dumps({
    #             "type": "meta",
    #             "data": {
    #                 "miner_model_name": self.llm.model_name,
    #                 "graphql_agent_model_name": graphql_agent.llm.model_name,
    #                 "elapsed": elapsed,
    #                 "status_code": status_code.value,
    #                 "error": error,
    #                 "graphql_agent_inner_tool_calls": graphql_agent_inner_tool_calls,
    #                 "usage_info": usage_info
    #             }
    #         }) + "\n"
    #         await send({
    #             "type": "http.response.body",
    #             "body": metadata_line.encode('utf-8'),
    #             "more_body": False
    #         })

    #     return synapse.create_streaming_response(token_streamer)

    # async def forward_organic_non_stream(self, task: OrganicNonStreamSynapse) -> OrganicNonStreamSynapse:
    #     print(f"forward_organic_non_stream: {task}")
    #     log = logger.bind(source=task.dendrite.hotkey)
    #     await self._handle_task(task, log)
    #     return task

    async def invoke_graphql_agent(self, synapse: SyntheticSynapse) -> str:
        _, _, graphql_agent = self.agent_manager.get_miner_agent(synapse.cid_hash)
        # For synthetic challenges, always attempt to answer without domain limitations
        response = await graphql_agent.query_no_stream(synapse.question, is_synthetic=True)
        answer = response.get('messages')[-1].content
        return answer

    async def invoke_miner_agent(self, synapse: SyntheticSynapse) -> str:
        agent_graph, _, _ = self.agent_manager.get_miner_agent(synapse.cid_hash)
        response = await agent_graph.ainvoke(
            {"messages": [{"role": "user", "content": synapse.question}]}
        )
        answer = response.get('messages')[-1].content
        return answer

    async def refresh_agents(self, force_load=False):
        current_dir = Path(__file__).parent
        save_project_dir = current_dir / "projects" / "miner"

        model = os.environ.get("MINER_LLM_MODEL", "gpt-5-mini")
        self.llm = ChatOpenAI(
            model=model,
            temperature=1
        )

        self.agent_manager = AgentManager(
            save_project_dir=Path(save_project_dir),
            llm_synthetic=self.llm,
        )

        await self.agent_manager.start(pull=False, role="miner", silent=True)

        logger.info(f"[MINER] Using LLM model: {model} for miner self-owned agent")
        logger.info(f"[MINER] Using KEY: {utils.format_openai_key()}")



questions = [
    # "What is the total amount of query spending that has been refunded to consumers across all indexing service deployments?",
    # "What is the total amount of tokens currently distributed as commission to the service provider who has the highest annualized percentage yield for delegators in the current era?",
    # "Which indexer currently offers the highest annualized percentage yield for its delegators?",
    # "What are the top 3 network operators ranked by the total amount of rewards they have successfully claimed and withdrawn from the protocol to date?",
    # "What is the total amount of tokens currently being withdrawn across all ongoing undelegation requests in the network?",
    # "What are the top three software version deployments that have the highest daily request limit according to their associated service templates?",
    # "What is the total amount of service fees currently locked in all active service agreements across the network?",
    # "What is the total amount of tokens currently committed to the single largest token swap buy order that is still active in the network?",
    # "What is the total amount of tokens across all pending withdrawal and unstaking requests that have not yet been claimed?",
    # "What are the top three service providers that currently have the highest remaining capacity to accept new token delegations from the community?",
    # "What is the total amount of tokens currently locked as security deposits across all valid service offers created by consumers?",
    # "What is the total amount of tokens currently staked by the single largest contributor to the network's voting system for a specific subnet project?",
    # "What are the top 3 network operators ranked by the amount of commission they have earned during the previous era?",
    # "What is the total amount of tokens currently involved in the single largest penalty amount ever deducted during a service offer withdrawal operation?",
    # "What are the top 3 network operators ranked by the total amount of their own tokens they have currently staked in the network?",
    # "What are the top three service plans that have the highest price set for their indexing services?",
    # "What is the total number of service plan templates that are currently active and available for use within the network?",
    # "What is the total amount of tokens that have been burned or lost due to penalties across all indexer reward distributions?",
    # "What is the total amount of query fees that have been refunded to users across all recorded indexing service transactions?",
    # "What is the total amount of tokens currently being circulated throughout the network, excluding those held in reserved administrative wallets?",
    # "What is the total amount of tokens currently distributed as rewards to the single service provider who has received the highest amount of stake allocations from the network across all eras?",
    # "What is the total amount of tokens currently distributed across all community airdrop allocations?",
    # "What are the top three software version deployments that currently have the highest number of service providers actively indexing them?",
    # "What is the total amount of query fees that have been refunded to consumers across all query service transactions in the network?",
    # "What is the total amount of tokens currently available in the single largest payment channel established between a service consumer and a provider?",
    # "What is the total amount of tokens currently distributed as rewards to the three indexers who have experienced the longest accumulated duration of allocation overflows to date?",
    # "What is the total amount of tokens currently involved in exchange orders that are actively waiting to be traded in the network's built-in swap system?",
    # "What are the top three software version deployments that have the highest total amount of service fees spent by consumers through payment channels?",
    # "What are the top 3 network projects that currently have the highest ratio of consumer boost funds relative to their total stake allocation?",
    # "What are the top three communities or individuals who have successfully claimed the largest cumulative amount of tokens through various distribution rounds?",
    # "What is the total amount of tokens currently staked by all delegators across the three indexers with the highest total stake?",
    # "What is the total amount of tokens currently held in the wallets of the top three largest token holders in the network?",
    # "What is the total amount of tokens currently locked in all payment channels that are still in the open status?",
    # "What is the total number of service providers who are currently operating in a terminated status?",
    # "What are the top 3 network operators ranked by the amount of work they performed across all project versions in the most recent era?",
    # "What is the total number of service requests that can be handled daily across all currently active service plans in the network?",
    # "What is the total number of service providers who are currently proxying their on-chain operations through an active controller address?",
    # "What is the total amount of tokens currently distributed as commission to all network operators across all eras?",
    # "What is the total amount of tokens that have been returned to participants after the cancellation of their specific withdrawal operations?",
    # "What is the total number of distinct indexing service providers that are currently marked as active in the network?",
    # "What are the top 3 project versions ranked by the total volume of daily requests they are permitted to handle based on their associated service plan templates?",
    # "What is the total amount of tokens currently staked by the single most active participant in project governance voting across all subnets?",
    # "What are the top 3 network operators ranked by the longest duration of time they spent in an allocation overflow state during the most recent era?",
    # "What is the total amount of tokens that administrative owners have withdrawn from unclaimed community distribution rounds?",
    # "What is the total number of service providers who currently have an active commission rate set above zero percent?",
    # "What is the total amount of SQT tokens currently held in the wallets of all network participants, excluding those in the zero address?",
    # "What is the total number of service providers who have had at least one dispute case raised against them by a fisherman?",
    # "What is the total number of indexing service deployments that have been specifically classified as being associated with artificial intelligence models?",
    # "What is the total amount of SQT tokens currently being held by the protocol as security deposits within service offers that have already expired?",
    # "What is the total amount of SQT tokens that have been burned across all recorded withdrawal transactions?",
    # "What is the total amount of tokens currently involved in all reported disputes that are still ongoing?",
    # "What are the top 3 network operators ranked by the total amount of self-stake they have committed to the protocol as of the most recent era?",
    # "What is the total amount of SQT tokens that was distributed as rewards to all project versions specifically for serving queries during the most recent reward era?",
    # "What is the total number of indexing project versions that are currently being actively operated by at least one network participant?",
    # "What is the total amount of work performed by all service providers across all data deployments during the most recent era?",
    # "What is the total amount of tokens currently allocated as self-stake by the indexer that has achieved the highest yield for its delegators in the most recent era?",
    # "What are the top 3 network operators ranked by the total amount of rewards they have successfully claimed and withdrawn from the protocol to date?",
    # "What is the total combined amount of all query payments settled through state channels since the network began?",
    # "What is the total amount of tokens that have been slashed from service operators due to accepted dispute cases?",
    # "What is the total amount of SQT tokens that have been transferred in the most recent transaction recorded on the network?",
    # "What are the top 3 network deployments that currently have the highest amount of combined rewards earned across all distribution sources in the most recent era?",
    # "What are the top 3 network operators ranked by the annualized yield they provided to their delegators in the most recent era?",
    # "What are the top 3 network operators that currently have the largest amount of available capacity for new token delegations?",
    # "What is the total amount of SQT tokens that have been burned as penalties for missed labor requirements across all network providers in the most recent era?",
    # "What are the top 3 token holders ranked by their current balance of SQT tokens?",
    # "What is the total amount of tokens currently committed to voting for projects within the governance subnets?",
    # "What is the total amount of tokens that have been returned to consumers as refunds for query service orders across all project versions?",
    # "What are the top 3 network operators that have the highest commission rate currently set for their services?",
    # "What is the total amount of SQT tokens that have been staked as a booster by all consumers to increase the rewards for specific projects?",
    
    # "What is the total amount of tokens currently locked in all active state channels between consumers and indexers?",
    "What are the IDs of the three projects that have earned the highest total amount of rewards for their work on the network?"
]


if __name__ == "__main__":
    import sys
    
    # Allow testing specific question(s) via command line args
    # Usage: python test_miner.py [start_idx] [end_idx]
    # Examples:
    #   python test_miner.py          # Test all questions
    #   python test_miner.py 1        # Test question 1 only
    #   python test_miner.py 1 5      # Test questions 1-5
    
    start_idx = int(sys.argv[1]) - 1 if len(sys.argv) > 1 else 0
    end_idx = int(sys.argv[2]) if len(sys.argv) > 2 else len(questions)
    
    # Clamp indices
    start_idx = max(0, min(start_idx, len(questions) - 1))
    end_idx = max(start_idx + 1, min(end_idx, len(questions)))
    
    questions_to_test = questions[start_idx:end_idx]
    # questions_to_test = questions[-1:]
    question_range = f"{start_idx + 1}-{end_idx}" if end_idx - start_idx > 1 else str(start_idx + 1)
    
    test_miner = TestMiner()
    asyncio.run(test_miner.start())
    block_height = asyncio.run(utils.get_latest_block("https://index-api.onfinality.io/sq/subquery/subquery-mainnet", "subql"))
    
    print(f"\n{'='*80}")
    print(f"Testing Question(s) {question_range} of {len(questions)} total")
    print(f"Block Height: {block_height}")
    print(f"{'='*80}\n")
    
    # Metrics tracking
    results = []
    tool_usage_stats = {}
    
    for local_idx, question in enumerate(questions_to_test, 1):
        task = SyntheticSynapse(
            id="QmZozXZDYWs2sRGCaSUP1F4x2b3hvDYCyWP53NjFRzkqWg_ac958ff1",
            cid_hash="QmZozXZDYWs2sRGCaSUP1F4x2b3hvDYCyWP53NjFRzkqWg_ac958ff1",
            block_height=block_height,
            question=question,
        )
        synapse = asyncio.run(test_miner.forward_synthetic_non_stream(task))

        # Extract tool information
        tool_hit = getattr(synapse, 'tool_hit', [])
        graphql_agent_inner_tool_calls = getattr(synapse, 'graphql_agent_inner_tool_calls', [])
        
        # Format tool names
        selected_tools = []
        tool_names_only = []  # For tracking (without count suffix)
        if tool_hit:
            for name, count in tool_hit:
                selected_tools.append(f"{name} (x{count})")
                tool_names_only.append(name)
                # Track tool usage
                tool_usage_stats[name] = tool_usage_stats.get(name, 0) + count
        if graphql_agent_inner_tool_calls:
            selected_tools.extend(graphql_agent_inner_tool_calls)
            tool_names_only.extend(graphql_agent_inner_tool_calls)
            for tool_name in graphql_agent_inner_tool_calls:
                tool_usage_stats[tool_name] = tool_usage_stats.get(tool_name, 0) + 1
        
        tool_display = ", ".join(selected_tools) if selected_tools else "❌ NO TOOL SELECTED"
        
        # Status indicator
        has_tool = len(selected_tools) > 0
        status = "✅" if has_tool else "❌"
        error = getattr(synapse, 'error', None)
        error_display = f" | ERROR: {error}" if error else ""
        
        elapsed = getattr(synapse, 'elapsed_time', None)
        elapsed_s = f"{elapsed:.2f}s" if isinstance(elapsed, (int, float)) else "N/A"
        step_timings = getattr(synapse, 'step_timings', None) or []
        step_timings_s = utils.format_step_timings(step_timings)

        # Store result for metrics
        results.append({
            'question': question,
            'has_tool': has_tool,
            'tools': tool_names_only,
            'error': error,
            'response': getattr(synapse, 'response', None),
            'elapsed': elapsed,
            'step_timings': step_timings,
        })
        
        print(f"\n{'─'*80}")
        print(f"Question #{start_idx + local_idx}/{len(questions)}")
        print(f"{'─'*80}")
        print(f"Q: {question}")
        print(f"{'─'*80}")
        print(f"{status} Selected Tool(s): {tool_display}{error_display}")
        print(f"⏱️  Elapsed: {elapsed_s}")
        print(f"⏱️  Step timings: {step_timings_s}")
        print(f"{'─'*80}")
        response = getattr(synapse, 'response', 'N/A')
        if response and len(response) > 500:
            print(f"Response: {response[:500]}... (truncated, full length: {len(response)})")
        else:
            print(f"Response: {response}")
        print(f"{'─'*80}\n")
    
    # Print overall metrics
    print(f"\n{'='*80}")
    print("OVERALL METRICS")
    print(f"{'='*80}\n")
    
    total_tested = len(results)
    with_tools = sum(1 for r in results if r['has_tool'])
    without_tools = total_tested - with_tools
    with_errors = sum(1 for r in results if r['error'])
    
    success_rate = (with_tools / total_tested * 100) if total_tested > 0 else 0

    # Elapsed time stats
    elapsed_list = [r['elapsed'] for r in results if isinstance(r.get('elapsed'), (int, float))]
    total_elapsed = sum(elapsed_list)
    avg_elapsed = (total_elapsed / len(elapsed_list)) if elapsed_list else 0

    # Total time by step (across all questions)
    step_totals = defaultdict(float)
    for r in results:
        for s in r.get('step_timings') or []:
            step_totals[s['step']] += s['elapsed']
    step_totals_s = ", ".join(f"{k} {v:.2f}s" for k, v in sorted(step_totals.items(), key=lambda x: -x[1]))
    
    # Calculate breakdown
    no_tool_only = sum(1 for r in results if not r['has_tool'] and not r['error'])
    error_only = sum(1 for r in results if r['error'] and r['has_tool'])
    both_failed = sum(1 for r in results if not r['has_tool'] and r['error'])
    
    print(f"📊 Summary Statistics:")
    print(f"   Total Questions Tested: {total_tested}")
    print(f"   ✅ Questions with Tools Selected: {with_tools} ({with_tools/total_tested*100:.1f}%)")
    print(f"   ❌ Questions without Tools: {without_tools} ({without_tools/total_tested*100:.1f}%)")
    print(f"      └─ No tool only: {no_tool_only}")
    print(f"      └─ Error only: {error_only}")
    print(f"      └─ No tool + Error: {both_failed}")
    print(f"   ⚠️  Questions with Errors: {with_errors}")
    print(f"   📈 Success Rate: {success_rate:.1f}%")
    print(f"   ⏱️  Total Elapsed: {total_elapsed:.2f}s")
    print(f"   ⏱️  Avg per Question: {avg_elapsed:.2f}s")
    if step_totals_s:
        print(f"   ⏱️  Total by Step: {step_totals_s}")
    print()
    
    # All tested questions with status
    print(f"📋 All Tested Questions ({total_tested}):")
    print(f"{'─'*80}")
    for i, r in enumerate(results, start=start_idx + 1):
        status_icon = "✅" if r.get('has_tool', False) else "❌"
        error_icon = " ⚠️" if r.get('error') else ""
        tools_info = f" → {', '.join(r.get('tools', []))}" if r.get('tools') else ""
        el = r.get('elapsed')
        elapsed_info = f" [{el:.2f}s]" if isinstance(el, (int, float)) else ""
        question_text = r.get('question', 'Unknown question')
        print(f"   {status_icon}{error_icon} #{i:2d}: {question_text}{tools_info}{elapsed_info}")
    print()
    
    # Failed questions: no tools found OR errors
    failed_questions = [r for r in results if not r['has_tool']]
    error_questions = [r for r in results if r['error']]
    
    # Combine and deduplicate (a question can have both no tool and error)
    # Use question text as key to avoid duplicates
    all_failed = {}
    for r in failed_questions + error_questions:
        question_text = r['question']
        if question_text not in all_failed:
            all_failed[question_text] = r
        else:
            # Merge error info if present
            if r['error'] and not all_failed[question_text]['error']:
                all_failed[question_text]['error'] = r['error']
    
    if all_failed:
        print(f"❌ FAILED QUESTIONS - No Tools Found or Errors ({len(all_failed)}):")
        print(f"{'='*80}\n")
        
        # Detailed breakdown
        for question_text, r in sorted(all_failed.items()):
            # Find the original index in the results list
            original_idx = next((i for i, res in enumerate(results) if res['question'] == question_text), None)
            question_num = start_idx + original_idx + 1 if original_idx is not None else "?"
            
            print(f"\n   Question #{question_num}:")
            print(f"   {'─'*76}")
            print(f"   Q: {r['question']}")
            print(f"   {'─'*76}")
            
            if not r['has_tool']:
                print(f"   ❌ Status: NO TOOL SELECTED")
            else:
                print(f"   ✅ Status: Tool selected: {', '.join(r['tools'])}")
            el = r.get('elapsed')
            if isinstance(el, (int, float)):
                print(f"   ⏱️  Elapsed: {el:.2f}s")
            st = r.get('step_timings')
            if st:
                print(f"   ⏱️  Step timings: {utils.format_step_timings(st)}")
            if r['error']:
                print(f"   ⚠️  Error: {r['error']}")
            
            if r['response']:
                response_preview = r['response'][:200] + "..." if len(r['response']) > 200 else r['response']
                print(f"   📝 Response: {response_preview}")
            else:
                print(f"   📝 Response: None")
        print(f"\n{'='*80}\n")
    else:
        print(f"✅ All questions successfully found tools with no errors!\n")
    
    # Tool usage statistics
    if tool_usage_stats:
        print(f"🔧 Tool Usage Statistics:")
        print(f"{'─'*80}")
        sorted_tools = sorted(tool_usage_stats.items(), key=lambda x: x[1], reverse=True)
        for tool_name, count in sorted_tools:
            print(f"   {tool_name}: {count} time(s)")
        print()
    
    # Questions with multiple tools
    multi_tool_questions = [r for r in results if len(r['tools']) > 1]
    if multi_tool_questions:
        print(f"🔀 Questions with Multiple Tools ({len(multi_tool_questions)}):")
        print(f"{'─'*80}")
        for r in multi_tool_questions:
            tools_str = ", ".join(r['tools'])
            # Find the original index in the results list
            original_idx = next((i for i, res in enumerate(results) if res['question'] == r['question']), None)
            question_num = start_idx + original_idx + 1 if original_idx is not None else "?"
            print(f"   #{question_num}: {tools_str}")
        print()
    
    print(f"{'='*80}\n")