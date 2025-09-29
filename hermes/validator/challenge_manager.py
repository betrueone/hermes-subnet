import asyncio
import os
from pathlib import Path
import time
import traceback
from typing import Tuple
from uuid import uuid4
import bittensor as bt
from langchain_openai import ChatOpenAI
from loguru import logger
from multiprocessing.synchronize import Event
import numpy as np
import torch
from common.agent_manager import AgentManager
from common.errors import ErrorCode
from common.protocol import SyntheticNonStreamSynapse
from common.settings import Settings
from common.table_formatter import table_formatter
import common.utils as utils
from hermes.validator.question_generator import question_generator
from hermes.validator.scorer_manager import ScorerManager
from hermes.validator.workload_manager import WorkloadManager


class ChallengeManager:
    settings: Settings
    uid: int
    round_id: int
    challenge_interval: int
    dendrite: bt.Dendrite
    llm_synthetic: ChatOpenAI
    llm_score: ChatOpenAI
    agent_manager: AgentManager
    scorer_manager: ScorerManager
    workload_manager: WorkloadManager
    synthetic_score: list
    event_stop: Event
    scores: torch.Tensor

    def __init__(
        self, 
        settings: Settings, 
        save_project_dir: str | Path, 
        uid: int, 
        dendrite: bt.Dendrite,
        organic_score_queue: list,
        synthetic_score: list,
        synthetic_model_name: str | None = None,
        score_model_name: str | None = None,
        event_stop: Event = None
    ):
        self.settings = settings

        # Configure synthetic challenge loop interval (default: 10 minutes)
        self.challenge_interval = int(os.getenv("CHALLENGE_INTERVAL", 600))  # seconds
        self.forward_miner_timeout = int(os.getenv("FORWARD_MINER_TIMEOUT", 60 * 3))  # seconds
        logger.info(f"[ChallengeManager] Synthetic challenge interval set to {self.challenge_interval} seconds")

        self.uid = uid
        self.round_id = 1
        self.dendrite = dendrite

        synthetic_model_name = synthetic_model_name or os.getenv("LLM_MODEL", "gpt-5")
        self.llm_synthetic = ChatOpenAI(
            model=synthetic_model_name,
            temperature=1
        )

        score_model_name = score_model_name or os.getenv("SCORE_LLM_MODEL", "o3")
        self.llm_score = ChatOpenAI(
            model=score_model_name,
            temperature=1
        )

        self.agent_manager = AgentManager(
            save_project_dir=Path(save_project_dir),
            llm_synthetic=self.llm_synthetic,
        )

        self.scorer_manager = ScorerManager(
            llm_score=self.llm_score,
            score_state_path=Path(self.settings.base_dir) / ".data" / "score_state.pt"
        )

        self.workload_manager = WorkloadManager(
            challenge_manager=self,
            organic_score_queue=organic_score_queue,
            work_state_path=Path(self.settings.base_dir) / ".data" / "workload_state.pt"
        )

        self.synthetic_score = synthetic_score
        self.event_stop = event_stop

        self._last_set_weight_time = time.time()
        # self.scores = torch.zeros_like(torch.tensor(self.settings.metagraph.S), dtype=torch.float32)
        # self.device = 'cpu'
        self.set_weight_interval = int(os.getenv("SET_WEIGHT_INTERVAL", 60 * 30))  # seconds
        logger.info(f"[ChallengeManager] Set weight interval set to {self.set_weight_interval} seconds")

        logger.info(f"[ChallengeManager] Using LLM model: {synthetic_model_name} for synthetic challenge")
        logger.info(f"[ChallengeManager] Using LLM model: {score_model_name} for scoring")

    async def start(self):
        mode = os.getenv("PROJECT_PULL_MODE", "pull")

        # pull projects & init agents
        await self.agent_manager.start(mode == "pull", role="validator")

        self.task = [
            asyncio.create_task(self.workload_manager.compute_organic_task()),
            asyncio.create_task(self.set_weight()),
            asyncio.create_task(self.challenge_loop())
        ]
        await asyncio.gather(*self.task)

    async def challenge_loop(self):
        while not self.event_stop.is_set():
            await asyncio.sleep(self.challenge_interval)

            projects = self.agent_manager.get_projects()
            if not projects:
                logger.warning("[ChallengeManager] No projects found, skipping this round.")
                await asyncio.sleep(self.challenge_interval)
                continue
            
            miner_uids, miner_hotkeys = self.settings.miners()
            uids = []
            hotkeys = []
            for idx, u in enumerate(miner_uids):
                if u != self.uid:
                    uids.append(u)
                    hotkeys.append(miner_hotkeys[idx])

            if not uids:
                logger.warning("[ChallengeManager] No available miners for challenge, skipping this round.")
                await asyncio.sleep(self.challenge_interval)
                continue

            project_score_matrix = []

            for cid, project_config in projects.items():
                challenge_id = str(uuid4())
                
                # generate challenge
                question = await question_generator.generate_question(cid, project_config.schema_content, self.llm_synthetic)
                if not question:
                    continue

                # generate ground truth
                success, ground_truth, ground_cost = await self.generate_ground_truth(cid, question)

                # Create challenge table
                table_formatter.create_synthetic_challenge_table(
                    round_id=self.round_id,
                    challenge_id=challenge_id,
                    cid=cid,
                    question=question,
                    success=success,
                    ground_truth=ground_truth,
                    ground_cost=ground_cost
                )
                if not success:
                    logger.error(f"[ChallengeManager] - {challenge_id} Failed to generate ground truth. {ground_truth}")
                    continue

                # query all miner
                logger.info(f"[ChallengeManager] - {challenge_id} query miners: {uids}")
                responses = await asyncio.gather(
                    *(self.query_miner(
                        uid=uid,
                        cid=cid,
                        challenge_id=challenge_id,
                        question=question
                    ) for uid in uids)
                )

                # score result
                zip_scores, ground_truth_scores, elapse_weights = await self.scorer_manager.compute_challenge_score(
                    ground_truth, 
                    ground_cost, 
                    responses,
                    challenge_id=challenge_id
                )
                project_score_matrix.append(zip_scores)

                table_formatter.create_synthetic_miners_response_table(
                    round_id=self.round_id,
                    challenge_id=challenge_id,
                    uids=uids,
                    responses=responses,
                    ground_truth_scores=ground_truth_scores,
                    elapse_weights=elapse_weights,
                    zip_scores=zip_scores,
                    cid=cid
                )

            if not project_score_matrix:
                logger.warning("[ChallengeManager] No valid project score matrix, skipping this round.")
                continue
    
            workload_score, workload_counts, log_quality_scores = await self.workload_manager.compute_workload_score(uids, hotkeys, challenge_id=challenge_id)
            new_ema_scores = self.scorer_manager.update_scores(
                uids, 
                hotkeys,
                project_score_matrix, 
                workload_score, 
                challenge_id=challenge_id
            )
            self.synthetic_score[0] = self.scorer_manager.get_last_synthetic_scores()

            table_formatter.create_synthetic_final_ranking_table(
                round_id=self.round_id,
                challenge_id=challenge_id,
                uids=uids,
                hotkeys=hotkeys,
                workload_counts=workload_counts,
                quality_scores=log_quality_scores,
                workload_score=workload_score,
                new_ema_scores=new_ema_scores
            )
            self.round_id += 1

    async def generate_ground_truth(self, cid: str, question: str) -> Tuple[bool, str | None, int]:
        start_time = time.perf_counter()
        success = False
        result = None
        try:
            agent = self.agent_manager.get_graphql_agent(cid)
            if not agent:
                result = f"No server agent found for cid: {cid}"
            else:
                response = await agent.query_no_stream(question)
                result = response.get('messages', [])[-1].content
                success = True
            
                if not result:
                    error = utils.try_get_invalid_tool_messages(response.get('messages', []))
                    if error:
                        logger.error(f"[ChallengeManager] - {cid} Failed to generate ground truth. {error}")
                        success = False
                        result = error
        except Exception as e:
            result = str(e)
            logger.error(f"[ChallengeManager] generate_ground_truth error for cid: {cid} {e}\n{traceback.format_exc()}")

        finally:
            return [success, result, utils.fix_float(time.perf_counter() - start_time)]

    async def query_miner(
        self, 
        uid: int, 
        cid: str, 
        challenge_id: str, 
        question: str, 
    ):
        synapse = SyntheticNonStreamSynapse(id=challenge_id, project_id=cid, question=question)
        start_time = time.perf_counter()
        
        try:
            r: SyntheticNonStreamSynapse = await self.dendrite.forward(
                axons=self.settings.metagraph.axons[uid],
                synapse=synapse,
                deserialize=False,
                timeout=self.forward_miner_timeout,
            )
            logger.debug(f"🔍 [ChallengeManager] - {challenge_id} MINER RESPONSE [UID: {uid}] - ✅ is_success: {r.is_success} - {r.dendrite.status_code}")
        except Exception as e:
            logger.error(f"🔍 [ChallengeManager] - {challenge_id} MINER RESPONSE [UID: {uid}] - ❌ Failed to query: {e}\n{traceback.format_exc()}")
            r.status_code = ErrorCode.FORWARD_SYNTHETIC_FAILED.value
            r.error = str(e)
        finally:
            r.elapsed_time = utils.fix_float(time.perf_counter() - start_time)
            return r

    async def set_weight(self):
        while True:
            await asyncio.sleep(10)
            if time.time() - self._last_set_weight_time > self.set_weight_interval:
                try:
                    scores_dict = self.scorer_manager.get_last_overall_scores()
                    uids = list(scores_dict.keys())
                    scores = list(scores_dict.values())
                    if not uids:
                        continue
                    scores = [s[0] for s in scores]
                    self._set_weights(uids, scores)

                    self._last_set_weight_time = time.time()
                except Exception as e:
                    logger.error(f"[ChallengeManager] Failed to set_weight: {e}")

    def _set_weights(self, uids: list[int], scores: list[float]):
        logger.info(f"[ChallengeManager] set_weights for uids: {uids}, scores: {scores}")
        scores_np = np.array(scores, dtype=np.float32)

        if np.all(scores_np == 0):
            logger.warning("[ChallengeManager] All scores are zero, skipping set weight.")
            return
        (
            processed_weight_uids,
            processed_weights,
        ) = bt.utils.weight_utils.process_weights_for_netuid(
                uids = np.array(uids, dtype=np.int64),
                # weights = raw_weights.detach().cpu().numpy().astype(np.float32),
                weights = scores_np,
                netuid=self.settings.netuid,
                subtensor=self.settings.subtensor,
                metagraph=self.settings.metagraph,
        )
        logger.info(f"processed_weight_uids: {processed_weight_uids}")
        logger.info(f"processed_weights: {processed_weights}")

        [suc, msg] = self.settings.subtensor.set_weights(
            wallet=self.settings.wallet,
            netuid=self.settings.netuid,
            uids=processed_weight_uids,
            weights=processed_weights,
            wait_for_finalization=False,
            version_key=10010,
        )
        logger.info(f"processed_weights result: {suc, msg}")

