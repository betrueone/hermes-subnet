import asyncio
from collections import deque
import os
from typing import Any, List, Tuple
import time
from collections import defaultdict
import threading

from loguru import logger

from typing import TYPE_CHECKING

from common.protocol import OrganicNonStreamSynapse
if TYPE_CHECKING:
    from hermes.validator.challenge_manager import ChallengeManager

class BucketCounter:
    def __init__(self, window_hours=3):
        self.bucket_seconds = 3600 # 1 hour per bucket
        self.window_buckets = window_hours
        self.buckets = defaultdict(int)  # {bucket_id: count}
        self._lock = threading.Lock()

    def tick(self) -> int:
        now = int(time.time())
        bucket_id = now // self.bucket_seconds
        with self._lock:
            self.buckets[bucket_id] += 1
            return self.buckets[bucket_id]

    def count(self):
        now = int(time.time())
        current_bucket = now // self.bucket_seconds
        total = 0
        with self._lock:
            # calculate total in the last `window_buckets` buckets
            for i in range(self.window_buckets):
                total += self.buckets.get(current_bucket - i, 0)
        return total

    def cleanup(self):
        # Periodically clean up expired buckets to save memory
        now = int(time.time())
        min_bucket = (now // self.bucket_seconds) - self.window_buckets
        with self._lock:
            self.buckets = {k: v for k, v in self.buckets.items() if k >= min_bucket}


class WorkloadManager:
    uid_organic_response_history: dict[int, deque[OrganicNonStreamSynapse]]
    uid_organic_workload_counter: dict[int, BucketCounter]
    challenge_manager: "ChallengeManager"
    organic_score_queue: list

    uid_sample_scores: dict[int, deque[float]]
    organic_task_compute_interval: int  # seconds
    organic_task_concurrency: int
    organic_task_sample_rate: int
    organic_workload_counter_full_purge_interval: int
    last_full_purge_time: float = time.time()

    def __init__(self, challenge_manager: "ChallengeManager", organic_score_queue: list):
        self.challenge_manager = challenge_manager
        self.organic_score_queue = organic_score_queue

        self.uid_sample_scores = {}
        self.uid_organic_workload_counter = defaultdict(BucketCounter)

        self._purge_lock = asyncio.Lock()

        self.organic_task_compute_interval = int(os.getenv("WORKLOAD_ORGANIC_TASK_COMPUTE_INTERVAL", 30))
        self.organic_task_concurrency = int(os.getenv("WORKLOAD_ORGANIC_TASK_CONCURRENCY", 5))
        self.organic_task_sample_rate = int(os.getenv("WORKLOAD_ORGANIC_TASK_SAMPLE_RATE", 5))
        self.organic_workload_counter_full_purge_interval = int(os.getenv("WORKLOAD_ORGANIC_WORKLOAD_COUNTER_FULL_PURGE_INTERVAL", 3600))

    async def collect(self, uid: int, response: OrganicNonStreamSynapse = None):
         async with self._purge_lock:
            cur = self.uid_organic_workload_counter[uid].tick()
            return cur

    async def purge(self, uids: list[int]):
        for uid in uids:
            if uid in self.uid_organic_workload_counter:
                self.uid_organic_workload_counter[uid].cleanup()

        now = time.time()
        if now - self.last_full_purge_time > self.organic_workload_counter_full_purge_interval:
            async with self._purge_lock:
                to_delete = []
                for uid, counter in list(self.uid_organic_workload_counter.items()):
                    counter.cleanup()
                    if counter.count() == 0:
                        to_delete.append(uid)
                for uid in to_delete:
                    del self.uid_organic_workload_counter[uid]
                self.last_full_purge_time = now
    
    async def compute_workload_score(self, uids: list[int], challenge_id: str = "") -> List[float]:
        await self.purge(uids)

        workload_counts = [self.uid_organic_workload_counter[uid].count() for uid in uids]
        min_workload = min(workload_counts) if workload_counts else 0
        max_workload = max(workload_counts) if workload_counts else 1

        log_quality_scores = []

        scores = [0.0] * len(uids)
        for idx, uid in enumerate(uids):
            quantity = workload_counts[idx]
            uid_quality_scores = self.uid_sample_scores.get(uid, [])
            log_quality_scores.append(list(uid_quality_scores))

            # quality score（EMA）
            if not uid_quality_scores:
                quality_ema = 0.0
            else:
                alpha = 0.7
                quality_ema = None
                for score in uid_quality_scores:
                    if quality_ema is None:
                        quality_ema = score
                    else:
                        quality_ema = alpha * score + (1 - alpha) * quality_ema

            # normalized workload score
            if max_workload == min_workload:
                normalized_workload = 0 if min_workload == 0 else 0.5
            else:
                normalized_workload = (quantity - min_workload) / (max_workload - min_workload)

            total_score = 0.5 * quality_ema + 0.5 * normalized_workload
            scores[idx] = total_score

        logger.info(f"[WorkloadManager] - {challenge_id} workload_counts: {workload_counts}, quality_scores: {log_quality_scores}, compute_workload_score: {scores}")
        return scores

    async def compute_organic_task(self):
        while True:
            await asyncio.sleep(self.organic_task_compute_interval)
            try:
                for i in range(self.organic_task_concurrency):
                    logger.debug(f"[WorkloadManager] Round {i+1}/{self.organic_task_concurrency} of computing organic workload scores")
                    
                    if self.organic_score_queue:
                        miner_uid, resp_dict = self.organic_score_queue.pop(0)
                        response = OrganicNonStreamSynapse(**resp_dict)

                        miner_uid_work_load = await self.collect(miner_uid)
                        if miner_uid_work_load % self.organic_task_sample_rate != 0:
                            logger.debug(f"[WorkloadManager] Skipping organic task computation for miner: {miner_uid} at count {miner_uid_work_load}")
                            continue

                        q = response.completion.messages[-1].content
                        logger.info(f"[WorkloadManager] compute organic task({response.id}) for miner: {miner_uid}, response: {response}. question: {q}")

                        success, ground_truth, ground_cost = await self.challenge_manager.generate_ground_truth(response.project_id, q)
                        if not success:
                            logger.warning(f"[WorkloadManager] Failed to generate ground truth for task({response.id}). {ground_truth}")
                            continue
            
                        logger.info(f"[WorkloadManager] Generated task({response.id}) ground truth: {ground_truth}, cost: {ground_cost}, miner.response: {response.response}")
                        zip_scores, _, _ = await self.challenge_manager.scorer_manager.compute_challenge_score(
                            ground_truth, 
                            ground_cost, 
                            [response],
                            challenge_id=response.id
                        )

                        if miner_uid not in self.uid_sample_scores:
                            self.uid_sample_scores[miner_uid] = deque(maxlen=20)

                        self.uid_sample_scores[miner_uid].append(zip_scores[0])
                        logger.info(f"[WorkloadManager] Updated organic workload score for uid {miner_uid},{zip_scores[0]}, {self.uid_sample_scores}")

                    await asyncio.sleep(1)

            except Exception as e:
                logger.error(f"[WorkloadManager] Error computing organic workload scores: {e}")