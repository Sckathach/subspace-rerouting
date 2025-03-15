import json
import os
import time
from typing import List

from filelock import FileLock

from ssr.evaluation import JudgeScore, call_judge


def rcall_judge(
    prompt: str, answer: str, timelapse: int = 60, n_queries: int = 15
) -> JudgeScore:
    return rate_limited_call(
        prompt,
        answer,
        api_call_func=call_judge,
        timelapse=timelapse,
        n_queries=n_queries,
    )


def load_queue(queue_file: str):
    if os.path.exists(queue_file):
        try:
            with open(queue_file, "r") as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
        except Exception as e:
            print("Queue load error:", e)
    return []


def save_queue(queue, queue_file):
    with open(queue_file, "w") as f:
        json.dump(queue, f)


def wait_for_slot(
    lock_file: str, queue_file: str, timelapse: int = 60, n_queries: int = 15
):
    while True:
        with FileLock(lock_file):
            now = time.time()
            queue = load_queue(queue_file)

            queue = [t for t in queue if now - t < timelapse]
            if len(queue) < n_queries:
                queue.append(now)
                save_queue(queue, queue_file=queue_file)
                return now

            sleep_for = timelapse - (now - queue[0])
        print(f"Rate limit hit. Sleeping for {sleep_for:.2f} sec.")
        time.sleep(sleep_for)


def rate_limited_call(
    *args,
    api_call_func,
    queue_file: str = "api_queue.json",
    timelapse: int = 60,
    n_queries: int = 15,
    n_retries: int = 10,
) -> JudgeScore:
    exceptions: List[Exception] = []
    for _ in range(n_retries):
        lock_file = queue_file + ".lock"
        _ = wait_for_slot(
            lock_file=lock_file,
            queue_file=queue_file,
            n_queries=n_queries,
            timelapse=timelapse,
        )
        try:
            result = api_call_func(*args)
            return result
        except Exception as e:
            exceptions.append(e)
            time.sleep(timelapse)

    raise ValueError(f"Tried {n_retries} times but failed. Errors: ", exceptions)
