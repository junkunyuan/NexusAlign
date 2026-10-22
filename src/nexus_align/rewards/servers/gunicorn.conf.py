"""Gunicorn config of the reward servers: one worker per GPU of this node, on localhost.

Follows reward-server's gunicorn.conf.py; the port and worker count come from the
REWARD_SERVER_PORT and REWARD_SERVER_WORKERS environment variables.
"""

import os

NUM_DEVICES = int(os.environ.get("REWARD_SERVER_WORKERS", 8))
USED_DEVICES = set()


def pre_fork(server, worker):
    # Runs on the server.
    global USED_DEVICES
    worker.device_id = next(i for i in range(NUM_DEVICES) if i not in USED_DEVICES)
    USED_DEVICES.add(worker.device_id)


def post_fork(server, worker):
    # Runs on the worker.
    os.environ["CUDA_VISIBLE_DEVICES"] = str(worker.device_id)


def child_exit(server, worker):
    # Runs on the server.
    global USED_DEVICES
    USED_DEVICES.remove(worker.device_id)


bind = f"127.0.0.1:{os.environ['REWARD_SERVER_PORT']}"
workers = NUM_DEVICES
worker_class = "sync"
timeout = 120
