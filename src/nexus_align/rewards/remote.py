"""Remote rewards: clients of reward servers that run in their own environments."""

import pickle

import requests
from requests.adapters import HTTPAdapter, Retry


class RemoteReward:
    """
    Client of a reward server on this node, posting pickled requests over HTTP.

    - Reward models with dependencies that conflict with training run as servers;
      see nexus_align/rewards/servers.
    - Server errors are retried a bounded number of times, so a dead server fails the
      job instead of hanging it.
    """

    def __init__(self, cfg_reward) -> None:
        self.name = cfg_reward.name
        self.url = cfg_reward.url
        self.timeout = cfg_reward.get("timeout", 120)
        self.session = requests.Session()
        self.session.trust_env = False  # the server is local, so skip any HTTP proxy
        retries = Retry(
            total=cfg_reward.get("max_retries", 10),
            backoff_factor=1,
            status_forcelist=[500],
            allowed_methods=False,
        )
        self.session.mount("http://", HTTPAdapter(max_retries=retries))

    def post(self, data: dict, path: str = "") -> dict:
        """Send one request to the url plus path and return the unpickled response."""
        data = pickle.dumps(data)
        response = self.session.post(self.url + path, data=data, timeout=self.timeout)
        response.raise_for_status()
        return pickle.loads(response.content)
