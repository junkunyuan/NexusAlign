"""
Shared route of the reward servers: pickled requests in, pickled responses out.

The servers import this file as a sibling module, because gunicorn runs them with
this directory on its path.
"""

import pickle
import traceback

from flask import request


def pickle_route(app, rule: str, fn) -> None:
    """Serve fn(data) at a POST rule. An error returns its traceback with status 500."""

    def inference():
        try:
            data = pickle.loads(request.get_data())
            return pickle.dumps(fn(data)), 200
        except Exception:
            response = traceback.format_exc()
            print(response)
            return response.encode("utf-8"), 500

    app.add_url_rule(rule, endpoint=rule, view_func=inference, methods=["POST"])
