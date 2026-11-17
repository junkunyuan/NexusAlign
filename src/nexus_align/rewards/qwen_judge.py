"""
Qwen judge reward: the expected holistic score of an image for its prompt, in [0, 1].

A vision-language model reads a judge prompt that rates instruction following,
plausibility, and aesthetics, and asks for {"score": <two decimals>}. The reward does
not generate the answer:

1. It forces the answer up to `{"score": 0.` and takes the expected tenths digit under
   the distribution of the next token over the ten digits.
2. It appends the most likely tenths digit and takes the expected hundredths digit.
3. The score is 0.1 * E[tenths] + 0.01 * E[hundredths].

The expected digits give a continuous score that repeats well, where the greedy answer
gives a few round values only.

The judge of the TwinTilt code is Qwen3.8-27B, which needs 56 GB in bf16, so one copy
per training process does not fit next to the policy. It therefore runs in the Qwen
judge server, one copy per node across its GPUs. See
nexus_align/rewards/servers/qwen_judge_server.py.

References:
    - TwinTilt:
        - Code: https://github.com/shenytzzz/twintilt (private)
            - src/nexus_align/rewards/qwen_judge.py: the judge prompt, the forced
              answer prefix, and the expected digits
    - Qwen3.8:
        - Weights: https://huggingface.co/Qwen/Qwen3.8-27B
"""

import torch

from nexus_align.rewards.remote import RemoteReward

JUDGE_PROMPT = """\
You are an expert evaluator of AI-generated images. You will be given a text
prompt and an image generated from it. Evaluate the image objectively and
assign a single holistic score between 0 and 1.

Consider the following aspects together:

1. Instruction following: Does the image accurately reflect all elements of
   the prompt (objects, attributes, counts, actions, spatial relations, text,
   style, and any knowledge or reasoning the prompt implies)?
2. Plausibility: Is the image structurally and physically coherent within its
   style (anatomy, proportions, perspective, lighting, shadows, and object
   interactions)?
3. Aesthetics: Is the image visually well-crafted (composition, color harmony,
   lighting, and clarity)?

Assign a score between 0 and 1 that reflects the overall quality of the image,
where 0 indicates the lowest possible quality and 1 indicates the highest.

Respond only with valid JSON in the following format:
{"score": <a number between 0 and 1, with two decimals>}"""
ANSWER_PREFIX = '{"score": 0.'  # the forced start of the answer


def expected_digits(logits: torch.Tensor, digit_ids: torch.Tensor) -> tuple:
    """
    Get the expected digit and the most likely digit of each row of next-token logits.

    The softmax covers only the ten digit tokens, so it renormalizes over them.
    """
    probs = logits[:, digit_ids].float().softmax(dim=-1)
    digits = torch.arange(10, device=probs.device, dtype=probs.dtype)
    return probs @ digits, probs.argmax(dim=-1)


class QwenJudge(RemoteReward):
    """Client of the Qwen judge server. It sends the images as raw uint8 arrays."""

    metrics = ("qwen_judge",)  # headline metrics to report
