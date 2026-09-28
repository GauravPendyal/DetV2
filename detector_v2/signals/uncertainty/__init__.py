"""Model-uncertainty signal (token logprob / entropy) — Stage 4 (not yet implemented).

Only active when the generator's logprobs/internals are actually passed in.
When unavailable it returns ``available=False`` cleanly and the detector
continues on its other signals — HalluciGuard's Base LLM is a separate service,
so this signal is expected to be absent most of the time.
"""
