"""Make every HuggingFace Hub HTTP call retry on rate-limit / transient errors.

Training launches many accelerate processes that each pull the base model, text
encoder, and the train/test datasets from the Hub in parallel; on a cold cache
that bursts past the rate limit -> HTTP 429 (MaxRetryError). Registering a
retrying requests session via huggingface_hub.configure_http_backend makes
*every* hub call (datasets + models + metadata) back off and retry transparently,
honouring the server's Retry-After header.
"""


def _retrying_session():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    kw = dict(
        total=8,
        backoff_factor=2.0,                       # 2,4,8,... seconds
        status_forcelist=(429, 500, 502, 503, 504),
        allowed_methods=None,                     # retry GET/HEAD/POST alike
        respect_retry_after_header=True,
    )
    try:
        retry = Retry(backoff_jitter=2.0, **kw)   # urllib3 >= 2: de-sync parallel ranks
    except TypeError:
        retry = Retry(**kw)
    adapter = HTTPAdapter(max_retries=retry)
    s = requests.Session()
    s.mount("https://", adapter)
    s.mount("http://", adapter)
    return s


def enable_hf_retries():
    """Install the retrying HTTP backend for huggingface_hub. Safe to call once at
    process startup, before any download. No-op (returns False) if unavailable."""
    try:
        from huggingface_hub import configure_http_backend
        configure_http_backend(backend_factory=_retrying_session)
        return True
    except Exception:
        return False
