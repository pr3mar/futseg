"""The single cuda/cpu decision point; backends receive a resolved string."""


def resolve_device(preference: str = "auto") -> str:
    """Resolve a device preference to a concrete device string.

    This is the only place in futseg that asks torch whether CUDA is available.
    Backends take the resolved string, which keeps the policy in one testable
    place and lets tests pass a string rather than mock torch internals.
    """
    if preference != "auto":
        return preference

    import torch  # lazy: keeps `futseg --help` and argument errors responsive

    return "cuda" if torch.cuda.is_available() else "cpu"
