# Store the version here so:
# 1) we don't load dependencies by storing it in __init__.py
# 2) we can import it in setup.py for the same reason
# 3) we can import it into your module module

__version__ = '4.2.0'


def gitversion():
    try:
        from .git_version import gitversion
    except ImportError:
        return "unknown"
    return gitversion
