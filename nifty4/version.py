# Store the version here so:
# 1) we don't load dependencies by storing it in __init__.py
# 2) we can import it in setup.py for the same reason
# 3) we can import it into your module module

__version__ = '3.9.0'

def gitversion():
    try:
        from .git_version import get_gitversion
    except ImportError:
        return "unknown"
    return get_gitversion()
