def _logger_init():
    import logging
    from . import dobj
    res = logging.getLogger('NIFTy4')
    res.setLevel(logging.DEBUG)
    if dobj.rank == 0:
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)
        res.addHandler(ch)
    else:
        res.addHandler(logging.NullHandler())
    return res

logger = _logger_init()
