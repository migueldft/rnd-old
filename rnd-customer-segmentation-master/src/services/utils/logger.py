import logging
import sys
import time


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        if 'log_time' in kw:
            name = kw.get('log_name', method.__name__.upper())
            kw['log_time'][name] = int((te - ts) * 1000)
        #else:
            #logger.info('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result
    return timed


def set_up_logging(name='review_mediator'):
    out_hdlr = logging.StreamHandler(sys.stdout)
    out_hdlr.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.addHandler(out_hdlr)

    return logger


def verify_params(params, required_fields, step):
    for req in required_fields:
        if not params.get(req, ""):
            raise Exception(f'Param "{req}" is required and it is not defined at step {step}.')