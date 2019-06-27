import logging

# set up logging to file - see previous section for more details
logging.basicConfig(level=logging.DEBUG,
                    format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                    datefmt='%Y-%m-%d %H:%M')
# define a Handler which writes INFO messages or higher to the sys.stderr
console = logging.StreamHandler()
console.setLevel(logging.DEBUG)
# set a format which is simpler for console use
formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
# tell the handler to use this format
console.setFormatter(formatter)

# Now, define a couple of other loggers which might represent areas in your
# application:


def log_in_out(func):

    def decorated_func(*args, **kwargs):
        print("Begin:  ", func.__name__)
        result = func(*args, **kwargs)
        print("End:  ", func.__name__)
        return result

    return decorated_func

