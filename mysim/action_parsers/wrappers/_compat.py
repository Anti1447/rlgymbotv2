# mysim/action_parsers/_compat.py
def safe_get_action_space(obj, agent=None):
    """
    Call obj.get_action_space() whether it expects 0 or 1 positional args.
    """
    try:
        return obj.get_action_space()         # new API (no args)
    except TypeError:
        return obj.get_action_space(agent)    # old API (agent arg)
