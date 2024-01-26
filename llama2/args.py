"""
NOTE:
hack import this args.py to change current py file global args
use: exec(open(f"{_cur_work_dir}/args.py").read()) instead of import this file
"""

def change_global_args(begin_index=1):
    import sys
    from ast import literal_eval
    for arg in sys.argv[begin_index:]:
        # assume it's a --key=value argument
        assert arg.startswith('--')
        key, val = arg.split('=')
        key = key[2:]
        if key in globals():
            try:
                # attempt to eval it it (e.g. if bool, number, or etc)
                attempt = literal_eval(val)
            except (SyntaxError, ValueError):
                # if that goes wrong, just use the string
                attempt = val
            # ensure the types match ok
            assert type(attempt) == type(globals()[key])
            # cross fingers
            print(f"Overriding: {key} = {attempt}")
            globals()[key] = attempt
        else:
            raise ValueError(f"Unknown config key: {key}")

def get_config_from_global_args():
    config_keys = [
        k
        for k, v in globals().items()
        if not k.startswith("_") and isinstance(v, (int, float, bool, str))
    ]
    config = {k: globals()[k] for k in config_keys}  # will be useful for logging
    return config