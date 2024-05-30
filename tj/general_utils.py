import atexit
import inspect
import math
import os
import re
import signal
import sys
import threading
import time
import traceback
from collections import defaultdict
from numbers import Number
from types import LambdaType
from typing import Dict

import numpy as np
import torch
from attrdict import AttrDict
from attrdict.utils import get_with_default


class TimeIt(object):
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

        self._with_name_stack = []

    def __call__(self, name):
        self._with_name_stack.append(name)
        return self

    def __enter__(self):
        self.start(self._with_name_stack[-1])
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.stop(self._with_name_stack.pop())

    def start(self, name):
        assert name not in self.start_times, name
        self.start_times[name] = time.time()

    def stop(self, name):
        assert name in self.start_times, name
        self.elapsed_times[name] += time.time() - self.start_times[name]
        self.start_times.pop(name)

    def elapsed(self, name):
        return self.elapsed_times[name]

    def reset(self):
        self.start_times = dict()
        self.elapsed_times = defaultdict(int)

    def as_np_dict(self, names=None):
        names = self.elapsed_times.keys() if names is None else names
        dc = AttrDict()
        for n in names:
            dc[n] = np.array([self.elapsed_times[n]])
        return dc

    def __str__(self):
        s = ""
        names_elapsed = sorted(self.elapsed_times.items(), key=lambda x: x[1], reverse=True)
        for name, elapsed in names_elapsed:
            if "total" not in self.elapsed_times:
                s += "{0}: {1: <10} {2:.1f}\n".format(self.prefix, name, elapsed)
            else:
                assert self.elapsed_times["total"] >= max(self.elapsed_times.values())
                pct = 100.0 * elapsed / self.elapsed_times["total"]
                s += "{0}: {1: <10} {2:.1f} ({3:.1f}%)\n".format(self.prefix, name, elapsed, pct)
        if "total" in self.elapsed_times:
            times_summed = sum([t for k, t in self.elapsed_times.items() if k != "total"])
            other_time = self.elapsed_times["total"] - times_summed
            assert other_time >= 0
            pct = 100.0 * other_time / self.elapsed_times["total"]
            s += "{0}: {1: <10} {2:.1f} ({3:.1f}%)\n".format(self.prefix, "other", other_time, pct)
        return s


class ThreadedTimeIt:
    def __init__(self, prefix=""):
        self.prefix = prefix
        self.timeit_by_thread: Dict[int, TimeIt] = dict()
        self._thread_count = 0

    def __call__(self, name):
        if threading.get_ident() not in self.timeit_by_thread.keys():
            # prefix only for later initialized threads.
            self.timeit_by_thread[threading.get_ident()] = TimeIt(
                prefix=self.prefix + ("/thread_" + str(threading.get_ident()) if self._thread_count > 0 else "")
            )
            self._thread_count += 1
        return self.timeit_by_thread[threading.get_ident()].__call__(name)

    def __enter__(self):
        return self.timeit_by_thread[threading.get_ident()].__enter__()

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.timeit_by_thread[threading.get_ident()].__exit__(exc_type, exc_val, exc_tb)

    def start(self, name):
        return self.timeit_by_thread[threading.get_ident()].start(name)

    def stop(self, name):
        return self.timeit_by_thread[threading.get_ident()].stop(name)

    def elapsed(self, name):
        return self.timeit_by_thread[threading.get_ident()].elapsed(name)

    def reset(self):
        [self.timeit_by_thread[key].reset() for key in self.timeit_by_thread.keys()]

    def as_np_dict(self, names=None):
        dc = AttrDict()
        for key in sorted(self.timeit_by_thread.keys()):
            dc.combine(self.timeit_by_thread[key].as_np_dict(names=names))
        return dc

    def __str__(self):
        return "".join(self.timeit_by_thread[k].__str__() for k in sorted(self.timeit_by_thread.keys()))


timeit = ThreadedTimeIt()


def exit_on_ctrl_c():
    def signal_handler(signal, frame):
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)


@atexit.register
def cleanup_on_main():
    # I don't like that python doesn't do this automatically...
    import psutil

    for child in psutil.Process().children(recursive=True):
        child.send_signal(signal.SIGHUP)


def pdb_on_exception(debugger="pdb", limit=100):
    """Install handler attach post-mortem pdb console on an exception."""
    pass

    def pdb_excepthook(exc_type, exc_val, exc_tb):
        traceback.print_tb(exc_tb, limit=limit)
        __import__(str(debugger).strip().lower()).post_mortem(exc_tb)

    sys.excepthook = pdb_excepthook


def ipdb_on_exception():
    return pdb_on_exception("ipdb")


def get_from_ls(obj, attr, ls, default_idx=None, map_fn=None):
    if default_idx is not None:
        new_at = get_with_default(obj, attr, ls[default_idx], map_fn=map_fn)
    else:
        new_at = obj[attr]

    assert new_at in ls, "Specified %s, but not in supported list %s" % (new_at, ls)
    return new_at


class dummy_context_mgr:
    """
    A dummy context manager - useful for having conditional scopes (such
    as @maybe_no_grad). Nothing happens in this scope.
    """

    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def maybe_context(do, get_context, *args):
    """
    Optionally loads a context

    Args:
        do (bool): if True, the returned context will be get_context(), otherwise
            it will be a dummy context
    """
    return get_context(*args) if do else dummy_context_mgr()


def subclass_overrides_fn(self, base_cls, fn_name) -> bool:
    try:
        fn = getattr(type(self), fn_name)
    except AttributeError:
        return False
    return fn is not getattr(base_cls, fn_name)


def is_array(arr):
    return isinstance(arr, np.ndarray) or isinstance(arr, torch.Tensor)


def python_print(dd, ret_string=False, prefix="", indent=4, prefix_initial=True):
    s = f"{prefix}d(\n" if prefix_initial else "d(\n"

    inner_prefix = prefix + (" " * indent)
    for k, v in dd.items():
        if isinstance(v, AttrDict):
            inner = python_print(v, ret_string=True, prefix=inner_prefix, prefix_initial=False)
        else:
            # special printing cases
            if isinstance(v, str):
                inner = f"'{v}'"
            elif inspect.isclass(v):
                inner = f"{v.__name__}"
            elif inspect.isfunction(v) and isinstance(v, LambdaType) and v.__name__ == "<lambda>":
                lm = inspect.getsource(v).strip()
                inner = lm[lm.find("lambda") :]
                if inner[-1] == ",":
                    inner = inner[:-1]
            elif hasattr(v, "params"):
                v_param_str = python_print(v.params, ret_string=True, prefix=inner_prefix, prefix_initial=False)
                inner = f"{type(v).__name__}({v_param_str})"
            else:
                inner = str(v)
        s += f"{inner_prefix}{k}={inner},\n"

    if dd.is_empty():
        s = s[:-1] + ")"
    else:
        s += f"{prefix})"

    if ret_string:
        return s
    else:
        print(s)


def round_to_n(x, n=1):
    # round scalar to n significant figure(s)
    return round(x, -int(math.floor(math.log10(abs(x))) + (n - 1)))


def is_next_cycle(current, period):
    return period > 0 and current % period == 0


def listify(value_or_ls, desired_len, value_type=Number):
    if isinstance(value_or_ls, value_type):
        value_or_ls = [value_or_ls] * desired_len
    else:
        value_or_ls = list(value_or_ls)
    assert len(value_or_ls) == desired_len
    return value_or_ls


def value_if_none(in_val, default_val):
    if in_val is None:
        return default_val
    return in_val


def strlist(obj):
    obj = list(obj)
    for s in obj:
        assert isinstance(s, str), s
    return obj


def params_to_object(params, *args):
    return params.cls(*args, **params.leaf_filter(lambda k, v: k != "cls").as_dict())


def dc_skip_keys(dc: AttrDict, *keys):
    dc_new = AttrDict()
    for key in dc:
        if key not in keys:
            dc_new[key] = dc[key]

    return dc_new


def dc_remove_node_leaf_keys(dc: AttrDict, *keys, regex=False):
    """Remove nodes and leafs."""
    out = AttrDict()
    for k in dc.leaf_keys():
        # compute the tree of nodes upward from the leaf key
        node = os.path.dirname(k)
        node_tree = []
        while node != "":
            node_tree.append(node)
            node = os.path.dirname(node)
        # if the key is not present, but also the nodes do not match at all
        match_fn = (lambda s, k: re.fullmatch(re.compile(k), s) is not None) if regex else (lambda s, k: s == k)
        if k not in keys and not any(match_fn(node, key) for key in keys for node in node_tree):
            out[k] = dc[k]

    return out


if __name__ == "__main__":
    data = AttrDict(
        node=AttrDict(
            key=1,
            key2=2,
            sub_node=AttrDict(sub_key=3, sub_key2=5),
        ),
        node2=AttrDict(key3="test"),
        base_key="out",
    )

    out = dc_remove_node_leaf_keys(data, "node")
    out.pprint()
    assert "node" not in out.keys()
    assert "node2" in out.keys()
    assert "node2/key3" in out.leaf_keys()
    assert "base_key" in out.keys()

    out = dc_remove_node_leaf_keys(data, "node/sub_node")
    out.pprint()
    assert "node" in out.keys()
    assert "node/sub_node" not in out.node_leaf_keys()
    assert "node2" in out.keys()
    assert "node2/key3" in out.leaf_keys()
    assert "base_key" in out.keys()