import os
from time import time
from tqdm import tqdm
from urllib.request import urlretrieve


def _download_urlretrieve(url, filename, verbose):
    """
    Download using urlretrieve.
    """
    class _(object):
        progress_bar = None

    def report_hook(count, chunk_size, total_size):
        if verbose < 2:
            return
        if _.progress_bar is None:
            if total_size == -1:
                total_size = None
            _.progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
        else:
            _.progress_bar.update(chunk_size)

    try:
        urlretrieve(url, filename, report_hook)
    except:
        if os.path.exists(filename):
            os.remove(filename)
        raise


def download(url, filename, verbose=2):
    """
    Download a file, if it does not already exist.

    url       Remote location.
    filename  Local location.
    verbose   Level of verbosity:
              * 0 silent.
              * 1 three log lines (from, to, time taken).
              * 2 log lines with progress bar.
    """
    if os.path.exists(filename):
        return
    assert verbose in {0, 1, 2}
    if verbose:
        print('Downloading: %s' % url)
        print('         to: %s' % filename)
    if verbose == 1:
        t0 = time()
    dirname = os.path.dirname(filename)
    if dirname and not os.path.exists(dirname):
        os.mkdir(dirname)
    _download_urlretrieve(url, filename, verbose)
    if verbose == 1:
        t = time() - t0
        print('...took %.3f sec.' % t)


def get_zen_dir():
    """
    Get the zen root dir.

    Override the default by setting $ZEN_HOME in your env vars.
    """
    d = os.environ.get('ZEN_HOME')
    if d:
        return d
    return os.path.expanduser('~/.zen/')


def get_dataset_dir(dataset):
    """
    Get the zen directory location for a given dataset.
    """
    return os.path.join(get_zen_dir(), 'dataset', dataset)
