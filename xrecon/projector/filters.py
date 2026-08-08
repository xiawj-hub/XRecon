import torch

FILTERS = {}


def register_filter(name):
    def decorator(func):
        FILTERS[name] = func
        return func
    return decorator


def filter_generate(num_det, det_size, filter_type="ramp", dtype=None):
    if filter_type not in FILTERS:
        raise ValueError(f"Filter type {filter_type} not supported. Available: {list(FILTERS.keys())}")
    return FILTERS[filter_type](num_det, det_size, dtype)


@register_filter("ramp")
def ramp(num_det, det_size, dtype):
    """Ramp filter (Ram-Lak) for CT reconstruction."""
    filt = torch.zeros(num_det * 2 - 1, dtype=dtype)
    idx = torch.arange(num_det * 2 - 1)
    x = idx - num_det + 1

    odd_mask = x % 2 != 0
    epsilon = 1e-12

    filt[odd_mask] = -1 / ((torch.pi ** 2) * (det_size ** 2) * ((x[odd_mask] ** 2) + epsilon))
    filt[num_det - 1] = 1 / (4 * det_size ** 2)

    return filt


@register_filter("shepplogan")
def shepplogan(num_det, det_size, dtype):
    """Shepp-Logan filter"""
    idx = torch.arange(num_det * 2 - 1)
    x = idx - num_det + 1
    filt = -2 / (torch.pi ** 2 * det_size ** 2 * (4 * x ** 2 - 1))
    return filt.to(dtype)


def _apply_window(filt, window_func):
    num_det = (len(filt) + 1) // 2
    w = torch.cat((torch.arange(0, num_det), torch.arange(-num_det + 1, 0))) * torch.pi / num_det
    window = window_func(w.double())
    filter_freq = torch.fft.fft(filt.double())
    filtered_freq = filter_freq * window
    return torch.fft.ifft(filtered_freq).real.float()


@register_filter("cosine")
def cosine(num_det, det_size, dtype):
    """Cosine filter"""
    ramp_filter = ramp(num_det, det_size, dtype)
    return _apply_window(ramp_filter, lambda w: torch.cos(w / 2))


@register_filter("hamming")
def hamming(num_det, det_size, dtype):
    """Hamming filter"""
    ramp_filter = ramp(num_det, det_size, dtype)
    return _apply_window(ramp_filter, lambda w: 0.54 + 0.46 * torch.cos(w))


@register_filter("hann")
def hann(num_det, det_size, dtype):
    """Hann filter"""
    ramp_filter = ramp(num_det, det_size, dtype)
    return _apply_window(ramp_filter, lambda w: 0.5 * (1 + torch.cos(w)))
