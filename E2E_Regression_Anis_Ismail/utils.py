def transform_y(y, m0_scale=500):
    return y / m0_scale


def inv_transform(y, m0_scale=500):
    return y * m0_scale


def logger(s, f, run_logger=True):
    print(s)
    if run_logger:
        f.write('%s\n' % str(s))
