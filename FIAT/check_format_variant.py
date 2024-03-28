import re
from FIAT.macro import IsoSplit, AlfeldSplit


def check_format_variant(variant, degree):
    if variant is None:
        variant = "integral"

    match = re.match(r"^integral(?:\((\d+)\))?$", variant)
    if match:
        variant = "integral"
        extra_degree, = match.groups()
        extra_degree = int(extra_degree) if extra_degree is not None else 0
        interpolant_degree = degree + extra_degree
        if interpolant_degree < degree:
            raise ValueError("Warning, quadrature degree should be at least %s" % degree)
    elif variant == "point":
        interpolant_degree = None
    else:
        raise ValueError('Choose either variant="point" or variant="integral"'
                         'or variant="integral(q)"')

    return variant, interpolant_degree


def parse_lagrange_variant(variant):
    options = variant.replace(" ", "").split(",")
    assert len(options) <= 2
    point_variant = "equispaced"
    splitting = None

    for pre_opt in options:
        opt = pre_opt.lower()
        if opt == "alfeld":
            splitting = AlfeldSplit
        elif opt == "iso":
            splitting = IsoSplit
        elif opt.startswith("iso"):
            match = re.match(r"^iso(?:\((\d+)\))?$", opt)
            k, = match.groups()
            splitting = lambda T: IsoSplit(T, int(k))
        elif opt in ["equispaced", "gll", "spectral"]:
            point_variant = opt
        else:
            raise ValueError("Illegal variant option")

    return splitting, point_variant
