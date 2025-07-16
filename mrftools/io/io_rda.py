""" read rda (MRS) files """

# raw text to find in RDA file (which is non utf-8)
BEGIN = b">>> Begin of header <<<"
END = b">>> End of header <<<"


def extract_voi(filename):
    header = read_rda_header(filename)

    # extract relevant info from header
    position = [
        header["VOIPositionSag"],
        header["VOIPositionCor"],
        header["VOIPositionTra"],
    ]

    normal = [
        header["VOINormalSag"],
        header["VOINormalCor"],
        header["VOINormalTra"],
    ]
    size = [
        header["VOIReadoutFOV"],
        header["VOIPhaseFOV"],
        header["VOIThickness"],
    ]
    rot = header["VOIRotationInPlane"]

    rotation = rot

    return {
        "position": position,
        "normal": normal,
        "size": size,
        "rotation": rotation,
    }


def read_rda_header(filename):
    """load rda file header"""
    with open(filename, "rb") as fp:
        lines = fp.read()
    raw = lines[lines.find(BEGIN) + len(BEGIN) : lines.find(END)]
    return parse_rda_header(raw)


def parse_rda_header(raw):
    """prase rda file header"""

    # encoding of rda file
    try:
        # encoding is ISO-8859-1
        decoded = raw.decode("iso_8859_1")
    except UnicodeDecodeError:
        raise IOError("Unknown encoding")

    header = {}
    for line in decoded.split("\n"):
        if not line.strip():
            # empty lines
            continue

        # split items
        item = [i.strip() for i in line.split()]
        if item[0][-1] != ":":
            continue

        # cast value
        if len(item) > 1:
            header[item[0][:-1]] = parse_value(item[1])

    return header


def parse_value(value):
    """parse string value into number if possible"""
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value.replace("'", "").replace('"', "")
