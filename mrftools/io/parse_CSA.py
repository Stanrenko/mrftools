""" parse Siemens CSA structure """

# coding=utf-8


def to_number(s):
    try:
        return int(s)
    except:
        pass

    try:
        return float(s)
    except:
        pass
    return s


def parse_ASCCONV(header):
    """parse Siemens' CSA header
    DICOM key: (0x0029, 0x1020)
    """
    header = header.decode(errors="ignore")
    raw = header[header.find("BEGIN ###") + 9 : header.find("### ASCCONV END")]

    split = raw.split("\n")
    d = {}
    for line in split:
        # print line
        if len(line.strip()) == 0:
            continue
        item = [i.strip() for i in line.split()]
        if item[1] != "=":
            continue

        d[item[0]] = to_number(item[2])
    return d
