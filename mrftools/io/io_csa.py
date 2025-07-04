""" parse Siemens's CSA  header """


def get_csa(stack, tag=(0x29, 0x1020), index=0):
    """return CSA dictionary from DICOM stack"""
    # get raw string and parse into CSA dictionary
    raw = stack[tag][index]
    return parse_csa(raw)


def parse_csa(raw_csa):
    """parse Siemens CSA field 'CSA Series Header Info' (0x0029, 0x1120)"""
    raw_csa = raw_csa.decode("utf-8", "ignore")
    begin = "### ASCCONV BEGIN"
    end = "### ASCCONV END ###"
    csa_subset = raw_csa[raw_csa.find(begin) + 9 : raw_csa.rfind(end)]

    split = csa_subset.split("\n")
    csa_dict = {}
    for i, line in enumerate(split):
        # print line
        if not line.strip():
            continue
        item = [i.strip() for i in line.split()]
        if item[1] != "=":
            continue
        csa_dict[item[0]] = parse_value(item[2])
    return csa_dict


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
