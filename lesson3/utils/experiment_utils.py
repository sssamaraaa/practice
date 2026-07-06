def build_architecture(kind, base):
    if kind == "constant":
        return [base, base, base]

    if kind == "expanding":
        return [base // 4, base // 2, base]

    if kind == "shrinking":
        return [base, base // 2, base // 4]
    

def build_config(hidden):
    layers = []

    for h in hidden:
        layers.append({"type": "linear", "size": h})
        layers.append({"type": "relu"})

    return {
        "layers": layers
    }