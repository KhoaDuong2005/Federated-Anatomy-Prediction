def validate_modality(modality: str):
    allowed = ["mri", "xray", "ct", "unknown"]
    if modality not in allowed:
        raise ValueError(f"Invalid modality {modality}, currently supported: {', '.join(allowed)}")
    return modality

def validate_body_part(body_part: str):
    allowed = ["chest", "brain", "leg", "abdomen", "unknown"]
    if body_part not in allowed:
        raise ValueError(f"Invalid body_part {body_part}, currently supported: {', '.join(allowed)}")
    return body_part

def validate_is_anatomy(is_anatomy: str):
    if is_anatomy == "True":
        return "True"
    elif is_anatomy == "False":
        return "False"
    elif is_anatomy == None:
        return None
    raise ValueError(f"Invalid input {is_anatomy}, currently support [True, False]")