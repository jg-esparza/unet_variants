
def review_registry_availability(registry, kind, name):
    key = name.lower().strip()
    if key not in registry:
        available = ", ".join(sorted(registry.keys()))
        raise ValueError(f"Unknown {kind} '{name}'. Available: [{available}]")
    return key

def validate_instance(name, obj, expected_type):
    if not isinstance(obj, expected_type):
        raise TypeError(f"Builder for '{name}' did not return '{expected_type}'")
