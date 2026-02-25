import numpy as np

# def assign_env_config(self, kwargs):
#     print("Assigning configuration...")
#     for key, value in kwargs.items():
#         print(f"Trying to set {key} to {value}")
#         if hasattr(self, key):
#             print(f"Setting {key} to {value}")
#             setattr(self, key, value)
#         else:
#             print(f"{self} has no attribute, {key}")
#             raise AttributeError(f"{self} has no attribute, {key}")

def assign_env_config(self, kwargs):
    print("Assigning configuration...")
    print(len(kwargs), "kwargs")
    for key, value in kwargs.items():
        print(f"Trying to set {key} to {value!r}")

        # 1) ensure it's in the schema
        if key not in self._CONFIG_SCHEMA:
            raise AttributeError(f"{self!r} has no config attribute '{key}'")

        # 2) type‐check
        expected_type = self._CONFIG_SCHEMA[key]
        if not isinstance(value, expected_type):
            raise TypeError(
                f"Config '{key}' expects type {expected_type.__name__}, "
                f"got {type(value).__name__}"
            )

        # 3) finally setattr
        print(f"Setting {key} to {value!r}")
        setattr(self, key, value)

def flatten_dict(dictionary, parent_key='', separator=';'):
    """
    Recursively flatten a nested dictionary or list into a dict of (string_key -> value).

    Changes from the original:
    - We also flatten lists, producing sub-keys like "parent;listKey;0", "parent;listKey;1", etc.
    - Non-numeric or empty placeholders are handled in a consistent way.
    """
    items = []
    for key, value in dictionary.items():
        # Convert the current key to string for safe concatenation
        str_key = str(key)
        new_key = parent_key + separator + str_key if parent_key else str_key

        if isinstance(value, dict):
            # Recursively flatten any nested dictionaries
            if value:
                items.extend(flatten_dict(value, new_key, separator=separator).items())
            else:
                # If it's an empty dict, store it as 0 or skip it
                items.append((new_key, 0.0))

        elif isinstance(value, list):
            # Flatten each item in the list
            for i, elem in enumerate(value):
                child_key = f"{new_key}{separator}{i}"
                if isinstance(elem, dict):
                    # If list element is another dict, recurse
                    if elem:
                        items.extend(flatten_dict(elem, child_key, separator=separator).items())
                    else:
                        items.append((child_key, 0.0))
                elif isinstance(elem, list):
                    # If list element is itself a list, wrap it in a dict for recursion
                    # or flatten inline. Here we wrap in a dict to reuse flatten_dict logic:
                    sub_list_dict = {i: elem}
                    items.extend(flatten_dict(sub_list_dict, child_key, separator=separator).items())
                else:
                    # Base case: numeric or string or something else
                    items.append((child_key, elem))
        
        else:
            # Base case: numeric or string or other
            items.append((new_key, value))

    return dict(items)

def flatten_and_track_mappings(dictionary, separator=';'):
    """
    1) Recursively flatten the input (dict + possibly nested lists)
    2) Build a mapping from array index -> path
    3) Convert any non-numeric to 0
    4) Return (flattened_array, index_mapping)
    """
    # 1) Flatten
    flattened_dict = flatten_dict(dictionary, separator=separator)

    # 2) Build index->keypath mappings and numeric array
    mappings = []
    flattened_values = []
    for index, (key, value) in enumerate(flattened_dict.items()):
        path_components = key.split(separator)
        mappings.append((index, path_components))

        # 3) Convert non-numeric to 0.0
        if isinstance(value, (int, float)):
            flattened_values.append(value)
        else:
            flattened_values.append(0.0)

    # 4) Convert to numpy array
    flattened_array = np.array(flattened_values, dtype=np.float32)
    return flattened_array, mappings