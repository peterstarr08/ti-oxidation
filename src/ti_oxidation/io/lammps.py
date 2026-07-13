import io
import re

def read_file(path):
    with open(path, 'r') as f:
        content = f.read()
        print(f"File {path} read successfully!")
    return content

class LAMMPS_Input:
    def __init__(self, content):
        self.content = content

    def __str__(self):
        return str(self.content)

    @classmethod
    def from_file(cls, path):
        inp = cls(read_file(path))
        return inp

    def update_variable(self, var_name, value):
        regEx = fr'variable\s*{var_name}\s*equal\s*.*\n'
        print(f"Variable found {regEx}")
        value = f'variable {var_name} equal {value}\n'
        print(f"Replacing with {value}")

        return LAMMPS_Input(re.sub(regEx, value, self.content))



def lammps_log_extract(path, keys):
    keys = set(keys)
    result = {k: None for k in keys}

    with open(path, 'r') as f:
        for line in f:
            if '=' not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()

            if key in keys:
                result[key] = value.strip()

    return result
