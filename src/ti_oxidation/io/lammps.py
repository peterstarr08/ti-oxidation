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

    def find_variable(self, var_list):
        regEx = re.findall(rf"^variable\s*[{"".join(var_list)}]\s*equal\s.*\n", self.content, re.MULTILINE)
        content = [( item.split()[1], float(item.split()[-1])) for item in regEx]
        if len(content)==0:
            print(f"Warning: No results for {var_list} ")
        return content 


    def update_variable(self, var_name, value):
        regEx = fr'variable\s*{var_name}\s*equal\s*.*\n'
        #print(f"Variable found {regEx}")
        value = f'variable {var_name} equal {value}\n'
        #print(f"Replacing with {value}")

        return LAMMPS_Input(re.sub(regEx, value, self.content))


# For backward compatibility
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


def parse_energy_minim(lines):
    # print(lines)
    keys = {}
    for i, line in enumerate(lines):
        if "Stopping criterion" in line:
            keys['stop'] = line.split('=')[-1].strip()

        if "Energy initial, next-to-last, final =" in line:
            energy_vals = lines[i+1].strip().split()
            keys['ener_init'] = float(energy_vals[0])
            keys['ener_penult'] = float(energy_vals[1])
            keys['ener_final'] = float(energy_vals[2])

            keys['dE'] = keys['ener_final'] - keys['ener_penult']
        if "Force max component" in line:
            keys['force_max_comp'] = float(line.split('=')[-1].strip().split()[-1])

    return keys


class LAMMPS_Log:
    def __init__(self, path):
        self.path = path
        with open(path) as f:
            lines = f.readlines()
        self.variables = {}
        for i, line in enumerate(lines):
            # Check for minim block
            if 'Minimization stats:' in line:
                self.minim_stat = parse_energy_minim(lines[i:i+8])

            # Potential variables
            if '=' in line:
                strip_search = line.split('=')
                strip_var = strip_search[0].strip()
                if not ' ' in strip_var:
                    strip_value = strip_search[-1].strip()
                    self.variables[strip_var] = strip_value
