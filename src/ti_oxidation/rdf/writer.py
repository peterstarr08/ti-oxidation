import io

def write_rdf(file, grs, rdf_bins):
    with open(file, mode='w') as f:
        f.write('index,r,g(r)\n')
        for index, (r, gr) in enumerate(zip(rdf_bins, grs)):
            f.write(f'{index},{r},{gr}\n')
        print(f"File written to {file}")
