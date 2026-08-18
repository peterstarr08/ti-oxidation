import io

def write_rdf(file, grs, rdf_bins):
    with open(file, mode='w') as f:
        f.write('index,r,g(r)\n')
        for index, (r, gr) in enumerate(zip(rdf_bins, grs)):
            f.write(f'{index},{r},{gr}\n')
        print(f"File written to {file}")

def write_rdf_log(file, grs, rdf_bins, file_log, size, local_density):
    write_rdf(file, grs, rdf_bins)
    with open(file_log, mode='w') as f:
        f.write(f"{size}\n{local_density}")
        print(f"Log file written to {file_log}")
