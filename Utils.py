import pandas as pd
def generate_custom_header(file_path):
    """Generate a custom header by concatenating the second and third rows."""
    # Read the first three rows to create the custom header
    header_rows = pd.read_csv(file_path, header=None, nrows=3)
    # Create the new header by concatenating the second and third rows
    new_header = header_rows.iloc[1] + '_' + header_rows.iloc[2]
    return new_header

def read_csv_in_chunks(file_path, chunksize=1000):
    """Read a CSV file in chunks, using a custom header."""
    new_header = generate_custom_header(file_path)
    # Create an iterator for the CSV file, skipping the first three rows
    chunk_iter = pd.read_csv(file_path, skiprows=3, header=None, chunksize=chunksize, iterator=True)
    
    # Iterate over each chunk and yield it with the new header
    for chunk in chunk_iter:
        chunk.columns = new_header
        yield chunk