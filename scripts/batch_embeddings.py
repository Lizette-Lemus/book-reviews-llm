import pandas as pd
import numpy as np

import os
import time

from openai import OpenAI
from dotenv import load_dotenv

# Load API key from .env file
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key = OPENAI_API_KEY)
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API key not found! Make sure it's in the .env file.")

MODEL = "text-embedding-3-small"
BASE_DIR = os.path.abspath(os.path.join(os.getcwd(), ".."))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATA_PATH = os.path.join(DATA_DIR, "books_data.csv")

# Define batch storage directory
BATCH_DIR = os.path.join(DATA_DIR, "batch_embeddings")
os.makedirs(BATCH_DIR, exist_ok=True)

# Load the CSV file
df = pd.read_csv(DATA_PATH)

# Replace NaNs with a placeholder
df["description"] = df["description"].fillna("MISSING")

# Convert descriptions to a list
texts = df["description"].tolist()

# Define batch size (adjust for API limits)
batch_size = 500

# Start timer
start_time = time.time()
all_batch_files = []
n_batches = (len(texts) + batch_size - 1) // batch_size

for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_start_time = time.time()
        # Generate embeddings for the batch
        response = client.embeddings.create(input=batch, model=MODEL)
        batch_embeddings = [item.embedding for item in response.data]

        batch_end_time = time.time()
        batch_time = batch_end_time - batch_start_time

        # Create a batch dataframe
        batch_df = pd.DataFrame({
            "description": batch,
            "embedding": batch_embeddings
        })

        # Save batch to a file::
        batch_filename = os.path.join(BATCH_DIR, f"batch_{i // batch_size + 1}.csv")
        batch_df.to_csv(batch_filename, index=False)
        all_batch_files.append(batch_filename)
        batch_number = (i // batch_size) + 1 
        print(f"Processed {batch_number} of {n_batches} total batches ({batch_time:.2f} sec) and saved to {batch_filename}")

        time.sleep(1)  # Prevent hitting rate limits

# End timer
end_time = time.time()
total_time = end_time - start_time

print(f"Completed embedding generation for {len(df)} rows!")
print(f"Total time taken: {total_time:.2f} seconds")

# Merge all batch files into a final CSV
final_file = os.path.join(DATA_DIR, "books_data_embeddings.csv")
df_all = pd.concat([pd.read_csv(f) for f in all_batch_files], ignore_index=True)
df_all.replace("MISSING", np.nan, inplace=True)  # Restore NaNs
df_all.to_csv(final_file, index=False)

print(f"All batches merged and saved as {final_file}")

