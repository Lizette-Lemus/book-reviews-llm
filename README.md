# Book Reviews LLM
**Semantic search using OpenAI embeddings.**  

## Overview
This project leverages **OpenAI’s text-embedding-3-small model** to enable **semantic search** for **book recommendations**.

## Project Structure
```
book-reviews-llm/
│── data/
│   ├── books_data_sample.csv                # Sample example of original dataset
│   ├── books_data_sample_embeddings.csv     # Embeddings example 
│── notebooks/
│   ├── 01_generate_embeddings.ipynb         # Generate embeddings for a sample of books_data.csv
│   ├── 02_embeddings_visualization.ipynb    # Visualize embeddings
│   ├── 03_query_completion.ipynb            # Custom query completion example
│── scripts/
│   ├── batch_embeddings.py   # Script for batch processing embeddings
│── requirements.txt
│── .env
│── README.md
```

## Data 

Download 
```books_data.csv``` from [Amazon Book Reviews](https://www.kaggle.com/datasets/mohamedbakhet/amazon-books-reviews) and save in the ```data/``` directory

## Create a Virtual Environment & Install Dependencies
```
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Set Up OpenAI API Key
Create a ```.env``` file and add your Open AI key ```OPENAI_API_KEY=your-api-key-here```

## Generate embeddings
Inside ```scripts```  run ```python batch_embeddings.py``` 

This will:
- Store intermediate batch files in data/batches/
- Merge all batches into a final ```books_data_embeddings.csv``` file

## Custom query completion
To explore book recommendations using semantic search, open the Jupyter Notebook: ```03_query_completion.ipynb```

This notebook shows how to:
- Retrieve similar books based on meaning
- Use custom query completion to answer a question about retrieved book reviews

## Contact
Reach out via GitHub Issues or email me at llg920420@gmail.com
