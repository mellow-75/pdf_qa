# Running the code - 
Follow the given steps to run on your local
#### ->PDF Text Extraction: Extracts text from PDFs using the fitz library.
#### ->NLP Preprocessing: Preprocesses text with tokenization, stop word removal, lemmatization, and punctuation removal.
#### ->Embedding Models: Supports multiple embedding models including Word2Vec, GloVe, FastText and BERT.
#### ->User Interface: Simple Streamlit interface for uploading PDFs, inputting questions, and selecting embedding models.
#### ->Answer Retrieval: Finds and presents the most relevant sentences in response to user queries.

### Installation

#### Clone the repository: 
```bash
git clone https://github.com/mellow-75/pdf_qa.git
```

#### Create the Directory:
```bash
mkdir -p static
```

#### Install the required packages:
```bash
pip install -r requirements.txt
```

#### Run the Streamlit app:

```bash
streamlit run pdf_ans.py
```

### Methodology

#### PDF Text Extraction
#### Text is extracted from PDF documents using the fitz library (PyMuPDF). This ensures that text from various PDF formats is accurately captured for further processing.

#### Text Preprocessing
#### The extracted text undergoes preprocessing to prepare it for analysis:
#### -> Tokenization: Breaking down text into individual words or tokens.
#### -> Stop Word Removal: Removing common words that do not carry significant meaning.
#### -> Lemmatization: Converting words to their base or dictionary form.
#### -> Punctuation Removal: Eliminating punctuation marks to clean the text.

#### Text Embedding Models
#### The system applies various embedding models to generate numerical representations of the text:
#### -> Word2Vec (Continuous Bag of Words (CBOW) and Skip-Gram)
#### -> GloVe
#### -> FastText
#### -> BERT

#### These models transform text into vectors, allowing for efficient similarity computations.

### User Interaction
#### The Streamlit interface provides a simple platform for users to interact with the system:
#### ->Upload PDFs
#### ->Input Questions
#### ->Select Embedding Models

### Answer Generation
#### Based on the chosen embedding model, the system computes similarity between the user's query and the sentences in the document. The most relevant sentences are then displayed as answers.

### Results
#### The PDF_Answering_ai accurately retrieves the most relevant sentences from PDF documents in response to user queries, providing precise and informative answers.


