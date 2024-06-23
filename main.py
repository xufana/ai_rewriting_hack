import openai
import os
import sys
import re
import fire
import random
import pandas as pd

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from langchain_qdrant import Qdrant
from langchain.docstore.document import Document
from langchain_community.embeddings.huggingface import HuggingFaceEmbeddings

from utils import clean_markdown


def contains_markdown(text):
    # Define a list of common Markdown patterns
    markdown_patterns = [
        r'\*\*(.*?)\*\*',        # Bold (**bold**)
        r'\*(.*?)\*',            # Italics (*italics*)
        r'\_\_(.*?)\_\_',        # Bold (__bold__)
        r'\_(.*?)\_',            # Italics (_italics_)
        r'\~\~(.*?)\~\~',        # Strikethrough (~~strikethrough~~)
        r'\`(.*?)\`',            # Inline code (`code`)
        r'\#\s.*',               # Headers (# Header)
        r'\>\s.*',               # Blockquotes (> Blockquote)
        r'\[(.*?)\]\((.*?)\)',   # Links [text](url)
        r'\!\[(.*?)\]\((.*?)\)', # Images ![alt text](url)
        r'\n\s*\n',              # Paragraphs (blank line)
        r'\n\d+\.\s',            # Ordered lists (1. item)
        r'\n\-\s',               # Unordered lists (- item)
    ]
    
    for pattern in markdown_patterns:
        if re.search(pattern, text):
            return True
    return False

def contains_russian(text):
    # Define a regular expression pattern to match any Russian character
    russian_pattern = r'[а-яА-ЯёЁ]'
    
    # Search for the pattern in the text
    if re.search(russian_pattern, text):
        return True
    else:
        return False

def extract_number_from_string(s: str) -> int:
    """
    Extract a number from a string
    """

    pattern = r'\d+'
    match_ = re.search(pattern, s)
    if match_:
        return int(match_.group())
    else:
        return 1 # if LLM could not decide whether it is a generated text, rewrite it anyway

def check_ai(text, llm="gpt-3.5-turbo-0125") -> bool:
    """
    A classifier to check whether the text is AI-generated or not.
    """

    system = """You are a helpful assistant that checks if the text was written by AI, not by a human.
    Answer only 0, if the text is written by a human.
    Answer only 1, if the text is written by an AI.
    I need only one number. I'll give you 20$ if you follow my instructions correctly and give me just one number."""

    response = openai.chat.completions.create(
        model=llm,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"Text: {text}"}
        ],
        max_tokens=5,
        # logprobs=1,
    )
    answer = response.choices[0].message.content.strip()
    text_class = extract_number_from_string(answer)
    return text_class == 1


def paraphrase_text(text,  model="gpt-3.5-turbo"):
    """
    Trying to rewrite a text.
    """

    system = """You are a dedicated writer that paraphrases texts, while keeping the original style and answer using the same language.
    Try not to fantasize and avoid list structures in data presentation, use natural language. Do not use bullet-point structure or lists. Do not use complex vocabulary
    Do not use many linking words, do not repeat arguments, but use various sentence structures. Instead of using advanced words talk to me like I am not a native speaker.
    I'll give you 100$ if you do everything correctly. Use the original language of the user"""

    response = openai.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": system + '\n' + text}
        ],
        temperature=0.85,
        top_p=0.85,
        presence_penalty=-1,
    )
    paraphrased_text = response.choices[0].message.content.strip()
    return paraphrased_text

def main(input_file='input.txt', output_file='output.txt'):
    openai.api_key = os.getenv('AI_TOKEN')

    if openai.api_key is None:
        print("Error: AI_TOKEN environment variable is not set.")
        sys.exit(1)

    try:
        with open(input_file, 'r', encoding='utf-8') as file:
            text = file.read()
    except FileNotFoundError:
        print(f"Error: The file {input_file} was not found.")
        sys.exit(1)

    russian = contains_russian(text)

    db_docs = pd.read_csv("qdrant_data/texts.csv")
    db_docs = [Document(page_content=row.paraphrased, metadata={"original": row.text}) for _, row in db_docs.iterrows()]

    embeddings = HuggingFaceEmbeddings(
        model_name="DeepPavlov/distilrubert-tiny-cased-conversational-v1"
        )
    qdrant = Qdrant.from_documents(
        db_docs,
        embeddings,
        location=":memory:",
        collection_name="my_documents",
        )
        
    qdrant_retriever = qdrant.as_retriever(search_kwargs={"k": 10}, verbose=True)

    print("checking AI-gen text...")
    
    if contains_markdown(text):
        # if there is a markdown, we rewrite text completely
        relevant = qdrant_retriever.get_relevant_documents(text)

        if russian:
            context = '\n'.join([f"Перефразируй данный текст: {relevant_doc.page_content}\nИтог: {relevant_doc.metadata['original']}" for relevant_doc in relevant])
            context = context + f"\nПерефразируй данный текст: {text}\nИтог: "
        else:
            context = '\n'.join([f"Paraphrase the following text: {relevant_doc.page_content}\nResult: {relevant_doc.metadata['original']}" for relevant_doc in relevant])
            context = context + f"\nParaphrase the following text: {text}\nResult: "
        
        text = paraphrase_text(context)
    elif check_ai(text):
        # if there is no markdown we use classifier and rewrite only some parts of the text
        text_splitter = SemanticChunker(OpenAIEmbeddings(openai_api_key=os.getenv('AI_TOKEN')))
        docs = text_splitter.create_documents([text.strip()])
        if (len(docs) == 0):
            docs = text.strip().split('\n')

        try:
            sample_ids = random.sample(range(0, len(docs) - 1), 5)
        except:
            sample_ids = [0]
        result = []

        for i, doc in enumerate(docs):
            if i in sample_ids:
                relevant = qdrant_retriever.get_relevant_documents(doc.page_content)

                if russian:
                    context = '\n'.join([f"Перефразируй данный текст: {relevant_doc.page_content}\nИтог: {relevant_doc.metadata['original']}" for relevant_doc in relevant])
                    context = context + f"\nПерефразируй данный текст: {doc.page_content}\nИтог: "
                else:
                    context = '\n'.join([f"Paraphrase the following text: {relevant_doc.page_content}\nResult: {relevant_doc.metadata['original']}" for relevant_doc in relevant])
                    context = context + f"\nParaphrase the following text: {doc.page_content}\nResult: "
                chunk = paraphrase_text(context)
            else:
                chunk = doc.page_content
            result.append(chunk)
        
        text = '\n'.join(result)

    if contains_markdown(text):
        text = clean_markdown(text)

    with open(output_file, 'w', encoding='utf-8') as file:
        file.write(text)

    print(f"Paraphrased text has been saved to {output_file}")

if __name__ == "__main__":
    fire.Fire(main)