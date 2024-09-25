import gradio as gr
import os
import numpy as np
import hashlib
from anthropic import Anthropic
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from bidi.algorithm import get_display
import arabic_reshaper

# Load environment variables
load_dotenv()

# Initialize Pinecone
def init_pinecone():
    pc = Pinecone(api_key=os.getenv('PINECONE_API_KEY'))
    if os.getenv('PINECONE_INDEX') not in pc.list_indexes().names():
        pc.create_index(
            name=os.getenv('PINECONE_INDEX'),
            dimension=1536,
            metric='euclidean',
            spec=ServerlessSpec(cloud='aws')
        )
    return pc

pc = init_pinecone()
index = pc.Index(os.getenv('PINECONE_INDEX'))

# Initialize Anthropic
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
claude = Anthropic(api_key=anthropic_api_key)

def embed_query(query_text):
    try:
        response = claude.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1536,
            system="You are an assistant that helps create embeddings. Please provide a 1536-dimensional embedding for the given text. Respond only with the embedding values separated by commas, enclosed in square brackets.",
            messages=[
                {"role": "user", "content": f"Generate a 1536-dimensional embedding for this text: {query_text}"}
            ]
        )
        embedding = response.content[0].text
        embedding_list = [float(x) for x in embedding.strip('[]').split(',')]
        if len(embedding_list) != 1536:
            raise ValueError(f"Expected 1536 dimensions, but got {len(embedding_list)}")
        return embedding_list
    except Exception as e:
        print(f"Warning: Error in embedding query: {str(e)}")
        hash_object = hashlib.md5(query_text.encode())
        hash_hex = hash_object.hexdigest()
        seed_value = int(hash_hex, 16) % (2**32 - 1)
        np.random.seed(seed_value)
        return np.random.rand(1536).tolist()

def query_pinecone(query, top_k=5):
    query_embedding = embed_query(query)
    result = index.query(vector=query_embedding, top_k=top_k, include_metadata=True)
    return result['matches']

def ask_claude_about_documents(query_text):
    documents = query_pinecone(query_text)
    combined_documents = "\n\n".join([doc['metadata']['text'] for doc in documents])
    full_prompt = f"המסמכים שקיבלתי הם: \n\n{combined_documents}\n\nשאלה: {query_text}"
    try:
        response = claude.messages.create(
            model="claude-3-opus-20240229",
            max_tokens=1024,
            system="You are an assistant that helps answer questions in Hebrew based on provided documents.",
            messages=[
                {"role": "user", "content": full_prompt}
            ]
        )
        return response.content[0].text
    except Exception as e:
        raise Exception(f"Error querying Claude: {str(e)}")

def display_hebrew(text):
    reshaped_text = arabic_reshaper.reshape(text)
    bidi_text = get_display(reshaped_text)
    return bidi_text

def chat_function(message, history):
    response = ask_claude_about_documents(message)
    displayed_response = display_hebrew(response)
    return displayed_response

# Gradio Interface
iface = gr.ChatInterface(
    chat_function,
    title="Hebrew RAG Chat with Claude and Pinecone",
    description="Ask questions in Hebrew about documents stored in Pinecone.",
    examples=[
        "על פי מה נקבע שכר המינימום בישראל?",
        "מהם החוקים העיקריים הנוגעים לזכויות עובדים בישראל?",
        "כיצד מחושבים דמי אבטלה בישראל?"
    ],
    theme=gr.themes.Soft()
)

iface.launch()