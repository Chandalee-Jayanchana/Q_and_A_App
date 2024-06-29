import warnings
import time

warnings.filterwarnings("ignore")

from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter  # Update this if there's a new equivalent
from langchain_community.vectorstores import Pinecone
from langchain_openai import OpenAIEmbeddings, ChatOpenAI


# from langchain.document_loaders import  PyPDFDirectoryLoader
from langchain.text_splitter import CharacterTextSplitter
# can use  recursive text splitter also. 
import os
import pinecone
# from langchain.vectorstores import Pinecone 
# from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chains import RetrievalQA
# from langchain.chat_models import ChatOpenAI
import streamlit as st
from dotenv import load_dotenv

# load_dotenv()


# PINECONE_API_KEY_3 = os.getenv('PINECONE_API_KEY_3')
# PINECONE_ENV_3 = os.getenv('PINECONE_ENV_3')
# peopa1 = os.getenv('peopa1')


# OPENAI_API_KEY_5 = os.getenv('OPENAI_API_KEY_5')

# os.environ['OPENAI_API_KEY_5'] = OPENAI_API_KEY_5


load_dotenv()


peopa1 = os.getenv('peopa1')
# PINECONE_ENV_3 = os.getenv('PINECONE_ENV_3')
OPENAI_API_KEY_5 = os.getenv('OPENAI_API_KEY_5')




from pinecone import Pinecone, ServerlessSpec
def ensure_pinecone_index():
    index_name = "test"
    dimension = 1536
    metric = "cosine"
    cloud = "aws"
    region = "us-east-1"

    pc = Pinecone(api_key=peopa1)
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=ServerlessSpec(cloud=cloud, region=region)
        )
    index = pc.Index(name=index_name)
    return index



# Create or ensure the index is ready
index = ensure_pinecone_index()


# def doc_preprocessing():
#     loader = PyPDFDirectoryLoader(    #load pdf documents
#         path='/content/drive/MyDrive/ST-Testing/Diabetes', 
#         glob='**/*.pdf'    # only the PDFs
#         # show_progress=True
#     )
#     docs = loader.load()
#     text_splitter = CharacterTextSplitter(  #splitting loaded documents
#         chunk_size=1000, 
#         chunk_overlap=0
#     )
#     docs_split = text_splitter.split_documents(docs)
#     return docs_split


def doc_preprocessing():
    loader1 = PyPDFDirectoryLoader(path='Diabetes', glob='**/*.pdf')
    loader2 = PyPDFDirectoryLoader(path='.', glob='Heart_Diseases.pdf')
    
    docs1 = loader1.load()
    docs2 = loader2.load()
    
    all_docs = docs1 + docs2
    
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    docs_split = text_splitter.split_documents(all_docs)
    return docs_split


from langchain_community.vectorstores.pinecone import Pinecone
# import pinecone
def embedding_db():
    os.environ["PINECONE_API_KEY"] = peopa1
    # embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY_5)
    Pinecone.from_existing_index(  index_name='test', embedding=embeddings)
    docs_split = doc_preprocessing()
    # index.upsert(docs_split)
    doc_db = Pinecone.from_documents(documents=docs_split, embedding=embeddings, index_name='test')
    return doc_db


# embeddings =OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY_5)
# doc_split=doc_preprocessing()
# doc_db = Pinecone.from_documents(documents=doc_split, embedding=embeddings, index_name='test')
# embedding_db()


llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key=OPENAI_API_KEY_5)


from langchain.chains import LLMChain
from langchain.chains.conversational_retrieval.prompts import CONDENSE_QUESTION_PROMPT
from langchain.chains.question_answering import load_qa_chain




from langchain.chains.qa_with_sources import load_qa_with_sources_chain

from langchain.chains.question_answering import load_qa_chain


chain = load_qa_chain(llm)


os.environ["PINECONE_API_KEY"] = peopa1
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY_5)

doc_split=doc_preprocessing()
doc_db = Pinecone.from_documents(documents=doc_split, embedding=embeddings, index_name='test')

def qa_vector_store(chain, question):
  inputs = {
    "input_documents": doc_db.similarity_search(question, k=3),
    "question": question
    }
  response = chain(inputs, return_only_outputs=True)
  outputs = response["output_text"]
  return outputs





#chatgpt animation

def stream_data(data):
    for word in data.split(" "):
        yield word + " "
        time.sleep(0.02)



def main():
    
    st.markdown("""
        <style>
        h1 {
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            z-index: 5000000;
            padding: 10px;
            box-shadow: 0px 1px 50px rgba(0, 0, 0, 0.1);
            font-family: 'Helvetica';
            font-size: 40px;
            color: navy;
            background-color: white;
            text-align: center; /* Align the text to the left */
             /* Use viewport width unit for responsive padding */
            }
        h3 {
            position: fixed;
            left: 0;
            top: 100px; /* Adjust this value based on the actual height of the h1 element */
            width: 100%;
            z-index: 2;
            font-family: 'Arial';
            font-size: 20px;
            color: Black;
            background-color: white;
            padding: 10px 100px; /* Top and bottom padding of 10px, left and right padding of 40px */
            margin: 0;
            text-align: left;
            
            padding-left: 780px;
            }
        
        
        /* Adjust the top margin of the main content container to prevent overlap */
        .main .block-container {
        padding-top: 120px;  /* Adjust this value to the combined height of h1 and h3 */
        }
        
        
          /* Chat message styles */
        .stChatMessage {
            background-color: #f0f0f0;
        }
        .stChatMessage .st-ck-user {
            background-color: #e1f5fe;  /* Light blue background for user messages */
        }
        .stChatMessage .st-ck-assistant {
            background-color: #dcedc8;  /* Light green background for assistant messages */
        }
        .stChatMessage p {
            font-size: 16px;  /* Adjust the font size of chat messages */
            color: black;  /* Font color for chat messages */
        }
         
        </style>
        """, unsafe_allow_html=True)
    
    st.title('Time Series Repository :clock1: :chart_with_upwards_trend:')
    
    st.markdown(' ### Embedded with OpenAI  :keyboard:', unsafe_allow_html=True)

    # # Style for questions and answers
    # st.markdown("""
     
    #     """, unsafe_allow_html=True)

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

 

# Display the conversation history- (this show history once)
    for idx, (q, a) in enumerate(st.session_state.conversation_history):
        with st.chat_message("user"):
                 st.write(f"Q{idx+1}: " ,q)
        with st.chat_message("assistant"):
                 st.write(f"A{idx+1}: " ,a)



    prompt = st.chat_input("Ask Your Question ...", key="new_question")

        # Process user input to get the answer
    if prompt:
        with st.spinner('Generating Answer...'):
            answer = qa_vector_store(chain, prompt)
            st.session_state.conversation_history.append((prompt, answer))

        # Display the latest question and answer
        latest_q, latest_a = st.session_state.conversation_history[-1]
        with st.chat_message("You asked:"):
            st.write(f"Q{len(st.session_state.conversation_history)}: {latest_q}")
        with st.chat_message("Assistant replied:"):
            st.write(f"A{len(st.session_state.conversation_history)}: {latest_a}")


if __name__ == "__main__":
    main()
