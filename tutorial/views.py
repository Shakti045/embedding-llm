from rest_framework import status
from rest_framework.decorators import api_view
from rest_framework.response import Response

from langchain.document_loaders.csv_loader import CSVLoader
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.chains import LLMChain
from decouple import config

secret_key = config('OPEN_AI_API_KEY')
loader = CSVLoader(file_path="tutorial/salaries.csv")
documents = loader.load()

embeddings = OpenAIEmbeddings(api_key=secret_key)
db = FAISS.from_documents(documents, embeddings)


def retrieve_info(query):
    similar_response = db.similarity_search(query, k=3)

    page_contents_array = [doc.page_content for doc in similar_response]

    # print(page_contents_array)

    return page_contents_array


# 3. Setup LLMChain & prompts
llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo-16k-0613")

template = """
You are a world class business development representative. 
I will share a prospect's message with you and you will give me the best answer that 
I should send to this prospect based on past best practies, 
and you will follow ALL of the rules below:

1/ Response should be very similar or even identical to the past best practies, 
in terms of length, ton of voice, logical arguments and other details

2/ If the best practice are irrelevant, then try to mimic the style of the best practice to prospect's message

Below is a message I received from the prospect:
{message}

Here is a list of best practies of how we normally respond to prospect in similar scenarios:
{best_practice}

Please write the best response that I should send to this prospect:
"""

prompt = PromptTemplate(
    input_variables=["message", "best_practice"],
    template=template
)

chain = LLMChain(llm=llm, prompt=prompt)


# 4. Retrieval augmented generation
def generate_response(message):
    best_practice = retrieve_info(message)
    response = chain.run(message=message, best_practice=best_practice)
    return response





@api_view(['POST'])
def postquery(request):
    try:
        querry = request.data['querry'];
        if(querry == ''):
          return Response({'Success':False,'Message':'Empty query'}, status=status.HTTP_400_BAD_REQUEST)
        responce = generate_response(querry)
        return Response({'Success':True,'Message':'Query responce fetched','Responce':responce},status=status.HTTP_200_OK)
    except Exception as e:
        return Response({'Success':False,'Message':str(e)}, status=status.HTTP_400_BAD_REQUEST)