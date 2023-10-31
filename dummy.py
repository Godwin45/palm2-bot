from langchain.vectorstores import FAISS
from langchain.llms import GooglePalm
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
import os
from dotenv import load_dotenv

app = Flask(__name__)

load_dotenv()
llm = GooglePalm(google_api_key=os.environ["GOOGLE_API_KEY"], temperature=0.1)
instructor_embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large")
vectordb_file_path = "faiss_index"

def generate_answer(incoming_que):
  vectordb = FAISS.load_local(vectordb_file_path, instructor_embeddings)
  retriever = vectordb.as_retriever(score_threshold=0.7)
  prompt_template = """Given the following context and a question, generate an answer based on this context only.
  In the answer try to provide as much text as possible from "response" section in the source document context without making much changes.
  If the answer is not found in the context, kindly state "I don't know." Don't try to make up an answer.
  CONTEXT: {context}
  QUESTION: {question}"""
  PROMPT = PromptTemplate(
    template=prompt_template, input_variables=["context", "question"]
  )

  answer = RetrievalQA.from_chain_type(llm=llm,
                                        chain_type="stuff",
                                        retriever=retriever,
                                        input_key="query",
                                        return_source_documents=True,
                                        chain_type_kwargs={"prompt": PROMPT})

  # Extract the answer from the RetrievalQA object
  return answer



# Define a route to handle incoming requests
@app.route('/chatgpt', methods=['POST'])
def chatgpt():
    incoming_que = request.values.get('Body', '').lower()
    print("Question: ", incoming_que)
    # Generate the answer using LangChain
    answer = generate_answer(incoming_que)
    print("BOT Answer: ", answer)
    bot_resp = MessagingResponse()
    msg = bot_resp.message()
    msg.body(answer)
    return bot_resp

# Run the Flask app
if __name__ == "__main__":
    app.run(host='0.0.0.0', debug=False, port=5000)
