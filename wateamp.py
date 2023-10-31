from flask import Flask, request, jsonify
import openai
from twilio.twiml.messaging_response import MessagingResponse

app = Flask(__name__)

openai.api_key = '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'

def generate_answer(question):
    model_engine = "text-davinci-002"
    prompt = (f"Q:{question}\n"
              "A:")
    
    response = openai.Completion.create(
        engine = model_engine,
        prompt = prompt,
        max_tokens = 1024,
        n = 1,
        stop = None,
        temperature = 0.7,
    )

    answer = response.choices[0].text.strip()
    return answer

@app.route('/chatgpt', methods = ['POST'])
def chatgpt():
    incoming_que = request.values.get('Body', '').lower()
    print(incoming_que)

    answer = generate_answer(incoming_que)
    print(answer)
    bot_resp = MessagingResponse()
    msg = bot_resp.message()
    msg.body(answer)

    return str(bot_resp)

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port = 5000)