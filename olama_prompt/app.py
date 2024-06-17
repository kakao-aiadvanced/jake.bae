from flask import Flask, request
import requests
import json

app = Flask(__name__)


@app.route('/ollama/chat/history/teacher')
def history_teacher():
    response_json = query_teacher("history")
    return response_json["message"]


@app.route('/ollama/chat/math/teacher')
def math_teacher():
    response_json = query_teacher("math")
    return response_json["message"]


def query_teacher(field: str):
    data = {
        "model": "llama3",
        "stream": False,
        "prompt": "you are" + field + "teacher",
        "messages": [
            {
                "role": "user",
                "content": request.args["query"]
            }
        ]
    }
    headers = {'Content-Type': 'application/json; chearset=utf-8'}
    res = requests.post('http://localhost:11434/api/chat', data=json.dumps(data), headers=headers)
    print(res)
    response_json = json.loads(res.content)
    return response_json


@app.route('/ollama/chat/ordering', methods=['POST'])
def ordering():
    params = json.loads(request.get_data(), encoding='utf-8')
    data = {
        "model": "llama3",
        "stream": False,
        "prompt": " you are math teacher"
                  "few shot:"
                  "66, 19, 134, 32, 45 => 19, 32, 66, 45, 134"
                  "5,4,3,2,1 => 1,2,3,4,5",
        "messages": [
            {
                "role": "user",
                "content": "Sort in ascending order: numbers " + str(params['numbers']) + ""
                                                                                          "The answer format is [1, 3, 4, 5, 6 ...]"
            }
        ]
    }

    print(str(params['numbers']))
    headers = {'Content-Type': 'application/json; chearset=utf-8'}
    res = requests.post('http://localhost:11434/api/chat', data=json.dumps(data), headers=headers)
    print(res)

    response_json = json.loads(res.content)
    return response_json["message"]


if __name__ == '__main__':
    app.run()
