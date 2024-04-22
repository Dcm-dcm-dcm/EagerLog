import dashscope
from dashscope import Generation
from http import HTTPStatus
import json
import os

limlt_try = 10

dashscope.api_key = os.environ['QWEN_KEY']

def get_qwen(prompt, call_num=1):
    llm_parameters = {
        "temperature": 1.0,
        "top_p": 1e-10,
        "top_k": None,
    }

    response = Generation.call(
        model='qwen-plus',  # qwen-plus
        prompt=prompt,
        temperature=llm_parameters["temperature"],
        top_p=llm_parameters["top_p"],
        top_k=llm_parameters["top_k"],
    )
    if response.status_code == HTTPStatus.OK:
        llm_response = json.dumps(response.output, indent=4, ensure_ascii=False)
        res_in_dict = json.loads(llm_response)
        return res_in_dict['text']
    else:
        if call_num >limlt_try:
            print('qwen api time out')
            return ''
        return get_qwen(prompt, call_num+1)





if __name__ == "__main__":
    prompt = 'hello'
    print(get_qwen(prompt))
