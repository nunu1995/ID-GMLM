import openai
import os
import numpy as np
from sklearn.decomposition import PCA


def get_criteria_weights_from_gpt4(criteria_descriptions, data_descriptions):
    criteria_summary = "\n".join(criteria_descriptions)
    data_summary = "\n".join(data_descriptions)
    messages = [
        {"role": "system", "content": "You are an intelligent assistant. According to the given criteria descriptions and data, please evaluate the potential importance of each criterion."},
        {"role": "user", "content": f"Criteria descriptions:\n{criteria_summary}\n\nData (each row represents a different alternative with values for each criterion followed by a label indicating the alternative's overall quality):\n{data_summary}"}
    ]

    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=messages,
        temperature=0.5,
        max_tokens=550,
        n=1,
        stop=None
    )

    last_response = response['choices'][0]['message']['content']
    return last_response.strip()


def get_gpt_emb(prompt):
    embedding = openai.Embedding.create(
        input=prompt,
        model="text-embedding-ada-002"
    )["data"][0]["embedding"]

    return np.array(embedding)


def dm_weight():
    os.environ["http_proxy"] = " "
    os.environ["https_proxy"] = " "

    openai.api_key = 'your_openai_api_key'

    criteria_descriptions = [
        "The cost of implementation, considering the required investment and operational costs.",
        "The potential for revenue generation, including sales increase and new market opportunities.",
        "The impact on customer satisfaction, focusing on improving customer experience and engagement.",
        "The ease of integration with existing systems, considering compatibility and required changes.",
        "The scalability, focusing on the ability to grow and handle increased workloads.",
        "The environmental impact, considering sustainability and eco-friendliness.",
        "The regulatory compliance, ensuring adherence to laws and regulations."
    ]

    data_descriptions = [
        'a1 0.6764 1.0414 0.2638 0.1216 1.0200 1.0872 1.0675 2',
        'a2 0.2240 0.1867 1.0978 1.0791 0.5821 1.0105 -0.0294 1',
        'a3 0.1444 0.2771 0.3729 1.1951 1.5634 1.1230 1.2379 2',
        'a4 0.6547 1.6751 0.2443 0.9336 0.6045 -0.0328 0.0065 1',
        'a5 0.6536 0.5645 -0.1384 0.9429 0.0653 0.0864 0.0742 1'
    ]

    weight_descriptions = get_criteria_weights_from_gpt4(criteria_descriptions, data_descriptions)
    criteria_weights = get_gpt_emb(weight_descriptions)

    pca = PCA(n_components=1)
    weights = pca.fit_transform(criteria_weights.reshape(1, -1)).flatten()

    weights = np.abs(weights)
    weights /= np.sum(weights)
    return weights