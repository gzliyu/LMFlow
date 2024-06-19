from datasets import load_from_disk
from vllm import LLM, SamplingParams
import json
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', type=str)

    args = parser.parse_args()
    
    model_name = args.model_name
    
    dataset = load_from_disk(f'/home/nlpintern1/liyu/dataset/stack-exchange-paired-dummy/data/test')[:250]
    questions_all = dataset['question']
    response_j = dataset['response_j']
    response_k = dataset['response_k']
    
    prompts = [f"Question: {question}\n\nAnswer: " for question in questions_all]
    
    sampling_params = SamplingParams(max_tokens=2048)
    trained_llm = LLM(
        model=f"/home/nlpintern1/liyu/models/{model_name}",
    )
    trained_model_results = trained_llm.generate(prompts, sampling_params)
    del trained_llm

    output_root = f"/home/nlpintern1/liyu/models/{model_name}"

    results = [
        {
            "question": question,
            "trained": trained_output.outputs[0].text,
            "answer_chosen": j,
            "answer_rejected": k
        }
        for question, trained_output, j, k in zip(
            questions_all,
            trained_model_results,
            response_j,
            response_k
        )
    ]

    with open(output_root + f"/{model_name}.json", "w", encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False)
