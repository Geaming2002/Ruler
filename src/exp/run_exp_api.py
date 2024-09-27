import argparse
import openai
import os
from tqdm import tqdm
from utils import load_jsonl, save_jsonl
from dotenv import load_dotenv


load_dotenv()


def main(args):
    client = openai.OpenAI(
        api_key=args.key, base_url=os.getenv("OPENAI_BASE_URL")
    )
    def get_completion_openai(
        prompt: str,
        model: str,
    ) -> str:
        """
        Generate a completion using the OpenAI API.

        Args:
            prompt (str): The user's prompt or query.
            model (str, optional): The name of the OpenAI model to use for generating the completion.
                Defaults to "gpt-4-turbo".
        """
        response = client.chat.completions.create(
            model=model,
            top_p=1,
            max_tokens=2048,
            messages=[
                {"role": "user", "content": prompt},
            ],
        )
        return response.choices[0].message.content

    # raw data load
    df = load_jsonl(args.dataset_path)
    # load tokenizer and llm
    for idx in tqdm(range(len(df))):
        instruction = df[idx]["Instruction"]
        targetlength = df[idx]["TargetLength"] if "TargetLength" in df[idx] else ""
        if targetlength != "":
            targetlength = targetlength.replace(">", "more than ")
            question = f"{instruction}\nThe response should have a word count of {targetlength} words."
            df[idx]["prompt"] = question
            flag = False
            while not flag:
                try:
                    output = get_completion_openai(question, args.model)
                    flag = True
                except Exception as e:
                    print(e)
            df[idx]["output"] = output
        if idx % 10 == 0:
            save_jsonl(args.output_path, df)
    # save to output_path
    save_jsonl(args.output_path, df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default=None)
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--output_path", type=str, default=None)
    parser.add_argument("--key", type=str, default=None)
    args = parser.parse_args()
    main(args)
