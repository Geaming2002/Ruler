from utils.config import MetaLengthToken

# default templates


class ModelTemplate:
    STOP_TOKENS = {
        "mistral": ["</s>"],
        "glm": ["<|endoftext|>", "<|user|>", "<|observation|>"],
        "gemma": ["<eos>", "<end_of_turn>"],
        "llama": ["<|end_of_text|>", "<|eot_id|>"],
        "internlm": ["</s>", "<|im_end|>"],
        "deepseek": ["<｜end▁of▁sentence｜>"],
        "yi": ["<|im_end|>", "<|endoftext|>"],
        "qwen": ["<|im_end|>", "<|endoftext|>"],
    }

    @staticmethod
    def get_stop_tokens(model_name: str):
        for key in ModelTemplate.STOP_TOKENS:
            if key in model_name.lower():
                return ModelTemplate.STOP_TOKENS[key]
        raise KeyError

    @staticmethod
    def apply_template_for_generation(instruction, targetlength, tokenizer):
        if targetlength != "":
            targetlength = targetlength.replace(">", "more than ")
            question = f"{instruction}\nThe response should have a word count of {targetlength} words."
        else:
            question = instruction
        messages = [
            {"role": "user", "content": question},
        ]
        prompt = tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        return prompt


# MLT templates


class Llama3_MLT_Template:
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": mlt+output},
    ]

    <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{mlt}{output}<|eot_id|>
    """

    BOS_TOKEN: str = "<|begin_of_text|>"
    EOS_TOKEN: str = "<|end_of_text|>"
    STOP_TOKENS: list[str] = ["<|end_of_text|>", "<|eot_id|>"] + [
        MLT[0] for MLT in MetaLengthToken
    ]

    @staticmethod
    def apply_template(instruction, mlt, output):
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{mlt}{output}<|eot_id|>"
        return prompt

    @staticmethod
    def apply_template_for_generation(instruction, targetlength=""):
        mlt = f"[MLT:{targetlength}]" if targetlength != "" else ""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{mlt}"
        return prompt


class Qwen_MLT_Template:
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": mlt+output},
    ]

    <|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{mlt}{output}<|im_end|>
    """

    BOS_TOKEN: str = None
    EOS_TOKEN: str = "<|im_end|>"
    STOP_TOKENS: list[str] = [
        "<|im_end|>",
        "<|endoftext|>",
    ] + [MLT[0] for MLT in MetaLengthToken]

    @staticmethod
    def apply_template(instruction, mlt, output):
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{mlt}{output}<|im_end|>"
        return prompt

    @staticmethod
    def apply_template_for_generation(instruction, targetlength=""):
        mlt = f"[MLT:{targetlength}]" if targetlength != "" else ""
        prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{mlt}"
        return prompt


class Yi_MLT_Template:
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": mlt+output},
    ]

    <|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{mlt}{output}<|im_end|>
    """

    BOS_TOKEN: str = "<|startoftext|>"
    EOS_TOKEN: str = "<|im_end|>"
    STOP_TOKENS: list[str] = [
        "<|im_end|>",
        "<|endoftext|>",
    ] + [MLT[0] for MLT in MetaLengthToken]

    @staticmethod
    def apply_template(instruction, mlt, output):
        prompt = f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{mlt}{output}<|im_end|>"
        return prompt

    @staticmethod
    def apply_template_for_generation(instruction, targetlength=""):
        mlt = f"[MLT:{targetlength}]" if targetlength != "" else ""
        prompt = (
            f"<|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{mlt}"
        )
        return prompt


class internlm_MLT_Template:
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": mlt+output},
    ]

    <s><|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{mlt}{output}<|im_end|>
    """

    BOS_TOKEN: str = "<s>"
    EOS_TOKEN: str = "</s>"
    STOP_TOKENS: list[str] = ["</s>", "<|im_end|>"] + [
        MLT[0] for MLT in MetaLengthToken
    ]

    @staticmethod
    def apply_template(instruction, mlt, output):
        prompt = f"<s><|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{mlt}{output}<|im_end|>"
        return prompt

    @staticmethod
    def apply_template_for_generation(instruction, targetlength=""):
        mlt = f"[MLT:{targetlength}]" if targetlength != "" else ""
        prompt = f"<s><|im_start|>user\n{instruction}<|im_end|>\n<|im_start|>assistant\n{mlt}"
        return prompt


class glm_MLT_Template:
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": mlt+output},
    ]

    [gMASK]<sop><|user|>\ninstruction<|assistant|>\nmlt+output
    """

    BOS_TOKEN: str = "[gMASK]<sop>"
    EOS_TOKEN: str = "<|endoftext|>"
    STOP_TOKENS: list[str] = [
        "<|endoftext|>",
        "<|user|>",
        "<|observation|>",
    ] + [MLT[0] for MLT in MetaLengthToken]

    @staticmethod
    def apply_template(instruction, mlt, output):
        prompt = f"[gMASK]<sop><|user|>\n{instruction}<|assistant|>\n{mlt}{output}"
        return prompt

    @staticmethod
    def apply_template_for_generation(instruction, targetlength=""):
        mlt = f"[MLT:{targetlength}]" if targetlength != "" else ""
        prompt = f"[gMASK]<sop><|user|>\n{instruction}<|assistant|>\n{mlt}"
        return prompt


class deepseek_MLT_Template:
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": mlt+output},
    ]

    <｜begin▁of▁sentence｜>User: INSTRUCTION\n\nAssistant: MLT+OUTPUT<｜end▁of▁sentence｜>
    """

    BOS_TOKEN: str = "<｜begin▁of▁sentence｜>"
    EOS_TOKEN: str = "<｜end▁of▁sentence｜>"
    STOP_TOKENS: list[str] = [
        "<｜end▁of▁sentence｜>",
    ] + [MLT[0] for MLT in MetaLengthToken]

    @staticmethod
    def apply_template(instruction, mlt, output):
        prompt = f"<｜begin▁of▁sentence｜>User: {instruction}\n\nAssistant: {mlt}{output}<｜end▁of▁sentence｜>"
        return prompt

    @staticmethod
    def apply_template_for_generation(instruction, targetlength=""):
        mlt = f"[MLT:{targetlength}]" if targetlength != "" else ""
        prompt = f"<｜begin▁of▁sentence｜>User: {instruction}\n\nAssistant: {mlt}"
        return prompt


class gemma_MLT_Template:
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": mlt+output},
    ]

    <bos><start_of_turn>user\nINSTRUCTION<end_of_turn>\n<start_of_turn>model\nMLT+OUTPUT<end_of_turn>
    """

    BOS_TOKEN: str = "<bos>"
    EOS_TOKEN: str = "<eos>"
    STOP_TOKENS: list[str] = [
        "<eos>",
        "<end_of_turn>",
    ] + [MLT[0] for MLT in MetaLengthToken]

    @staticmethod
    def apply_template(instruction, mlt, output):
        prompt = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{mlt}{output}<end_of_turn>"
        return prompt

    @staticmethod
    def apply_template_for_generation(instruction, targetlength=""):
        mlt = f"[MLT:{targetlength}]" if targetlength != "" else ""
        prompt = f"<bos><start_of_turn>user\n{instruction}<end_of_turn>\n<start_of_turn>model\n{mlt}"
        return prompt


class mistral_MLT_Template:
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": mlt+output},
    ]

    <s>[INST] INSTRUCTION [/INST]MLT+OUTPUT</s>
    """

    BOS_TOKEN: str = "<s>"
    EOS_TOKEN: str = "</s>"
    STOP_TOKENS: list[str] = [
        "</s>",
    ] + [MLT[0] for MLT in MetaLengthToken]

    @staticmethod
    def apply_template(instruction, mlt, output):
        prompt = f"<s>[INST] {instruction} [/INST]{mlt}{output}</s>"
        return prompt

    @staticmethod
    def apply_template_for_generation(instruction, targetlength=""):
        mlt = f"[MLT:{targetlength}]" if targetlength != "" else ""
        prompt = f"<s>[INST] {instruction} [/INST]{mlt}"
        return prompt


class custom_Template:
    """
    messages = [
        {"role": "user", "content": instruction},
        {"role": "assistant", "content": mlt+output},
    ]

    <|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{mlt}{output}<|eot_id|>
    """

    BOS_TOKEN: str = "<|begin_of_text|>"
    EOS_TOKEN: str = "<|end_of_text|>"
    STOP_TOKENS: list[str] = ["<|end_of_text|>", "<|eot_id|>"] + [
        MLT[0] for MLT in MetaLengthToken
    ]
    SPECIAL_TOKENS: list[str] = [
        "<|begin_of_text|>",
        "<|end_of_text|>",
        "<|start_header_id|>",
        "<|end_header_id|>",
        "<|eot_id|>",
    ]

    @staticmethod
    def apply_template(instruction, mlt, output):
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{mlt}{output}<|eot_id|>"
        return prompt

    @staticmethod
    def apply_template_for_instruction(instruction):
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt

    @staticmethod
    def apply_template_for_generation(instruction, targetlength=""):
        mlt = f"[MLT:{targetlength}]" if targetlength != "" else ""
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{instruction}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{mlt}"
        return prompt
    def apply_template_for_generation_vanilla(instruction, targetlength=""):
        targetlength = targetlength.replace(">", "more than ")
        question = f"{instruction}\nThe response should have a word count of {targetlength} words."
        prompt = f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{question}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n"
        return prompt


TemplatesMapping = {
    "default": ModelTemplate,
    "llama3": Llama3_MLT_Template,
    "qwen": Qwen_MLT_Template,
    "yi": Yi_MLT_Template,
    "internlm": internlm_MLT_Template,
    "glm": glm_MLT_Template,
    "deepseek": deepseek_MLT_Template,
    "gemma": gemma_MLT_Template,
    "mistral": mistral_MLT_Template,
    "custom": custom_Template,
}
