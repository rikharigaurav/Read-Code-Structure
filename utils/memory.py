from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatOpenAI, ChatMistralAI
from langchain.prompts import ChatPromptTemplate
from transformers import AutoTokenizer
import os

# llm = ChatOpenAI(model="gpt-4", temperature=0)
llm = ChatMistralAI(api_key=os.getenv('MISTRAL_API_KEY'))
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

def process_fragment(fragment, current_index, total_fragments, prompt, memory):
    """
    Processes a file fragment using the LLM with memory.

    Args:
        fragment (str): The current file fragment.
        current_index (int): The index of the current fragment.
        total_fragments (int): Total number of fragments.
        prompt (str): Information about the file type and key contents.
        memory (ConversationBufferMemory): LangChain memory instance.

    Returns:
        dict: A dictionary containing the summary and incomplete context points.
    """

    document_status = "[START OF DOCUMENT]" if current_index == 0 else "[CONTINUED]"
    document_end = "[END OF DOCUMENT]" if current_index == total_fragments - 1 else "[TO BE CONTINUED]"

    # Construct the input message
    input_message = f"""
    Document Fragment {current_index + 1}/{total_fragments}:
    {document_status}

    prompt: {prompt}

    fragment:
    {fragment}

    {document_end}

    Instructions:
    - Retain context from previous fragments
    - Prepare for potential continuation
    - Do not generate final conclusion until last fragment
    """

    # Add structured output to llm
    structured_llm = llm.with_structured_output()

    # Run the LLM
    response = structured_llm.predict(input_message, memory=memory)

    # Save the context in memory
    memory.chat_history.append((input_message, response))

    return {
        "summary": response,  # Assumes response contains the fragment summary
        "incomplete_context": "Incomplete context identified by the LLM."  # Placeholder for extracted points
    }

# Example of processing multiple fragments
def process_file_in_chunks(file_content, prompt):
    max_tokens = 4000
    """
    Splits the file into fragments and processes each fragment iteratively.

    Args:
        file_content (str): The full file content.
        fragment_size (int): The size of each chunk.
        prompt (str): Information about the file type and key contents.

    Returns:
        list: A list of summaries and context information for each fragment.
    """
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    tokens = tokenizer.encode(file_content)
    fragments = []
    
    for i in range(0, len(tokens), max_tokens):
        # Ensure fragment doesn't break mid-sentence
        fragment_tokens = tokens[i:i+max_tokens]
        fragment = tokenizer.decode(fragment_tokens)

    # Trim to last complete sentence
        if i + max_tokens < len(tokens) and '.' in fragment:
            fragment = fragment.rsplit('.', 1)[0] + '.'

        fragments.append(fragment)

    total_fragments = len(fragments)

    results = []
    for current_index, fragment in enumerate(fragments):
        result = process_fragment(fragment, current_index, total_fragments, prompt, memory)
        results.append(result)

    return results


def process_llm_calls(file_path, prompt):
    with open(file_path, 'r', encoding='utf-8') as file:
            content = file.read()

    results = process_file_in_chunks(content, prompt)
    summary = ""
    # Output results
    for idx, result in enumerate(results):
        print(f"Fragment {idx + 1} Summary")
        # print(result["summary"])
        summary += f"\n result[\"summary\"] \n"
        # print("Incomplete Context:", result["incomplete_context"])
        # print("\n---\n")

    namespace = "kaf"
    return summary, test_framework, test_reference_dict