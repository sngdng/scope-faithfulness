import re
import string
from dataclasses import dataclass
from typing import ClassVar, Dict, Optional, Tuple

from scope.data.utils import (
    camel_case_to_natural_text,
    get_highlighted_subtable,
    process_subtable,
)


class BaseTask:
    def format_input(self, x):
        raise NotImplementedError

    def prepare_batch(self, batch):
        return [self.format_input(x) for x in batch]

    def add_chat_tokens_and_response(self, prompt, chat_tokens, with_input):
        if chat_tokens is not None:
            prompt = f"{chat_tokens[0]} {prompt.strip()} {chat_tokens[1]}"
            if with_input:
                prompt = f"{prompt} {self.template['response_start']}{self.template['chat_start']}"
        else:
            if with_input:
                prompt = f"{prompt}\n\n{self.template['response_start']}"
        return prompt


@dataclass
class GenerationTask(BaseTask):
    with_input: bool = True

    def format_input(self, x, with_input=None):
        if with_input is None:
            with_input = self.with_input

        if with_input:
            return x["prompt"]
        else:
            return x["prompt_no_input"]


@dataclass(frozen=True)
class TottoTask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nThis is a highlighted cells description task. The goal of this task is to generate the language description given table cells.\n\n### Input:\n$linearized_input\n\n### Question:\nPlease generate a one-sentence description to describe the given highlighted table cells."
        ),
        "prompt_no_input": string.Template(""),
        "response_start": "### Response:\n\n",
        "chat_start": "Here is a natural language description of the highlighted cells in the table:\n\n",
    }
    with_input: bool = True
    chat_tokens: Optional[Tuple] = None

    def format_input(self, x, with_input=None):
        linearized_input = x["linearized_input"]

        if with_input is None:
            with_input = self.with_input
        if with_input:
            prompt = self.template["prompt_input"].safe_substitute(
                linearized_input=linearized_input
            )
        else:
            prompt = self.template["prompt_no_input"].template
        prompt = self.add_chat_tokens_and_response(prompt, self.chat_tokens, with_input)

        return prompt

    def format_eval(self, x):
        table = x["table"]
        table_page_title = x["table_page_title"]
        table_section_title = x["table_section_title"]
        cell_indices = x["highlighted_cells"]
        subtable = get_highlighted_subtable(
            table=table, cell_indices=cell_indices, with_heuristic_headers=True
        )

        parent_table, source = process_subtable(
            subtable=subtable,
            table_page_title=table_page_title,
            table_section_title=table_section_title,
        )

        return source, parent_table


@dataclass(frozen=True)
class WebNLGTask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nThis is a graph-to-text task. The goal of this task is to generate the language description given knowledge graphs.\n\n### Input:\n$linearized_input\n\n### Question:\nPlease generate one natural language description to describe the given knowledge graph."
        ),
        "prompt_no_input": string.Template(""),
        "response_start": "### Response:\n\n",
        "chat_start": "Here is a natural language description of the knowledge graph:\n\n",
    }
    with_input: bool = True
    chat_tokens: Optional[Tuple] = None

    def format_input(self, x, with_input=None):
        knowledge_graph = x["input"]
        triple_str = ""
        for triple in knowledge_graph:
            e1, r, e2 = triple.split(" | ")
            e1, r, e2 = e1.strip(), r.strip(), e2.strip()
            triple_str += f"<H> {e1} <R> {r} <T> {e2} "
        triple_str = triple_str.strip()

        if with_input is None:
            with_input = self.with_input
        if with_input:
            prompt = self.template["prompt_input"].safe_substitute(
                linearized_input=triple_str
            )
        else:
            prompt = self.template["prompt_no_input"].template
        prompt = self.add_chat_tokens_and_response(prompt, self.chat_tokens, with_input)
        return prompt

    def preprocess_triple(self, triple):
        e1, r, e2 = triple.split(" | ")
        e1, e2 = e1.strip().replace("_", " "), e2.strip().replace("_", " ")
        r = camel_case_to_natural_text(r.strip())

        return e1, r, e2

    def format_eval(self, x):
        source = []
        parent_table = []

        for triple in x["input"]:
            e1, r, e2 = self.preprocess_triple(triple)

            source.append(f"{e1} | {r} | {e2}")
            parent_table.append((r, e1 + " " + e2))

        return source, parent_table


@dataclass(frozen=True)
class DARTTask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nThis is a triple-to-text task. The goal of this task is to generate the language description given a list of triples.\n\n### Input:\n$linearized_input\n\n### Question:\nPlease generate one natural language description to describe the given list of triples."
        ),
        "prompt_no_input": string.Template(""),
        "response_start": "### Response:\n\n",
        "chat_start": "Here is a natural language description of the list of triples:\n\n",
    }
    with_input: bool = True
    chat_tokens: Optional[Tuple] = None

    def format_input(self, x, with_input=None):
        tripleset = x["tripleset"]
        triples = []
        for triple in tripleset:
            e1, r, e2 = triple[0], triple[1], triple[2]
            e1, r, e2 = e1.strip(), r.strip(), e2.strip()
            triples.append(f"{e1} : {r} : {e2}")
        triple_str = " | ".join(triples)

        if with_input is None:
            with_input = self.with_input
        if with_input:
            prompt = self.template["prompt_input"].safe_substitute(
                linearized_input=triple_str
            )
        else:
            prompt = self.template["prompt_no_input"].template
        prompt = self.add_chat_tokens_and_response(prompt, self.chat_tokens, with_input)
        return prompt

    def preprocess_triple(self, triple):
        e1, r, e2 = triple[0], triple[1], triple[2]
        e1, e2 = e1.strip().replace("_", " "), e2.strip().replace("_", " ")
        r = r.strip().replace("[TITLE]", "title").strip().lower().replace("_", " ")

        return e1, r, e2

    def format_eval(self, x):
        source = []
        parent_table = []

        entity_found = False
        entity = ""
        for triple in x["tripleset"]:
            e1, r, e2 = self.preprocess_triple(triple)
            if not entity_found:
                if e1.isnumeric() or "[" in e1 or "]" in e1:
                    if not e2.isnumeric() and not "[" in e2 and not "]" in e2:
                        entity = e2
                        entity_found = True
                else:
                    entity = e1
                    entity_found = True

            source.append(f"{e1} | {r} | {e2}")
            parent_table.append((r, e1 + " " + e2))

        return source, parent_table, entity


@dataclass(frozen=True)
class E2ETask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nThis is a meaning representation description task. The goal of this task is to generate the language description given meaning representations.\n\n### Input:\n$linearized_input\n\n### Question:\nPlease generate one natural language description to describe the given meaning representation."
        ),
        "prompt_no_input": string.Template(""),
        "response_start": "### Response:\n\n",
        "chat_start": "Here is a natural language description of the meaning representation:\n\n",
    }
    with_input: bool = True
    chat_tokens: Optional[Tuple] = None

    def format_input(self, x, with_input=None):
        mr = x["meaning_representation"]

        if with_input is None:
            with_input = self.with_input
        if with_input:
            prompt = self.template["prompt_input"].safe_substitute(linearized_input=mr)
        else:
            prompt = self.template["prompt_no_input"].template
        prompt = self.add_chat_tokens_and_response(prompt, self.chat_tokens, with_input)
        return prompt

    def format_eval(self, x):
        source = []
        parent_table = []

        list_facts = x["meaning_representation"].split(", ")
        for i, fact in enumerate(list_facts):
            mrc_format = "(.*)\[(.*)\]"
            try:
                attr, value = re.compile(mrc_format).match(fact).groups()
                if i == 0:
                    if attr == "name":
                        entity1 = value
                        source.append(f"{value} | name | {value}")
                        parent_table.append(("name", value))
                        continue
                    else:
                        entity1 = "a restaurant"
            except:
                print("Error formatting fact: ", fact)
                continue

            e1 = entity1.strip()
            rel = attr.strip()
            e2 = value.strip()
            source.append(f"{e1} | {rel} | {e2}")
            parent_table.append((rel, e2))

        return source, parent_table


@dataclass(frozen=True)
class E2ETaskNoName(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nThis is a meaning representation description task. The goal of this task is to generate the language description given meaning representations.\n\n### Input:\n$linearized_input\n\n### Question:\nPlease generate one natural language description to describe the given meaning representation.\n\n### Response:"
        ),
    }
    with_input: bool = True

    def format_input(self, x, with_input=None):
        mr = x["meaning_representation"]

        list_facts = mr.split(", ")
        list_facts_no_name = []
        for i, fact in enumerate(list_facts):
            mrc_format = "(.*)\[(.*)\]"
            try:
                attr, value = re.compile(mrc_format).match(fact).groups()
                if attr == "name":
                    continue
                else:
                    list_facts_no_name.append((attr, value))
            except:
                print("Error formatting fact: ", fact)
                continue
        mr_no_name = ", ".join(
            [f"{attr}[{value}]" for attr, value in list_facts_no_name]
        )

        return self.template["prompt_input"].safe_substitute(
            linearized_input=mr_no_name
        )


@dataclass
class FeTaQATask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nThis is a free-form table question answering task. The goal for this task is to answer the given question based on the given table.\n\n### Input:\n$input_context\n\n### Question:\n$question"
        ),
        "prompt_no_input": string.Template(""),
        "response_start": "### Response:\n\n",
        "chat_start": "Here is an answer to the question:\n\n",
    }
    with_input: bool = True
    chat_tokens: Optional[Tuple] = None

    def linearize_table(self, sample):
        # Extract the table array from the sample
        table_array = sample["table_array"]

        # Initialize an empty list to hold the linearized rows
        linearized_rows = []

        # Iterate over each row in the table array
        for row in table_array:
            # Join the cells in the row with a vertical bar and add to the list
            linearized_rows.append(" | ".join(["", *row, ""]))

        # Join all linearized rows with the [SEP] token
        linearized_table = "[TAB]" + "[SEP]".join(linearized_rows)

        # Add the linearized_table as a new key to the sample dictionary

        return linearized_table

    def format_eval(self, x):
        table = x["table_array"]
        table_page_title = x["table_page_title"]
        source = []
        parent_table = []

        set_key_value = set()
        for row_id, cell_id in x["highlighted_cell_ids"]:
            key = table[0][cell_id]
            row = table[row_id]
            cell = row[cell_id]
            set_key_value.add((key, cell))

        for key, cell in set_key_value:
            source.append(f"{table_page_title} | {key} | {cell}")
            parent_table.append((key, cell))

        return source, parent_table

    def format_input(self, x, with_input=None):
        linearized_table = self.linearize_table(x)
        page_title = x["table_page_title"]
        question = x["question"]

        page_title_str = (
            f"[TLE] The Wikipedia page title of this table is {page_title}."
        )

        input_context = f"{page_title_str} {linearized_table}"

        if with_input is None:
            with_input = self.with_input
        if with_input:
            prompt = self.template["prompt_input"].safe_substitute(
                input_context=input_context, question=question
            )
        else:
            prompt = self.template["prompt_no_input"].safe_substitute(
                question=question,
            )
        prompt = self.add_chat_tokens_and_response(prompt, self.chat_tokens, with_input)
        return prompt


@dataclass
class HighlightedFeTaQATask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nThis is a free-form table question answering task."
            " The goal for this task is to answer the given question based on the given table and the highlighted cells.\n\n### Input:\n$input_context\nThe highlighted cells of the table are: $highlighted_cells\n\n### Question:\n$question"
        ),
        "prompt_no_input": string.Template(""),
        "response_start": "### Response:\n\n",
        "chat_start": "Here is an answer to the question:\n\n",
    }
    with_input: bool = True
    chat_tokens: Optional[Tuple] = None

    def linearize_table(self, sample):
        # Extract the table array from the sample
        table_array = sample["table_array"]

        # Initialize an empty list to hold the linearized rows
        linearized_rows = []

        # Iterate over each row in the table array
        for row in table_array:
            # Join the cells in the row with a vertical bar and add to the list
            linearized_rows.append(" | ".join(["", *row, ""]))

        # Join all linearized rows with the [SEP] token
        linearized_table = "[TAB]" + "[SEP]".join(linearized_rows)

        # Process the highlighted cells
        highlighted_cells = []
        highlighted_cells_ids = sample["highlighted_cell_ids"]
        for row_id, cell_id in highlighted_cells_ids:
            row = table_array[row_id]
            cell = row[cell_id]
            highlighted_cells.append(f"[{cell}]")

        return linearized_table, " ".join(highlighted_cells)

    def format_eval(self, x):
        table = x["table_array"]
        table_page_title = x["table_page_title"]
        source = []
        parent_table = []

        set_key_value = set()
        for row_id, cell_id in x["highlighted_cell_ids"]:
            key = table[0][cell_id]
            row = table[row_id]
            cell = row[cell_id]
            set_key_value.add((key, cell))

        for key, cell in set_key_value:
            source.append(f"{table_page_title} | {key} | {cell}")
            parent_table.append((key, cell))

        return source, parent_table

    def format_input(self, x, with_input=None):
        linearized_table, highlighted_cells = self.linearize_table(x)
        page_title = x["table_page_title"]
        question = x["question"]

        page_title_str = (
            f"[TLE] The Wikipedia page title of this table is {page_title}."
        )

        input_context = f"{page_title_str} {linearized_table}"

        if with_input is None:
            with_input = self.with_input
        if with_input:
            prompt = self.template["prompt_input"].safe_substitute(
                input_context=input_context,
                highlighted_cells=highlighted_cells,
                question=question,
            )
        else:
            prompt = self.template["prompt_no_input"].safe_substitute(
                question=question,
            )
        prompt = self.add_chat_tokens_and_response(prompt, self.chat_tokens, with_input)
        return prompt


@dataclass(frozen=True)
class TableInstructTask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n$instruction\n\n### Input:\n$input_seg\n\n### Question:\n$question\n\n### Response:"
        ),
        "prompt_no_input": string.Template(
            "Below is an instruction that describes a task. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\n$instruction\n\n### Response:"
        ),
    }

    def format_input(self, x, with_input):
        instruction = x["instruction"]
        input_seg = x["input_seg"]
        question = x["question"]

        if with_input:
            return self.template["prompt_input"].safe_substitute(
                instruction=instruction, input_seg=input_seg, question=question
            )
        else:
            return self.template["prompt_no_input"].safe_substitute(
                instruction=instruction
            )


@dataclass(frozen=True)
class XSumTask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWrite a one-sentence summary of the input article.\n\n### Input:\n$article\n\n### Question:\nPlease generate a one-sentence summary of the given article."
        ),
        "prompt_no_input": string.Template(""),
        "response_start": "### Response:\n\n",
        "chat_start": "Here is a one-sentence summary of the given article:\n\n",
    }
    with_input: bool = True
    chat_tokens: Optional[Tuple] = None

    def format_input(self, x, with_input=None):

        if with_input is None:
            with_input = self.with_input
        if with_input:
            prompt = self.template["prompt_input"].safe_substitute(
                article=x["document"]
            )
        else:
            prompt = self.template["prompt_no_input"].template
        prompt = self.add_chat_tokens_and_response(prompt, self.chat_tokens, with_input)
        return prompt


@dataclass(frozen=True)
class SamsumTask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWrite a concise brief of the discussion that should be rather short, extract important pieces of information, include names of interlocutors, be written in the third person.\n\n### Input:\n$dialogue\n\n### Question:\nPlease write a short summary of the given discussion."
        ),
        "prompt_no_input": string.Template(""),
        "response_start": "### Response:\n\n",
        "chat_start": "Here is a short sentence summary of the given discussion:\n\n",
    }
    with_input: bool = True
    chat_tokens: Optional[Tuple] = None

    def format_input(self, x, with_input=None):

        if with_input is None:
            with_input = self.with_input
        if with_input:
            prompt = self.template["prompt_input"].safe_substitute(
                dialogue=x["dialogue"]
            )
        else:
            prompt = self.template["prompt_no_input"].template
        prompt = self.add_chat_tokens_and_response(prompt, self.chat_tokens, with_input)
        return prompt


@dataclass(frozen=True)
class CNNDMTask(BaseTask):
    template: ClassVar[Dict[str, string.Template]] = {
        "prompt_input": string.Template(
            "Below is an instruction that describes a task, paired with an input that provides further context. "
            "Write a response that appropriately completes the request.\n\n"
            "### Instruction:\nWrite a summary of the input article.\n\n### Input:\n$article\n\n### Question:\nPlease generate a summary of the given article."
        ),
        "prompt_no_input": string.Template(""),
        "response_start": "### Response:\n\n",
        "chat_start": "Here is a summary of the given article:\n\n",
    }
    with_input: bool = True
    chat_tokens: Optional[Tuple] = None

    def format_input(self, x, with_input=None):

        if with_input is None:
            with_input = self.with_input
        if with_input:
            prompt = self.template["prompt_input"].safe_substitute(article=x["article"])
        else:
            prompt = self.template["prompt_no_input"].template
        prompt = self.add_chat_tokens_and_response(prompt, self.chat_tokens, with_input)
        return prompt
