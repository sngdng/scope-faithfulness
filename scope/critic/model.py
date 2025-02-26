import torch.nn as nn
from transformers import AutoModel


class SimpleClassifierModel(nn.Module):
    """
    Encode text with pretrained model, concatenate the embedding with LLMs scores and use MLP.
    """

    def __init__(self, checkpoint, freeze=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint
        )  # , device_map="auto", load_in_8bit=True)
        if freeze:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.dense1 = nn.Linear(
            self.model.config.hidden_size, self.model.config.hidden_size
        )
        self.act_func = nn.ReLU()
        self.dense2 = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        hidden_state = self.dropout(hidden_state)
        hidden1 = self.dense1(hidden_state)
        hidden1 = self.act_func(hidden1)
        hidden1 = self.dropout(hidden1)
        result = self.dense2(hidden1)
        return result


class SimpleClassifierModelWithBN(nn.Module):
    """
    Encode text with pretrained model, concatenate the embedding with LLMs scores and use MLP.
    """

    def __init__(self, checkpoint, freeze=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint
        )  # , device_map="auto", load_in_8bit=True)
        if freeze:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.dense1 = nn.Linear(
            self.model.config.hidden_size, self.model.config.hidden_size
        )
        self.act_func = nn.ReLU()
        self.batch_norm = nn.BatchNorm1d(self.model.config.hidden_size)
        self.dense2 = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        hidden_state = self.dropout(hidden_state)
        hidden1 = self.dense1(hidden_state)
        hidden1 = self.act_func(hidden1)
        hidden1 = self.batch_norm(hidden1)
        hidden1 = self.dropout(hidden1)
        result = self.dense2(hidden1)
        return result


class SimpleClassifierModelWithBNSELU(nn.Module):
    """
    Encode text with pretrained model, concatenate the embedding with LLMs scores and use MLP.
    """

    def __init__(self, checkpoint, freeze=False):
        super().__init__()
        self.model = AutoModel.from_pretrained(
            checkpoint
        )  # , device_map="auto", load_in_8bit=True)
        if freeze:
            for param in self.model.base_model.parameters():
                param.requires_grad = False
        self.dropout = nn.Dropout(0.1)
        self.dense1 = nn.Linear(
            self.model.config.hidden_size, self.model.config.hidden_size
        )
        self.act_func = nn.SELU()
        self.batch_norm = nn.BatchNorm1d(self.model.config.hidden_size)
        self.dense2 = nn.Linear(self.model.config.hidden_size, 1)

    def forward(self, input_ids, attention_mask):
        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = outputs.last_hidden_state[:, 0]
        hidden_state = self.dropout(hidden_state)
        hidden1 = self.dense1(hidden_state)
        hidden1 = self.act_func(hidden1)
        hidden1 = self.batch_norm(hidden1)
        hidden1 = self.dropout(hidden1)
        result = self.dense2(hidden1)
        return result
