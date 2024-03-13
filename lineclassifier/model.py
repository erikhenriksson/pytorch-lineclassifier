from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel
from torch import nn
import torch

from transformers import XLMRobertaConfig


class DotDict(dict):
    def __getattr__(self, key):
        if key in self:
            return self[key]


class CustomXLMRobertaConfig(XLMRobertaConfig):
    def __init__(self, sep_id=0, **kwargs):
        super().__init__(**kwargs)
        self.sep_id = sep_id


class LineClassificationHead(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        classifier_dropout = (
            config.classifier_dropout
            if config.classifier_dropout is not None
            else config.hidden_dropout_prob
        )

        self.dropout = nn.Dropout(classifier_dropout)
        self.out_proj = nn.Linear(config.hidden_size, config.num_labels)

    def forward(self, features, **kwargs):
        x = features
        x = self.dropout(x)
        x = self.dense(x)
        x = torch.tanh(x)
        x = self.dropout(x)
        x = self.out_proj(x)
        return x


class XLMRobertaForLineClassification(XLMRobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_labels = config.num_labels
        self.config = config
        self.sep_id = config.sep_id
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = LineClassificationHead(config)

        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        labels=None,
        inputs_embeds=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]

        batch_size, seq_length, hidden_size = sequence_output.size()

        focus_mask = input_ids == self.sep_id

        # Assuming SEP tokens enclose the text of interest, find the indexes of SEP tokens
        focus_indices = focus_mask.nonzero(as_tuple=True)
        pooled_output = torch.zeros(
            batch_size,
            hidden_size,
            dtype=sequence_output.dtype,
            device=sequence_output.device,
        )

        for i in range(batch_size):
            # Extract the sequence between the two SEP tokens for each example
            seq_focus_indices = focus_indices[1][focus_indices[0] == i]
            if (
                len(seq_focus_indices) > 1
            ):  # Ensure there are at least two SEP tokens to define a range
                start_idx = seq_focus_indices[0] + 1
                end_idx = seq_focus_indices[1]
                focused_seq = sequence_output[i, start_idx:end_idx, :]
                # Mean pooling over the focused sequence
                pooled_output[i] = focused_seq.mean(dim=0)

        logits = self.classifier(pooled_output)

        return DotDict({"logits": logits})
