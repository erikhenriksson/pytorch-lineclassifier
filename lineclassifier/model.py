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

        pooled_output = torch.zeros(
            batch_size,
            hidden_size,
            dtype=sequence_output.dtype,
            device=sequence_output.device,
        )

        for i in range(batch_size):
            # Find the indices of SEP tokens for the current sequence
            sep_indices = (input_ids[i] == self.sep_id).nonzero(as_tuple=True)[0]
            if sep_indices.size(0) >= 2:  # Ensure there are at least two SEP tokens
                # Extract the segment between the first two SEP tokens
                start_idx, end_idx = sep_indices[0].item() + 1, sep_indices[1].item()
                target_segment = sequence_output[i, start_idx:end_idx]
                # Apply mean pooling over the target segment
                pooled_output[i] = target_segment.mean(dim=0)

        logits = self.classifier(pooled_output)

        return DotDict({"logits": logits})
