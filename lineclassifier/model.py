from transformers import XLMRobertaModel, XLMRobertaPreTrainedModel
from torch import nn
import torch

from transformers import XLMRobertaConfig


class CustomXLMRobertaConfig(XLMRobertaConfig):
    def __init__(self, max_lines=100, **kwargs):
        super().__init__(**kwargs)
        self.max_lines = max_lines


# Copied from transformers.models.roberta.modeling_roberta.RobertaClassificationHead with Roberta->XLMRoberta
class LineClassificationHead(nn.Module):
    """Head for sentence-level classification tasks."""

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
        x = features  # These are the <s> tokens for all lines
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
        self.max_lines = config.max_lines
        self.roberta = XLMRobertaModel(config, add_pooling_layer=False)
        self.classifier = LineClassificationHead(config)

        # Initialize weights and apply final processing
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
        special_token_id = self.config.bos_token_id

        special_token_mask = input_ids == special_token_id

        line_features = torch.zeros(
            (batch_size, self.max_lines, hidden_size), device=sequence_output.device
        )

        for i in range(batch_size):
            line_indices = special_token_mask[i].nonzero(as_tuple=True)[0]

            num_lines = line_indices.size(0)
            lines_pooled = []
            print("example")
            for j in range(num_lines):  # Exclude the last </s> token
                start_idx = line_indices[j].item() + 1  # Skip the <s> token
                if j + 1 == len(line_indices):
                    end_idx = seq_length - 1

                else:

                    end_idx = line_indices[j + 1].item()
                print(start_idx, end_idx)
                line_repr = sequence_output[i, start_idx:end_idx, :]

                line_repr = torch.mean(line_repr, dim=0, keepdim=True)
                lines_pooled.append(line_repr)

            # lines = sequence_output[i, line_indices, :]
            lines = torch.cat(lines_pooled, dim=0)

            if num_lines < self.max_lines:
                padding = torch.zeros(
                    (self.max_lines - num_lines, hidden_size),
                    device=sequence_output.device,
                )
                lines = torch.cat((lines, padding), dim=0)
            line_features[i] = lines

        logits = self.classifier(
            line_features.view(-1, hidden_size)
        )  # Reshape for classification head

        return logits
