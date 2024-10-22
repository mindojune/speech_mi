from typing import List, Optional, Tuple, Union

import torch
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import nn
from torch.nn import CrossEntropyLoss

from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.utils import logging
from transformers import LlamaModel, LlamaForCausalLM


logger = logging.get_logger(__name__)


class AudioLlamaForCausalLM(LlamaForCausalLM):
    _tied_weights_keys = ["lm_head.weight"]

    def __init__(self, config):
        super().__init__(config)
        self.model = LlamaModel(config)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Initialize weights and apply final processing
        self.post_init()

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        r"""
        Args:
            labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
                Labels for computing the causal language modeling loss. Indices
                should either be in `[0, ..., config.vocab_size]` or -100
                (see `input_ids` docstring). Tokens with indices set to `-100`
                are ignored (masked), the loss is only computed for the tokens
                with labels in `[0, ..., config.vocab_size]`.

        Returns:
            CausalLMOutputWithPast
        ```"""

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            batch_size, max_response_length = labels.shape
            seq_len = inputs_embeds.shape[1]
            
            # Create a labels_padded tensor filled with -100 (ignore_index)
            labels_padded = torch.full((batch_size, seq_len), -100, dtype=torch.long, device=labels.device)
            
            # Compute the actual lengths of the responses in the batch
            response_lengths = (labels != -100).sum(dim=1, keepdim=True)  # Shape: [batch_size, 1]
            
            # Create a sequence range tensor
            seq_range = torch.arange(seq_len, device=labels.device).unsqueeze(0).expand(batch_size, seq_len)
            
            # Create a mask for positions where labels should be placed
            mask = seq_range >= (seq_len - response_lengths)
            
            # Compute positions in labels to copy from
            labels_positions = (seq_range - (seq_len - response_lengths)).clamp(min=0, max=max_response_length - 1)
            
            # Copy labels into the padded labels tensor using the mask
            labels_padded[mask] = labels.gather(1, labels_positions)[mask]
            
            # Shift logits and labels for next-token prediction
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels_padded[..., 1:].contiguous()
            
            # Flatten the tensors
            loss_fct = CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1)
            )
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        """
        # TODO: Currently assumes a batch size of 1. Change to incorporate
        # sizes > 1.
        if labels is not None:
            loss = 0.0
            for sample_logits, sample_labels in zip(logits, labels):
                # # Shift so that tokens < n predict n
                # shift_logits = logits[..., :-1, :].contiguous()
                # shift_labels = labels[..., 1:].contiguous()

                sample_logits = sample_logits.unsqueeze(0)
                sample_labels = sample_labels.unsqueeze(0).to(self.device)

                # Shift so that tokens < n predict n, but only for tokens
                # corresponding to the response portion of the LLM.
                response_len = sample_labels.shape[1]
                shift_logits = sample_logits[..., -response_len:-1, :].contiguous()

                # Labels are only provided for the response portion of the LLM in
                # the first place.
                shift_labels = sample_labels[..., 1:].contiguous()

                # Flatten the tokens
                loss_fct = CrossEntropyLoss()
                shift_logits = shift_logits.view(-1, self.config.vocab_size)
                shift_labels = shift_labels.view(-1)

                # Enable model parallelism
                shift_labels = shift_labels.to(shift_logits.device)
                loss += loss_fct(shift_logits, shift_labels)

            # Manually perform mean reduction for cross entropy loss.
            loss /= logits.shape[0]

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        """
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, inputs_embeds=None, **kwargs
    ):
        if past_key_values:
            input_ids = input_ids[:, -1:]

        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None and position_ids is None:
            # create position_ids on the fly for batch generation
            position_ids = attention_mask.long().cumsum(-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values:
                position_ids = position_ids[:, -1].unsqueeze(-1)

        # if `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update(
            {
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": kwargs.get("use_cache"),
                "attention_mask": attention_mask,
            }
        )
        return model_inputs