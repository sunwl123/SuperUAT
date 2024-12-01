import numpy as np
import torch
from torch.nn import functional as F
from transformers import AutoTokenizer, RobertaForMaskedLM


class HookCloser:
    def __init__(self, model_wrapper):
        self.model_wrapper = model_wrapper

    def __call__(self, module, input_, output_):
        self.model_wrapper.curr_embedding = output_
        output_.retain_grad()


class RobertaClassifier(object):
    def __init__(self, device, loss_type, max_length=128, batch_size=8):
        self.loss_type = loss_type

        self.model = RobertaForMaskedLM.from_pretrained('/home/swl/roberta-large')
        self.tokenizer = AutoTokenizer.from_pretrained("/home/swl/roberta-large")
        self.tokenizer.add_tokens(["<trigger>"])

        self.embedding_layer = self.model.roberta.embeddings.word_embeddings
        self.curr_embedding = None
        self.hook = self.embedding_layer.register_forward_hook(HookCloser(self))
        self.embedding = self.embedding_layer.weight.detach().cpu().numpy()

        self.word2id = dict()
        for i in range(self.tokenizer.vocab_size):
            self.word2id[self.tokenizer.convert_ids_to_tokens(i)] = i

        self.trigger = []
        self.max_length = max_length
        self.device = device
        self.model.to(device)
        self.batch_size = batch_size

    def set_trigger(self, trigger):
        """Set the trigger.

        :param trigger: a list of tokens as the trigger. Each token must be a word in the vocabulary.
        """
        self.trigger = trigger

    def get_pred(self, input_):
        """Get prediction of the masked position in each sentence.

        :param input_: a list of sentences, each contains one <mask> token and one <trigger> token.
        :return: a list of integers meaning the predicted token id for the masked position.
        """
        return self.get_prob(input_).argmax(axis=1)

    def get_prob(self, input_):
        """Get prediction of the probability of words for the masked position in each sentence.

        :param input_: a list of sentences, each contains one <mask> token and one <trigger> token.
        :return: a matrix of n * v, where n is the number of sentences, and v is the number of words.
        """
        return self.get_grad([self.tokenizer.tokenize(sent) for sent in input_], [0] * len(input_))[0]

    def get_grad(self, tokenized_input, labels):
        """Get prediction of the probability of words for the masked position in each sentence, and the gradient of
        loss with respect to the trigger tokens.

        :param tokenized_input: a list of tokenized sentences. Each contains one <mask> token and one <trigger> token.
        :param labels: a list of integers showing the target word id.
        :return: A tuple of two matrices. The first n * v matrix shows the predicted probability distribution on the
            masked position. The second n * l * d matrix shows the gradient of the specified loss with respect to the
            word embedding of the trigger tokens.
        """
        v = self.predict(tokenized_input, labels)
        return v[0], v[1]

    def predict(self, input_, labels=None, return_loss=False):
        """Implementation of get_grad with optional loss in return."""
        sen_list = []          # a copy of input with <trigger> being replaced by actual trigger tokens
        mask_pos = []          # the index of <mask> in each sentence
        trigger_offset = []    # the index of the first trigger token

        sen_list_no_trigger = []    # a copy of input with <trigger> removed
        mask_pos_no_trigger = []    # the index of <mask> in each sentence (it can be different from mask_pos.)

        for text in input_:
            text = [tok.strip() for tok in text]
            trigger_index = text.index("<trigger>")
            text_t = text[:trigger_index] + self.trigger + text[trigger_index + 1:]
            mask_pos_tmp = text_t.index("<mask>")
            sen_list.append(text_t)
            # a CLS token will be added to the begining
            mask_pos.append(mask_pos_tmp + 1)
            trigger_offset.append(trigger_index + 1)

            text_t_no_trigger = text[:trigger_index] + text[trigger_index + 1:]
            sen_list_no_trigger.append(text_t_no_trigger)
            mask_pos_no_trigger.append(text_t_no_trigger.index("<mask>") + 1)

        sent_lens = [len(sen) for sen in sen_list]
        batch_len = max(sent_lens) + 2

        attentions = np.array([[1] * (len(sen) + 2) + [0] * (batch_len - 2 - len(sen))
                               for sen in sen_list], dtype='int64')
        sen_list = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list]
        sen_list_no_trigger = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list_no_trigger]
        input_ids = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len - 2 - len(sen)) for sen in sen_list], dtype='int64')

        input_ids_no_trigger = np.array([
            [self.tokenizer.cls_token_id] + sen + [self.tokenizer.sep_token_id]
            + [self.tokenizer.pad_token_id] * (batch_len - 2 - len(sen) - len(self.trigger))
            for sen in sen_list_no_trigger], dtype='int64')

        result = []
        result_grad = []

        if labels is None:
            labels = [0] * len(sen_list)
        labels = torch.LongTensor(labels).to(self.device)

        overall_loss = 0
        for i in range((len(sen_list) + self.batch_size - 1) // self.batch_size):
            curr_input_ids = input_ids[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask = attentions[i * self.batch_size: (i + 1) * self.batch_size]
            curr_label = labels[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos_no_trigger = mask_pos_no_trigger[i * self.batch_size: (i + 1) * self.batch_size]
            curr_mask_pos = mask_pos[i * self.batch_size: (i + 1) * self.batch_size]
            curr_trigger_offset = trigger_offset[i * self.batch_size: (i + 1) * self.batch_size]
            curr_input_ids_no_trigger = input_ids_no_trigger[i * self.batch_size: (i + 1) * self.batch_size]

            # ===== compute output embed without trigger if loss is embdis.
            if self.loss_type == "embdis":
                xs = torch.from_numpy(curr_input_ids_no_trigger).long().to(self.device)
                masks = torch.from_numpy(curr_mask).long().to(self.device)
                outputs = self.model.roberta(input_ids=xs, attention_mask=masks[:, len(self.trigger):],
                                             output_hidden_states=True)
                hidden = [outputs[0][idx, item, :] for idx, item in enumerate(curr_mask_pos_no_trigger)]
                hidden = torch.stack(hidden, dim=0)
                emb_no_trigger = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden))
            else:
                emb_no_trigger = None
            # =======================================

            xs = torch.from_numpy(curr_input_ids).long().to(self.device)
            masks = torch.from_numpy(curr_mask).long().to(self.device)
            outputs = self.model.roberta(input_ids=xs, attention_mask=masks, output_hidden_states=True)
            hidden = [outputs[0][idx, item, :] for idx, item in enumerate(curr_mask_pos)]
            hidden = torch.stack(hidden, dim=0)
            emb = self.model.lm_head.layer_norm(self.model.lm_head.dense(hidden))
            logits = self.model.lm_head.decoder(emb)
            prob = torch.softmax(logits, dim=1)

            if self.loss_type == "embdis":
                loss = emb - emb_no_trigger
                loss = -torch.sqrt((loss * loss).sum(dim=1)).sum()
                loss.backward()
            elif self.loss_type in ["prob", "prob_sq"]:
                loss = torch.gather(prob, dim=1, index=curr_label.unsqueeze(1)).squeeze(1)
                if self.loss_type == "prob_sq":
                    loss = loss * loss
                loss = loss.sum()
                loss.backward()
            elif self.loss_type in ["mprob", "mprob_sq"]:
                loss = prob.max(dim=1)[0]
                if self.loss_type == "mprob_sq":
                    loss = loss * loss
                loss = loss.sum()
                loss.backward()
            elif self.loss_type in ["ce"]:
                loss = F.cross_entropy(logits, curr_label, reduction="none")
                loss = -loss.sum()
                loss.backward()
            else:
                assert 0

            overall_loss += loss.detach().cpu().numpy()

            result.append(prob.detach().cpu())
            grads_tmp = self.curr_embedding.grad.clone().cpu().numpy()
            grads_tmp = [item[curr_trigger_offset[idx]:curr_trigger_offset[idx] + len(self.trigger)]
                         for idx, item in enumerate(grads_tmp)]

            result_grad.append(grads_tmp)
            self.curr_embedding.grad.zero_()
            self.curr_embedding = None
            del hidden
            del emb
            del emb_no_trigger
            del prob
            del loss

        result = np.concatenate(result, axis=0)
        result_grad = np.concatenate(result_grad, axis=0)
        if return_loss:
            return result, result_grad, overall_loss
        else:
            return result, result_grad
        
    def predict_label(self, input_, labels=None, return_loss=False):
        """Implementation of get_grad with optional loss in return."""
        sen_list = []          # a copy of input with <trigger> being replaced by actual trigger tokens
        mask_pos = []          # the index of <mask> in each sentence
        trigger_offset = []    # the index of the first trigger token

        sen_list_no_trigger = []    # a copy of input with <trigger> removed
        mask_pos_no_trigger = []    # the index of <mask> in each sentence (it can be different from mask_pos.)

        for text in input_:
            text = [tok.strip() for tok in text]
            trigger_index = text.index("<trigger>")
            text_t = text[:trigger_index] + self.trigger + text[trigger_index + 1:]
            mask_pos_tmp = text_t.index("<mask>")
            sen_list.append(text_t)
            # a CLS token will be added to the begining
            mask_pos.append(mask_pos_tmp + 1)
            trigger_offset.append(trigger_index + 1)

            text_t_no_trigger = text[:trigger_index] + text[trigger_index + 1:]
            sen_list_no_trigger.append(text_t_no_trigger)
            mask_pos_no_trigger.append(text_t_no_trigger.index("<mask>") + 1)

        sen_list = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list]
        sen_list_no_trigger = [self.tokenizer.convert_tokens_to_ids(sen) for sen in sen_list_no_trigger]
        
        if labels is None:
            labels = [0] * len(sen_list)
        labels = torch.LongTensor(labels).to(self.device)
        for i in range((len(sen_list) + self.batch_size - 1) // self.batch_size):
            curr_label = labels[i * self.batch_size: (i + 1) * self.batch_size]

        return curr_label
