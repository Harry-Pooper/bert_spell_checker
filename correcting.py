# coding: utf-8

import numpy as np
import tensorflow as tf
from keras_bert import load_trained_model_from_checkpoint

from base import tokenization
from base.modeling import BertConfig
from base.modeling import BertModel
from base.modeling import create_initializer
from base.modeling import get_activation
from base.modeling import layer_norm

MASK_ID = 103


class ModelFacade(object):
    def __init__(self):
        self.folder = 'rubert_model'
        self.config_path = self.folder + '/bert_config.json'
        self.checkpoint_path = self.folder + '/bert_model.ckpt'
        self.vocab_path = self.folder + '/vocab.txt'
        tf.compat.v1.disable_eager_execution()
        self.X = tf.compat.v1.placeholder(tf.int32, [None, None])

        config = BertConfig.from_json_file(self.config_path)
        modeling = BertModel(config=config, is_training=False, input_ids=self.X)
        output_layer = modeling.get_sequence_output()
        embeddings = modeling.get_embedding_table()

        with tf.compat.v1.variable_scope('cls/predictions'):
            with tf.compat.v1.variable_scope('transform'):
                input_tensor = tf.compat.v1.layers.dense(
                    output_layer,
                    units=config.hidden_size,
                    activation=get_activation(config.hidden_act),
                    kernel_initializer=create_initializer(
                        config.initializer_range
                    ),
                )
                input_tensor = layer_norm(input_tensor)
            output_bias = tf.compat.v1.get_variable(
                'output_bias',
                shape=[config.vocab_size],
                initializer=tf.zeros_initializer
            )
            logits = tf.matmul(input_tensor, embeddings, transpose_b=True)
            self.output_bias = output_bias
            self.logits = tf.nn.bias_add(logits, output_bias)


class Corrector(object):
    def __init__(self, possibleValues, russian=True):
        self.points = [',', ':', '-', ';', '—']
        self.possibleValues = possibleValues
        if russian:
            self.folder = 'rubert_model'
        else:
            self.folder = 'bert_model_en'
        self.config_path = self.folder + '/bert_config.json'
        self.checkpoint_path = self.folder + '/bert_model.ckpt'
        self.vocab_path = self.folder + '/vocab.txt'
        tf.compat.v1.disable_v2_behavior()
        tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
        self.tokenizer = tokenization.FullTokenizer(vocab_file=self.vocab_path, do_lower_case=False)
        self.model = load_trained_model_from_checkpoint(self.config_path, self.checkpoint_path, training=True)
        self.model.summary()

    def normalize(self, entryText):
        entryText = entryText.lower()
        while '  ' in entryText:
            entryText = entryText.replace('  ', ' ')
        for i in range(len(self.points)):
            entryText = entryText.replace(self.points[i], '')
        while '.' in entryText:
            entryText = entryText.replace('.', '[SEP]')
        return entryText.strip()

    # Исправление грамматических и орфографических ошибок
    def correctGrammar(self, entryText):
        words = entryText.split(' ')

        for i in range(len(words)):
            left = ' '.join(words[:i])
            right = ' '.join(words[i + 1:])
            masked_sentence = left + (' [MASK] ' if left else '[MASK] ') + right

            possible_words = self.possibleValues.edit_candidates(words[i], fast=False)
            possible_sentences = [masked_sentence.replace('[MASK]', word) for word in possible_words]
            scores = [self.get_score_for_idx(sentence, i + 1) for sentence in possible_sentences]
            scores_zip = list(zip(possible_sentences, scores))
            prob_scores = np.array(scores) / np.sum(scores)
            prob_list = list(zip(possible_sentences, prob_scores))
            prob_list.sort(key=lambda x: x[1])
            most_possible = prob_list[len(prob_list) - 1][0]
            words[i] = most_possible.split(' ')[i]

        return ' '.join(words)

    def get_score_for_idx(self, mask, idx):
        tokens = self.convert_to_tokens(mask)
        length = self.calculate_length_for_token(tokens, idx)

        input_ids = [self.tokens_to_ids_with_mask(tokens, idx, length)]
        mask_input = [[0] * 512 for i in range(len(input_ids))]
        seg_input = [[0] * 512 for i in range(len(input_ids))]
        for i in range(len(input_ids)):
            input_ids[i] = input_ids[i] + [0] * (512 - len(input_ids[i]))
            for j in range(len(input_ids[i])):
                if input_ids[i][j] == MASK_ID:
                    mask_input[i][j] = 1

        input_ids_arr = np.asarray(input_ids)
        mask_input_arr = np.asarray(mask_input)
        seg_input_arr = np.asarray(seg_input)
        predicts = self.model.predict([input_ids_arr, mask_input_arr, seg_input_arr])
        tokens_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        token_probabilities = list(zip(tokens_ids, [predicts[0][0, i + 1, x] for i, x in enumerate(tokens_ids)]))
        return np.prod([predicts[0][0, i, x] for i, x in enumerate(tokens_ids)])

    def calculate_length_for_token(self, tokens, m_idx):
        length = 1
        i = m_idx + 1
        while i < len(tokens):
            if tokens[i].startswith('##'):
                length += 1
                i += 1
            else:
                return length
        return length

    def tokens_to_ids_with_mask(self, tokens, m_idx, length):
        mask_tokens = tokens[:]
        i = 0
        while i < length:
            mask_tokens[m_idx + i] = '[MASK]'
            i += 1
        token_ids = self.tokenizer.convert_tokens_to_ids(mask_tokens)
        return token_ids

    def convert_to_tokens(self, entryText):
        sentences = entryText.split('[SEP]')
        tokens = ['[CLS]']
        result = ''
        for sentence in sentences:
            if not sentence:
                continue
            sentence = sentence.strip()
            tokens = tokens + self.tokenizer.tokenize(sentence) + ['[SEP]']
        return tokens

    def correctPunctuation(self, entryText):
        sentences = entryText.split('[SEP]')
        tokens = ['[CLS]']
        result = ''
        for sentence in sentences:
            if not sentence:
                continue
            sentence = sentence.strip()
            words = sentence.split(' ')
            s = '[MASK] '.join(words)
            result = result + s + '[SEP]'
            s = s.split('[MASK]')
            for i in range(len(s)):
                if i == 0:
                    tokens = tokens + self.tokenizer.tokenize(s[i])
                else:
                    tokens = tokens + ['[MASK]'] + self.tokenizer.tokenize(s[i])
            tokens = tokens + ['[SEP]']

        token_input = self.tokenizer.convert_tokens_to_ids(tokens)
        token_input = token_input + [0] * (512 - len(token_input))

        mask_input = [0] * 512
        for i in range(len(mask_input)):
            if token_input[i] == MASK_ID:
                mask_input[i] = 1

        seg_input = [0] * 512
        for i in range(len(seg_input)):
            if token_input[i] != 0 and token_input[i] != 101 and token_input[i] != 102:
                seg_input[i] = 1

        token_input = np.asarray([token_input])
        mask_input = np.asarray([mask_input])
        seg_input = np.asarray([seg_input])

        predicts = self.model.predict([token_input, seg_input, mask_input])[0]
        predicts_ids = np.argmax(predicts, axis=-1)[0][:len(tokens)]
        predicts_prob = list()
        for i in range(len(predicts_ids)):
            predicts_prob.append(predicts[0][i][predicts_ids[i]])
        predicts = list(zip(predicts_prob, predicts_ids))

        out = []
        for i in range(len(mask_input[0])):
            if mask_input[0][i] == 1:
                out.append(predicts[i])

        points_ids = self.tokenizer.convert_tokens_to_ids(self.points)
        for i in range(len(out)):
            replacement = ''
            if points_ids.__contains__(out[i][1]) and out[i][0] > 0.7:
                replacement = self.tokenizer.convert_ids_to_tokens([out[i][1]])
                replacement = replacement[0].replace('##', '')
            result = result.replace('[MASK]', replacement, 1)

        sentences = result.split('[SEP]')
        final = ''
        for sentence in sentences:
            if not sentence:
                continue
            sentence = sentence[0].upper() + sentence[1:]
            final = final + sentence + '. '
        return final


## Класс для получения всех возможных изменений
class PossibleValues:

    def __init__(self, russian=True):
        words = {}
        if russian:
            wordsFile = 'deprecated_model_multi/words.txt'
        else:
            wordsFile = 'bert_model_en/words.txt'
        with tf.io.gfile.GFile(wordsFile, "r") as reader:
            while True:
                token = tokenization.convert_to_unicode(reader.readline())
                if not token:
                    break
                token = token.strip()
                words[token] = token
        self.WORDS = words
        self.russian = russian

    def edit_step(self, word):
        if self.russian:
            letters = 'абвгдеёжзийклмнопрстуфхцчшщъыьэюя'
        else:
            letters = 'abcdefghijklmnopqrstuvwxyz'
        splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
        deletes = [L + R[1:] for L, R in splits if R]
        transposes = [L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1]
        replaces = [L + c + R[1:] for L, R in splits if R for c in letters]
        inserts = [L + c + R for L, R in splits for c in letters]
        return set(deletes + transposes + replaces + inserts)

    def edits2(self, word):
        return (e2 for e1 in self.edit_step(word)
                for e2 in self.edit_step(e1))

    def known(self, words):
        return set(w for w in words if w in self.WORDS)

    def get_begins(self, word):
        return set(w for w in self.WORDS if w.startswith(word) and len(w) <= len(word) + 2)

    def edit_candidates(self, word, assume_wrong=False, fast=True, return_self_if_found=True):
        if fast:
            ttt = self.known(self.edit_step(word)) or {word}
        else:
            ttt = self.known(self.edit_step(word)) or self.known(self.edits2(word)) or {word}

        if return_self_if_found and len(self.known([word])) > 0:
            return self.known([word])
            # if len(word) > 2:
            #     word_start = word[:len(word) - 1]
            #     return self.get_begins(word_start)
            # else:
            #     return self.known(self.edit_step(word)) or {word}

        ttt = self.known([word]) | ttt
        return list(ttt)

    def orph_edit_candidates(self, word):
        if len(word) > 2:
            word_start = word[:len(word) - 1]
            return self.get_begins(word_start)
        else:
            return self.known(self.edit_step(word)) or {word}
