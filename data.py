"""Dataset loader and data utilities.

Author:
    Shrey Desai and Yasumasa Onoe
"""

import collections
import itertools
import torch

from torch.utils.data import Dataset
from random import shuffle
from utils import cuda, load_dataset

import spacy

PAD_TOKEN = '[PAD]'
UNK_TOKEN = '[UNK]'


class Vocabulary:
    """
    This class creates two dictionaries mapping:
        1) words --> indices,
        2) indices --> words.

    Args:
        samples: A list of training examples stored in `QADataset.samples`.
        vocab_size: Int. The number of top words to be used.

    Attributes:
        words: A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
            All words will be lowercased.
        encoding: A dictionary mapping words (string) to indices (int).
        decoding: A dictionary mapping indices (int) to words (string).
    """
    def __init__(self, samples, vocab_size):
        self.words = self._initialize(samples, vocab_size)
        self.encoding = {word: index for (index, word) in enumerate(self.words)}
        self.decoding = {index: word for (index, word) in enumerate(self.words)}

    def _initialize(self, samples, vocab_size):
        """
        Counts and sorts all tokens in the data, then it returns a vocab
        list. `PAD_TOKEN and `UNK_TOKEN` are added at the beginning of the
        list. All words are lowercased.

        Args:
            samples: A list of training examples stored in `QADataset.samples`.
            vocab_size: Int. The number of top words to be used.

        Returns:
            A list of top words (string) sorted by frequency. `PAD_TOKEN`
            (at position 0) and `UNK_TOKEN` (at position 1) are prepended.
        """
        vocab = collections.defaultdict(int)
        for sample in samples:
            (_, passage, question) = sample[:3]
            for token in itertools.chain(passage, question):
                vocab[token.lower()] += 1
        top_words = [
            word for (word, _) in
            sorted(vocab.items(), key=lambda x: x[1], reverse=True)
        ][:vocab_size]
        words = [PAD_TOKEN, UNK_TOKEN] + top_words
        return words
    
    def __len__(self):
        return len(self.words)


class Tokenizer:
    """
    This class provides two methods converting:
        1) List of words --> List of indices,
        2) List of indices --> List of words.

    Args:
        vocabulary: An instantiated `Vocabulary` object.

    Attributes:
        vocabulary: A list of top words (string) sorted by frequency.
            `PAD_TOKEN` (at position 0) and `UNK_TOKEN` (at position 1) are
            prepended.
        pad_token_id: Index of `PAD_TOKEN` (int).
        unk_token_id: Index of `UNK_TOKEN` (int).
    """
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.pad_token_id = self.vocabulary.encoding[PAD_TOKEN]
        self.unk_token_id = self.vocabulary.encoding[UNK_TOKEN]

    def convert_tokens_to_ids(self, tokens):
        """
        Converts words to corresponding indices.

        Args:
            tokens: A list of words (string).

        Returns:
            A list of indices (int).
        """
        return [
            self.vocabulary.encoding.get(token, self.unk_token_id)
            for token in tokens
        ]

    def convert_ids_to_tokens(self, token_ids):
        """
        Converts indices to corresponding words.

        Args:
            token_ids: A list of indices (int).

        Returns:
            A list of words (string).
        """
        return [
            self.vocabulary.decoding.get(token_id, UNK_TOKEN)
            for token_id in token_ids
        ]


class QADataset(Dataset):
    """
    This class creates a data generator.

    Args:
        args: `argparse` object.
        path: Path to a data file (.gz), e.g. "datasets/squad_dev.jsonl.gz".

    Attributes:
        args: `argparse` object.
        meta: Dataset metadata (e.g. dataset name, split).
        elems: A list of raw examples (jsonl).
        samples: A list of preprocessed examples (tuple). Passages and
            questions are shortened to max sequence length.
        tokenizer: `Tokenizer` object.
        batch_size: Int. The number of example in a mini batch.
    """
    def __init__(self, args, path):
        self.args = args
        self.nlp = spacy.load("en_core_web_sm")
        self.net_to_index_dict = dict()
        self.net_to_index_dict[0] = 1
        self.dep_to_index_dict = dict()
        self.dep_to_index_dict[0] = 1
        self.meta, self.elems = load_dataset(path)
        self.samples = self._create_samples()
        self.tokenizer = None
        self.batch_size = args.batch_size if 'batch_size' in args else 1
        self.pad_token_id = self.tokenizer.pad_token_id \
            if self.tokenizer is not None else 0

    def net_to_index(self, id):
        if id not in self.net_to_index_dict:
            self.net_to_index_dict[id] = len(self.net_to_index_dict) + 1
        return self.net_to_index_dict[id]

    def dep_to_index(self, id):
        if id not in self.dep_to_index_dict:
            self.dep_to_index_dict[id] = len(self.dep_to_index_dict) + 1
        return self.dep_to_index_dict[id]

    def get_spacy_tags(self, text, data_tokens):
        spacy_doc = self.nlp(text)
        spacy_tokens = [token for token in spacy_doc]
        spacy_offsets = [token.idx for token in spacy_tokens]
        off_to_ind = dict(zip(spacy_offsets, range(len(spacy_offsets))))
        pos = [
            spacy_tokens[off_to_ind[offset]].pos if offset in off_to_ind else 0
            for (token, offset) in data_tokens
        ][:self.args.max_context_length]
        net = [
            self.net_to_index(spacy_tokens[off_to_ind[offset]].ent_type) if offset in off_to_ind else 0
            for (token, offset) in data_tokens
        ][:self.args.max_context_length]
        dep = [
            self.dep_to_index(spacy_tokens[off_to_ind[offset]].dep) if offset in off_to_ind else 0
            for (token, offset) in data_tokens
        ][:self.args.max_context_length]
        return pos, net, dep

    def _create_samples(self):
        """
        Formats raw examples to desired form. Any passages/questions longer
        than max sequence length will be truncated.

        Returns:
            A list of words (string).
        """
        samples = []
        for elem in self.elems:
            # Unpack the context paragraph. Shorten to max sequence length.
            passage = [
                token.lower() for (token, offset) in elem['context_tokens']
            ][:self.args.max_context_length]

            # Get parts of speech from Spacy
            ppos, pnet, pdep = self.get_spacy_tags(elem['context'], elem['context_tokens'])
            assert len(ppos) == len(passage)

            # Each passage has several questions associated with it.
            # Additionally, each question has multiple possible answer spans.
            for qa in elem['qas']:
                qid = qa['qid']
                question = [
                    token.lower() for (token, offset) in qa['question_tokens']
                ][:self.args.max_question_length]
                qpos, qnet, qdep = self.get_spacy_tags(qa['question'], qa['question_tokens'])
                assert len(qpos) == len(question)

                # Select the first answer span, which is formatted as
                # (start_position, end_position), where the end_position
                # is inclusive.
                answers = qa['detected_answers']
                answer_start, answer_end = answers[0]['token_spans'][0]
                samples.append(
                    (qid, passage, question, ppos, qpos, pnet, qnet, pdep, qdep, answer_start, answer_end)
                )
            if self.args.mini and len(samples) > 100:
                break
                
        return samples

    def _create_data_generator(self, shuffle_examples=False):
        """
        Converts preprocessed text data to Torch tensors and returns a
        generator.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A generator that iterates through all examples one by one.
            (Tuple of tensors)
        """
        if self.tokenizer is None:
            raise RuntimeError('error: no tokenizer registered')

        example_idxs = list(range(len(self.samples)))
        if shuffle_examples:
            shuffle(example_idxs)

        passages = []
        questions = []
        passages_pos = []
        questions_pos = []
        passages_net = []
        questions_net = []
        passages_dep = []
        questions_dep = []
        start_positions = []
        end_positions = []
        for idx in example_idxs:
            # Unpack QA sample and tokenize passage/question.
            qid, passage, question, ppos, qpos, pnet, qnet, pdep, qdep, answer_start, answer_end = self.samples[idx]

            # Convert words to tensor.
            passage_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(passage)
            )
            question_ids = torch.tensor(
                self.tokenizer.convert_tokens_to_ids(question)
            )
            # Convert ints to tensor.
            ppos_ids = torch.tensor(ppos)
            qpos_ids = torch.tensor(qpos)
            pnet_ids = torch.tensor(pnet)
            qnet_ids = torch.tensor(qnet)
            pdep_ids = torch.tensor(pdep)
            qdep_ids = torch.tensor(qdep)
            answer_start_ids = torch.tensor(answer_start)
            answer_end_ids = torch.tensor(answer_end)

            # Store each part in an independent list.
            passages.append(passage_ids)
            questions.append(question_ids)
            passages_pos.append(ppos_ids)
            questions_pos.append(qpos_ids)
            passages_net.append(pnet_ids)
            questions_net.append(qnet_ids)
            passages_dep.append(pdep_ids)
            questions_dep.append(qdep_ids)
            start_positions.append(answer_start_ids)
            end_positions.append(answer_end_ids)

        return zip( passages, questions, 
                    passages_pos, questions_pos, 
                    passages_net, questions_net, 
                    passages_dep, questions_dep, 
                    start_positions, end_positions)

    def _create_batches(self, generator, batch_size):
        """
        This is a generator that gives one batch at a time. Tensors are
        converted to "cuda" if necessary.

        Args:
            generator: A data generator created by `_create_data_generator`.
            batch_size: Int. The number of example in a mini batch.

        Yields:
            A dictionary of tensors containing a single batch.
        """
        current_batch = [None] * batch_size
        no_more_data = False
        # Loop through all examples.
        while True:
            bsz = batch_size
            # Get examples from generator
            for i in range(batch_size):
                try:
                    current_batch[i] = list(next(generator))
                except StopIteration:  # Run out examples
                    no_more_data = True
                    bsz = i  # The size of the last batch.
                    break
            # Stop if there's no leftover examples
            if no_more_data and bsz == 0:
                break

            passages = []
            questions = []
            ppos = []
            qpos = []
            pnet = []
            qnet = []
            pdep = []
            qdep = []
            start_positions = torch.zeros(bsz)
            end_positions = torch.zeros(bsz)
            max_passage_length = 0
            max_question_length = 0
            # Check max lengths for both passages and questions
            for ii in range(bsz):
                passages.append(current_batch[ii][0])
                questions.append(current_batch[ii][1])
                ppos.append(current_batch[ii][2])
                qpos.append(current_batch[ii][3])
                pnet.append(current_batch[ii][4])
                qnet.append(current_batch[ii][5])
                pdep.append(current_batch[ii][6])
                qdep.append(current_batch[ii][7])
                start_positions[ii] = current_batch[ii][8]
                end_positions[ii] = current_batch[ii][9]
                max_passage_length = max(
                    max_passage_length, len(current_batch[ii][0])
                )
                max_question_length = max(
                    max_question_length, len(current_batch[ii][1])
                )

            def pad_pq(ps, qs):
                padded_ps = torch.zeros(bsz, max_passage_length)
                padded_qs = torch.zeros(bsz, max_question_length)
                for iii, pq in enumerate(zip(ps, qs)):
                    pi, qi = pq
                    padded_ps[iii][:len(pi)] = pi
                    padded_qs[iii][:len(qi)] = qi
                return padded_ps, padded_qs

            # Assume pad token index is 0. Need to change here if pad token
            # index is other than 0.
            padded_passages, padded_questions = pad_pq(passages, questions)
            padded_ppos, padded_qpos = pad_pq(ppos, qpos)
            padded_pnet, padded_qnet = pad_pq(pnet, qnet)
            padded_pdep, padded_qdep = pad_pq(pdep, qdep)

            # Create an input dictionary
            batch_dict = {
                'passages': cuda(self.args, padded_passages).long(),
                'questions': cuda(self.args, padded_questions).long(),
                'ppos': cuda(self.args, padded_ppos).long(),
                'qpos': cuda(self.args, padded_qpos).long(),
                'pnet': cuda(self.args, padded_pnet).long(),
                'qnet': cuda(self.args, padded_qnet).long(),
                'pdep': cuda(self.args, padded_pdep).long(),
                'qdep': cuda(self.args, padded_qdep).long(),
                'start_positions': cuda(self.args, start_positions).long(),
                'end_positions': cuda(self.args, end_positions).long()
            }

            if no_more_data:
                if bsz > 0:
                    # This is the last batch (smaller than `batch_size`)
                    yield batch_dict
                break
            yield batch_dict

    def get_batch(self, shuffle_examples=False):
        """
        Returns a data generator that supports mini-batch.

        Args:
            shuffle_examples: If `True`, shuffle examples. Default: `False`

        Returns:
            A data generator that iterates though all batches.
        """
        return self._create_batches(
            self._create_data_generator(shuffle_examples=shuffle_examples),
            self.batch_size
        )

    def register_tokenizer(self, tokenizer):
        """
        Stores `Tokenizer` object as an instance variable.

        Args:
            tokenizer: If `True`, shuffle examples. Default: `False`
        """
        self.tokenizer = tokenizer
    
    def __len__(self):
        return len(self.samples)
