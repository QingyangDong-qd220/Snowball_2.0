# -*- coding: utf-8 -*-
"""
Phrase object
"""
from .order import Order

class Phrase(object):

    def __init__(self, sentence_tokens, relations, prefix_length, suffix_length, confidence=0.0, label='0', training=False, root_name=None, all_names=None):

        """Phrase Object

        Class for handling which relations and entities appear in a sentence, the base type used for clustering and generating extraction patterns

        Arguments:
            sentence_tokens {[list} -- The sentence tokens from which to generate the Phrase
            relations {list} -- List of Relation objects to be tagged in the sentence
            prefix_length {int} -- Number of tokens to assign to the prefix
            suffix_length {int} -- Number of tokens to assign to the suffix
        """

        self.label = label
        self.cluster_label = None
        self.main_cluster_label = None
        self.training = training
        self.sentence_tokens = sentence_tokens
        self.full_sentence = ' '.join(sentence_tokens)
        self.number_of_entities = 0
        self.relations = relations
        self.elements = {}
        self.entities = []
        self.order = []
        self.prefix_length = int(prefix_length) # do not change this once defined
        self.suffix_length = int(suffix_length)
        self.confidence = confidence
        self.root_name = root_name
        self.all_names = all_names
        if sentence_tokens and relations:
            self.create()


    def __repr__(self):

        return self.to_string()


    def to_string(self):

        output_string = 'phrase ' + self.label + ': '
        output_string += ' '.join(self.elements['prefix']['tokens']) + ' '
        if isinstance(self.entities[0].tag, tuple):
            output_string += '(' + ', '.join([i for i in self.entities[0].tag]) + ') '
        else:
            output_string += '(' + self.entities[0].tag + ') '
        for i in range(0, self.number_of_entities - 1):
            output_string += ' '.join(self.elements['middle_' + str(i+1)]['tokens']) + ' '
            if isinstance(self.entities[i+1].tag, tuple):
                output_string += '(' + ', '.join([i for i in self.entities[i+1].tag]) + ') '
            else:
                output_string += '(' + self.entities[i+1].tag + ') '
        output_string += ' '.join(self.elements['suffix']['tokens'])

        return output_string


    def create(self):

        """ Create a phrase from known relations"""

        sentence = self.sentence_tokens

        combined_entity_list = []
        for relation in self.relations:
            for entity in relation:
                if entity in combined_entity_list:
                    continue
                else:
                    combined_entity_list.append(entity)

        # Number of entities
        self.number_of_entities = len(combined_entity_list)
        number_of_middles = self.number_of_entities - 1

        # Determine the entitiy ordering
        sorted_entity_list = sorted(combined_entity_list, key=lambda t: t.start)
        self.entities = sorted_entity_list

        # Create ordering
        self.order = Order(tags=[e.tag for e in self.entities], root_name=self.root_name, all_names=self.all_names)

        # Create the phrase elements, prefix, middles, suffix
        prefix_tokens = list(sentence[sorted_entity_list[0].start - self.prefix_length : sorted_entity_list[0].start])
        if len(prefix_tokens) == 0:
            prefix_tokens = ['<Blank>']
        self.elements['prefix'] = {'tokens': prefix_tokens}

        for m in range(0, number_of_middles):
            prev_entity_end = sorted_entity_list[m].end
            next_entitiy_start = sorted_entity_list[m+1].start
            middle_tokens = list(sentence[prev_entity_end : next_entitiy_start])
            if len(middle_tokens) == 0:
                middle_tokens = ['<Blank>']
            self.elements['middle_' + str(m+1)] = {'tokens': middle_tokens}

        suffix_tokens = list(sentence[sorted_entity_list[-1].end : sorted_entity_list[-1].end + self.suffix_length])
        if len(suffix_tokens) == 0:
            suffix_tokens = ['<Blank>']
        self.elements['suffix'] = {'tokens': suffix_tokens}

        return


    def reset_elements(self, new_prefix_legnth, new_suffix_length):

        """ change the values of prefix_length and suffix_length"""

        for element in self.elements.keys():
            self.elements[element]['vector'] = None

        self.prefix_length = int(new_prefix_legnth)
        self.suffix_length = int(new_suffix_length)
        self.create()

        return