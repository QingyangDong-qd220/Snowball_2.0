# -*- coding: utf-8 -*-
"""
Snowball parser version 2.0
authors:
    qd220@cam.ac.uk
    cc889@cam.ac.uk
"""

import logging
import copy
import io
import os
from os.path import basename
import joblib, pickle
import time
from collections import OrderedDict
from pprint import pprint
import numpy as np
import six
from lxml import etree
from itertools import product, combinations, islice
from operator import attrgetter
import gc

from ..doc.document import Document, Paragraph
from ..doc.text import Sentence
from ..model.base import BaseModel
from ..parse.auto import AutoSentenceParser
from ..model import Compound, FloatType, StringType
from ..parse import Any, I, OneOrMore, Optional, R, W, ZeroOrMore, join, merge
from ..parse.cem import chemical_name
from .cluster import Cluster
from .entity import Entity
from .phrase import Phrase
from .relationship import Relation
from .order import Order
from .utils import match, KnuthMorrisPratt
from ..parse.base import BaseSentenceParser

log = logging.getLogger(__name__)



class Snowball(BaseSentenceParser):
    
    """
    Main Snowball class
    Usage: see the Snowball_Manual.ipynb file
    For full detail see the associated paper: https://www.nature.com/articles/sdata2018111
    """

    def __init__(self, model,
                 tc = 0.8,
                 tsim_h = 0.8,
                 tsim_l = None,
                 prefix_weight = 0.1,
                 middle_weight = 0.8,
                 suffix_weight = 0.1,
                 prefix_length = 1,
                 suffix_length = 1,
                 max_candidate_combinations = 5000,
                 value_before_unit = True,
                 entity_training_threshold = 20,
                 save_dir = 'Snowball_model/',
                 file_name = None,
                 clustering_method = 'max',
                 second_round = True,
                 compound_filter = None, 
                 phrase_filter = None,
                 normalization = True):

        self.model = model
        self.phrases = [] # labelled phrases but not yet clusterd
        self.clusters = []
        self.max_candidate_combinations = max_candidate_combinations
        self.value_before_unit = value_before_unit
        self.entity_training_threshold = entity_training_threshold
        if save_dir[-1] == "/":
            self.save_dir = save_dir
        else:
            self.save_dir = save_dir + "/"

        if file_name == None:
            self.file_name = model.__name__
        else:
            self.file_name = file_name
        self.trained_files = []
        self.clustered = False

        self.tc = tc
        self.input_tsim_h = tsim_h
        if tsim_l is None:
            self.input_tsim_l = self.input_tsim_h / 2
        else:
            self.input_tsim_l = tsim_l
        self.input_prefix_weight = prefix_weight
        self.input_middle_weight = middle_weight
        self.input_suffix_weight = suffix_weight
        self.input_prefix_length = int(prefix_length)
        self.input_suffix_length = int(suffix_length)
        self.clustering_method = clustering_method
        self.second_round = second_round
        self.detailed_model = self.expand_model(self.model)
        self.normalization = normalization
        if compound_filter == None:
            self.compound_filter = self.no_filter
        else:
            self.compound_filter = compound_filter
        if phrase_filter == None:
            self.phrase_filter = self.no_filter
        else:
            self.phrase_filter = phrase_filter

        for parameter in [tc, tsim_h, self.input_tsim_l, prefix_weight, middle_weight, suffix_weight]:
            if not 0.0 <= parameter <= 1.0:
                raise ValueError("Parameter values must be between 0 and 1;\ncheck: tc, tsim_h, tsim_l, prefix_weight, middle_weight, suffix_weight.")

        if abs(1 - prefix_weight - middle_weight - suffix_weight) >= 1e-3:
            raise ValueError("Weights must be normalized to 1")

        if self.input_tsim_l > tsim_h:
            raise ValueError("tsim_l must be smaller than tsim_h")
        
        if clustering_method not in ["all", "best", "first", "max", "min"]:
            raise ValueError("Invalid clustering method, it can be best / all / first / max / min.")

        self.save()


    def no_filter(self, input_object):

        """
        place holder for compound_filter and phrase_filter, which returns a boolean indicating whether a compound/phrase is kept or not.
        this default filter keeps everything. 
        """
        return True


    @property
    def num_of_clusters(self):

        return len(self.clusters)


    @property
    def num_of_phrases(self):

        """
        count the number of unclustered phrases plus the phrases inside existing clusters
        """

        nop = len(self.phrases)
        for c in self.clusters:
            nop += c.num_of_phrases

        return nop


    @property
    def tsim_h(self):
        return self.input_tsim_h


    @tsim_h.setter
    def tsim_h(self, new_tsim_h):

        self.uncluster()
        self.input_tsim_h = new_tsim_h
        if not 0.0 <= new_tsim_h <= 1.0:
            raise ValueError("tsim_h must be between 0 and 1")
        elif new_tsim_h < self.tsim_l:
            raise ValueError("tsim_h must not be smaller than tsim_l")

        return


    @property
    def tsim_l(self):

        return self.input_tsim_l


    @tsim_l.setter
    def tsim_l(self, new_tsim_l):

        self.input_tsim_l = new_tsim_l
        if not 0.0 <= new_tsim_l <= 1.0:
            raise ValueError("tsim_l must be between 0 and 1")
        elif new_tsim_l > self.tsim_h:
            raise ValueError("tsim_l must be smaller than tsim_h")


    @property
    def prefix_length(self):
        return self.input_prefix_length

    
    @prefix_length.setter
    def prefix_length(self, new_prefix_length):

        self.uncluster()
        self.input_prefix_length = int(new_prefix_length)
        for p in self.phrases:
            p.reset_elements(int(new_prefix_length), self.suffix_length)

        return


    @property
    def suffix_length(self):
        return self.input_suffix_length


    @suffix_length.setter
    def suffix_length(self, new_suffix_length):

        self.uncluster()
        self.input_suffix_length = int(new_suffix_length)
        for p in self.phrases:
            p.reset_elements(self.prefix_length, int(new_suffix_length))

        return


    @property
    def prefix_weight(self):

        return self.input_prefix_weight

    
    @prefix_weight.setter
    def prefix_weight(self, new_prefix_weight):

        self.uncluster()
        self.input_prefix_weight = new_prefix_weight

        if abs(1 - self.prefix_weight - self.middle_weight - self.suffix_weight) >= 1e-3:
            print("\nWeights not yet normalized to 1; check prefix_weight")

        return


    @property
    def middle_weight(self):
        
        return self.input_middle_weight

    
    @middle_weight.setter
    def middle_weight(self, new_middle_weight):

        self.uncluster()
        self.input_middle_weight = new_middle_weight

        if abs(1 - self.prefix_weight - self.middle_weight - self.suffix_weight) >= 1e-3:
            print("\nWeights not yet normalized to 1; check middle_weight")

        return


    @property
    def suffix_weight(self):
        
        return self.input_suffix_weight

    
    @suffix_weight.setter
    def suffix_weight(self, new_suffix_weight):

        self.uncluster()
        self.input_suffix_weight = new_suffix_weight

        if abs(1 - self.prefix_weight - self.middle_weight - self.suffix_weight) >= 1e-3:
            print("\nWeights not yet normalized to 1; check suffix_weight")

        return


    @classmethod
    def load(cls, path):

        """
        Load a snowball instance from file

        Arguments:
            path {str} -- path to the pkl file

        Returns:
            self -- A Snowball Instance
        """

        print('Loading model from existing file')
        # return joblib.load(path)
        with open(path, 'rb') as f:
            return pickle.load(f)


    def import_from_model(self, path, confidence_limit=1.0):

        """
        Import / copy all labeled phrases from another Snowball model to the current one

        Arguments:
            path {str} -- path to the Snowball pkl file
            confidence_limit {float} -- only import phrases with confidence score equal or greater than this value
        """

        with open(path, 'rb') as f:
            other_model = pickle.load(f)

        if other_model.model != self.model:
            # raise ValueError('Property model is different, cannot load data.')
            print("Loading from a different property: {}.".format(other_model.model.__name__))
        
        i = 0
        # copy unclustered phrases from other model
        for p in other_model.phrases:
            if p.confidence >= confidence_limit:
                p.label = str(self.num_of_phrases)
                p.cluster_label = None
                p.main_cluster_label = None
                if p.prefix_length != self.prefix_length or p.suffix_length != self.suffix_length:
                    p.reset_elements(self.prefix_length, self.suffix_length)
                self.phrases.append(p)
                i += 1

        # copy already clustered phrases from sub-clusters
        for c in other_model.clusters:
            for p in c.phrases:
                if p.confidence >= confidence_limit:
                    p.label = str(self.num_of_phrases)
                    p.cluster_label = None
                    p.main_cluster_label = None
                    if p.prefix_length != self.prefix_length or p.suffix_length != self.suffix_length:
                        p.reset_elements(self.prefix_length, self.suffix_length)
                    self.phrases.append(p)
                    i += 1
        
        self.save()
        print('Loaded {} phrases from: {}.\n'.format(i, other_model.file_name))

        return

    
    def import_from_list(self, path, confidence_limit=1.0):

        """
        Import / copy all labeled phrases from a list of phrases to the current model
        This method is more robust against version conflicts between ChemDataExtractor and Snowball

        Arguments:
            path {str} -- path to the Snowball pkl file
            confidence_limit {float} -- only import phrases with confidence score equal or greater than this value
        """

        with open(path, 'rb') as f:
            other_list = pickle.load(f)
        
        print(f'Importing labelled sentences from {path}.')

        i = 0
        for p in other_list:
            if p.confidence >= confidence_limit:
                p.label = str(self.num_of_phrases)
                p.cluster_label = None
                p.main_cluster_label = None
                if p.prefix_length != self.prefix_length or p.suffix_length != self.suffix_length:
                    p.reset_elements(self.prefix_length, self.suffix_length)
                self.phrases.append(p)
                i += 1

        self.save()
        print('Loaded {} phrases from: {}.\n'.format(i, path))
        return


    def export_to_list(self, path):

        """
        write all phrases (including the ones assigned into clusters) to a list file,
        so that labelled sentences can be imported by other models

        Arguments:
            path {str} -- path to the pkl file
        """

        # temp_model = copy.copy(self)
        phrase_list = copy.copy(self.phrases)
        for c in self.clusters:
            for p in c.phrases:
                temp = copy.copy(p)
                temp.cluster_label = None
                phrase_list.phrases.append(temp)

        with open(path, 'wb') as f:
            pickle.dump(phrase_list, f)
        print(f'Saved {len(phrase_list)} phrases to {path}.')

        return


    def save(self, details=False):

        """
        write the Snowball instance to a file

        Arguments:
            details {boolean} -- whether centroid patterns and relations are also saved as files
        """

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

        # joblib.dump(self, '{0.save_dir}{0.file_name}.sav'.format(self))
        with open('{0.save_dir}{0.file_name}.pkl'.format(self), 'wb') as f:
            pickle.dump(self, f)

        with io.open("{0.save_dir}{0.file_name}_clusters.txt".format(self), 'w+', encoding='utf-8') as f:
            f.write("Cluster set contains {} clusters.\n\n".format(self.num_of_clusters))
            for c in self.clusters:
                f.write("Cluster {0.label} contains {0.num_of_phrases} phrases:\n".format(c))
                for phrase in c.phrases:
                    f.write("\tconfidence = {0.confidence:.5f} | {0.full_sentence}\n".format(phrase))
                p = c.pattern
                f.write("Confidence score = {0.confidence:.5f} | centroid pattern = {0.string_like}\n\n".format(p))

        with io.open("{0.save_dir}{0.file_name}_phrases.txt".format(self), 'w+', encoding='utf-8') as f:
            f.write("Model contains {} unclustered phrases.\n\n".format(len(self.phrases)))
            for p in self.phrases:
                f.write("{0.full_sentence} | confidence = {0.confidence:.5f}\n".format(p))
                    
        if details:
            with io.open("{0.save_dir}{0.file_name}_patterns.txt".format(self), 'w+', encoding='utf-8') as f:
                for c in self.clusters:
                    p = c.pattern
                    f.write("{0.string_like} | confidence score = {0.confidence}\n\n".format(p))

            with io.open('{0.save_dir}{0.file_name}_relations.txt'.format(self), 'w+', encoding='utf-8') as wf:
                for c in self.clusters:
                    for phrase in c.phrases:
                        for relation in phrase.relations:
                            wf.write("{0} | Confidence = {0.confidence}\n".format(relation))

        return


    def train_from_corpus(self, corpus, skip=0):

        """
        train the snowball model on a specified corpus

        Arguments:
            corpus {str or list} -- path to a corpus of documents or list of training sentence/document objects
            skip {int} -- ignore the first several items in the corpus
        """

        if isinstance(corpus, str):
            corpus_list = os.listdir(corpus)
            for i, file_name in enumerate(corpus_list[skip:]):
                print(f'Training {i + skip + 1}/{len(corpus_list)}: {file_name}')
                f = os.path.join(corpus, file_name)
                self.train_from_file(f)

        elif isinstance(corpus, list):
            for s in corpus[skip:]:
                if isinstance(s, Sentence):
                    self.train_from_sentence(s)
                elif isinstance(s, Document):
                    self.train_from_document(s)
                else:
                    print('untrainable: list element is not a Sentence or a Document')
        
        else:
            print('untrainable: invalid corpus')

        return


    def train_from_file(self, filename):

        """
        Train Snowball from the elements of a file. This is the only training function that logs the filename
        When training from a file, ALWAYS COMPLETE THE FILE BEFORE STOPPING THE CODE

        Arguments:
            filename {str} -- the file path to parse
        """
        
        if basename(filename) in self.trained_files:
            print(f'Already trained: {basename(filename)}\n')
        else:
            with open(filename, 'rb') as f:
                d = Document().from_file(f)
            self.train_from_document(d)
            self.trained_files.append(basename(filename))
            self.save()

        return


    def train_from_document(self, d):

        """
        Train Snowball from a Document object

        Arguments:
            d {Document object} -- the document to parse
        """

        for p in d.paragraphs:
            for s in p.sentences:
                if s.end - s.start > 600: # skip sentence if it is too long
                    continue

                sent_definitions = s.definitions
                if sent_definitions:
                    self.model.update(sent_definitions)

                self.train_from_sentence(s)
        self.model.reset_updatables()

        return


    def train_from_sentence(self, s):

        """
        Train Snowball from a single sentence

        Arguments:
            s {Sentence object} -- a single sentence to parse
        """

        candidate_found = False
        problem = True
        candidate_relationships = self.find_candidates(s.tagged_tokens)
        # print(len(candidate_relationships))
        
        if len(candidate_relationships) == 0:
            pass
        elif len(candidate_relationships) <= self.entity_training_threshold: # relation-based training
            candidate_found = True
            print("\n\n{}\n".format(s))
            chosen_candidates = self.select_from_input(message="relationships: ", corpus=candidate_relationships, text=s.text)
            if chosen_candidates:
                new_phrase = Phrase(s.raw_tokens, chosen_candidates, self.prefix_length, self.suffix_length, confidence=1.0, label=str(self.num_of_phrases), training=True, root_name=self.model.__name__.lower(), all_names=list(self.detailed_model.keys()))
                self.phrases.append(new_phrase)
                self.save()

        else: # entity-based training
            candidate_found = True
            print("\n\n{}\n".format(s))
            pprint(self.find_entity(s.tagged_tokens))
            print(" ")
            grouped_entities = self.group_entity(self.find_entity(s.tagged_tokens))[0]
            # pprint(grouped_entities)

            # set the number of relationships from user
            while True:
                res = input("How many relations: ")
                if res == 'n':
                    with io.open("{0.save_dir}{0.file_name}_notes.txt".format(self), 'a', encoding='utf-8') as f:
                        f.write(f"{s.text}\n\n")
                    return candidate_found
                try:
                    num_of_relationships = int(res)
                    print(" \n")
                except ValueError:
                    print("bad input, try again.")
                    continue
                else:
                    break
            # print(num_of_relationships)

            # rearrange entity tags for better visuals
            main_model = self.model.__name__.lower()
            tag_list = ["compound__names", "compound__labels", "compound__roles", main_model+"__specifier", main_model+"__raw_value", main_model+"__raw_units"]
            for tag in self.detailed_model.keys():
                if tag == "compound" or tag == main_model:
                    continue
                else:
                    tag_list.extend([tag+"__specifier", tag+"__raw_value", tag+"__raw_units"])
            # pprint(tag_list)

            # construct relationships from entities
            chosen_candidates = []
            for i in range(1, num_of_relationships+1):
                chosen_candidate = []
                print("\n{}\n\nConstructing relationship {}/{}: \n\n".format(s, i, num_of_relationships))
                for tag in tag_list:
                    if tag not in grouped_entities.keys():
                        continue
                    else:
                        chosen_entity = self.select_from_input(message="{}: ".format(tag), corpus=grouped_entities[tag], text=s.text, keyword=tag, single=True)
                        print("\n")
                        if chosen_entity is not None:
                            chosen_candidate.append(chosen_entity)
                temp = sorted(chosen_candidate, key=attrgetter('start', 'tag', 'text'))
                r = Relation(temp, confidence=1.0)
                # if unvalid relation, give the user a chance to stop
                if not r.is_valid(value_before_unit=False):
                    check = input("Relation is not valid, press Enter to proceed, or Ctrl + C to abort.")
                print("\nRelation {}: {}\n".format(i, r))
                chosen_candidates.append(r)
            # pprint(chosen_candidates)

            # construct phrase from relationships
            if chosen_candidates:
                new_phrase = Phrase(s.raw_tokens, chosen_candidates, self.prefix_length, self.suffix_length, confidence=1.0, label=str(self.num_of_phrases), training=True, root_name=self.model.__name__.lower(), all_names=list(self.detailed_model.keys()))
                self.phrases.append(new_phrase)
                self.save()
                # print(new_phrase)

        return candidate_found


    def select_from_input(self, message, corpus, text, keyword="Candidate", single=False):

        """
        Parse user input to slice out selected items

        Arguments:
            message {str} -- hint before input
            corpus {list} -- all items to be selected
            text {str} -- full sentence
            keyword {str} -- a word before printing available choices
            single {boolean} -- whether only one choice is expected

        Returns:
            results {list} -- selected items
        """

        for i, candidate in enumerate(corpus):
            print("{} {}: {}\n".format(keyword, i, candidate))

        while True:
            try:
                results = []
                res = input(message)
                print(" ")
                if res:
                    if res == 'n' and not single:
                        with io.open("{0.save_dir}{0.file_name}_notes.txt".format(self), 'a', encoding='utf-8') as f:
                            f.write(f"{text}\n\n")
                        return None
                    numbers = list(OrderedDict.fromkeys(res.split(','))) # remove duplicate
                    for number in numbers:
                        results.append(corpus[int(number)])
            except (ValueError, IndexError) as error:
                print("\nBad input, try again. Use , to separate numbers.\n")
                continue
            else:
                break

        if single == True:
            if results:
                results = results[0]
            else:
                return None

        return results


    def find_entity(self, tagged_tokens):

        """
        Parse user input to slice out selected items

        Arguments:
            tagged_tokens {list} -- a list of tagged word tokens, each being part of a sentence

        Returns:
            entities_list {list} -- all entities found in the tokens
        """

        # print(tagged_tokens, "\n\n")
        entities_list = []
        sentence_parser = AutoSentenceParser(lenient=True)
        all_models = [] # individual models, not just names
        # print(type(tagged_tokens), tagged_tokens)
        tokens = [token[0] for token in tagged_tokens]
        # print(tokens)

        for model in self.model.flatten(): # random order
            if model.__name__ == 'Compound':
                continue
            elif model not in all_models:
                all_models.append(model)
        # print(all_models)

        # detect all entities matched by model
        for model in all_models:
            sentence_parser.model = model
            results = list(sentence_parser.root.scan(tagged_tokens))
            if results:
                for result in results:
                    # check this line if POS tagging is not good
                    # print(model.__name__, '\n', etree.tostring(result[0]), '\n')
                    for text, tag, parse_expression in self.retrieve_entities(model, result[0]):
                        # print("text and tag:", text, tag, "\n")
                        if tag.endswith('raw_units'):
                            if text.endswith('('): # remove left bracket (and space) of raw_units
                                text = text.split(' (')[0].split('(')[0]
                            elif text.endswith(')'): # remove right bracket (and space) of raw_units
                                text = text.split(' )')[0].split(')')[0]
                        pattern = Sentence(text).raw_tokens
                        # print(model.__name__, "pattern: ", pattern, tag)
                        for index in KnuthMorrisPratt(tokens, pattern):
                            # print("index: ", index)
                            entity = Entity(text, tag, parse_expression, index, index + len(pattern))
                            if entity in entities_list:
                                    break
                            else:
                                entities_list.append(entity)
        entities_list.sort(key=attrgetter('start', 'tag', 'text'))
        # pprint(self.detailed_model)
        # print("entities_list:")
        # pprint(entities_list)
        # print("\n\n")

        if not entities_list:
            return []

        # Filter out incomplete entities required by individual models
        entity_tags = []
        for entity in entities_list:
            if entity.tag not in entity_tags:
                entity_tags.append(entity.tag)
        # print(entity_tags)

        # insert your custom entity here
        # entities_list.append(Entity("CTPA", "compound__names", parse_expression, 19, 20))
        # entities_list.append(Entity("1.1", "bandgaptemp__raw_value", parse_expression, 34, 35))

        for name, detail in self.detailed_model.items():
            for attribute, requirement in detail.items():
                if requirement[0] and requirement[1]:
                    if '__'.join((name, attribute)) not in entity_tags: # entity required but not present
                        # print('triggered', '__'.join((name, attribute)))
                        return []
                elif not requirement[0] and requirement[1]:
                    if '__'.join((name, attribute)) not in entity_tags: # remove everything about this property
                        entities_list = [entity for entity in entities_list if not entity.tag.startswith(name)]
        # print('Final entities:')
        # pprint(entities_list)
        # print(' ')

        return entities_list


    def group_entity(self, entities_list):

        """
        sort out all possible combinations of entities based on property model requirements,
        so that all candidate relationships can be constructed

        Arguments:
            entities_list {list} -- all entities in a sentence

        Returns:
            entity_package {list} -- all combinations of entities, each corresponds to a set of property requirement
        """

        if not entities_list:
            return []

        # generate combinations of models fulfilling the requirement of parent models
        full_model_list = []
        for name, detail in self.detailed_model.items():
            if list(detail.values())[0][0] == True:
                full_model_list.append([name])
            else:
                full_model_list.append([name, None])
        
        grouped_model_list = []
        for model_combo in product(*full_model_list):
            grouped_model_list.append([i for i in list(model_combo) if i is not None])
        # pprint(grouped_model_list)
        # pprint(entities_list)
        # print(' ')

        # generate all possible entity groups specified by model requirements
        entity_package = []
        for grouped_model in grouped_model_list:
            grouped_entities = {}
            for entity in entities_list:
                if entity.tag.split("__")[0] not in grouped_model:
                    continue
                elif entity.tag not in grouped_entities.keys():
                    grouped_entities[entity.tag] = [entity]
                else:
                    grouped_entities[entity.tag].append(entity)
            if grouped_entities not in entity_package:
                entity_package.append(grouped_entities)
        # pprint(entity_package[0])
        # print(' ')

        return entity_package


    def find_candidates(self, tagged_tokens):

        """
        Find all candidate relationships of the property model within a sentence

        Arguments:
            tagged_tokens {list} -- tagged word tokens of a Sentence object

        Returns:
            candidate_relationships {list} -- all candidate relationships of a sentence
        """

        entity_package = self.group_entity(self.find_entity(tagged_tokens))

        candidate_relationships = []
        for i in entity_package:
            for candidate in product(*list(i.values())):
                temp = sorted(list(candidate), key=attrgetter('start', 'tag', 'text'))
                r = Relation(temp, confidence=1.0)
                if r.is_valid(value_before_unit=self.value_before_unit):
                    candidate_relationships.append(r)
        # print('all candidate relationships:')
        # pprint(candidate_relationships)

        return candidate_relationships


    def retrieve_entities(self, model, result):

        """
        Recursively retrieve the entities from a parse result for a given property model

        Arguments:
            model {QuantityModel object} -- ChemDataExtractor property model
            result {lxml.etree.element} -- The parse result
        
        Yields:
            (text, tag, parse_expression) -- an entity
        """

        if isinstance(result, list):
            for r in result:
                for entity in self.retrieve_entities(model, r):
                    yield entity
        else:
            for tag, field in model.fields.items():
                # print(tag))
                if hasattr(field, 'model_class'): # then it is a nested property or a compound
                    for nested_entity in self.retrieve_entities(field.model_class, result.xpath('./' + tag)):
                        yield nested_entity
                else:
                    text_list = result.xpath('./' + tag + '/text()')
                    for text in text_list:
                        yield (text, '__'.join((model.__name__.lower(), tag)), field.parse_expression)
    

    def expand_model(self, model):

        """
        unfold a property model to see the requirements on all items of a property model

        Arguments:
            model {QuantityModel object} -- the property model

        Returns:
            detailed_model {dict} -- the requirements of each name/tag
        """

        def get_info(model, parent_required):
            for tag, field in model.fields.items():
                if hasattr(field, 'model_class'): # then it is a nested property or a compound
                    for nested_model in get_info(field.model_class, field.required):
                        yield nested_model
                else:
                    yield (model.__name__.lower(), tag, [parent_required, field.required])
                    
        detailed_model = {}
        for name, tag, required in get_info(self.model, True):
            if name not in detailed_model.keys():
                detailed_model[name] = {}
            if tag in detailed_model[name].keys():
                continue
            else:
                detailed_model[name][tag] = required
        # pprint(detailed_model)

        return detailed_model


    def cluster_all(self, method=None):

        """
        assign all labelled phrases to clusters and clear cache of labelled phrases

        Arguments:
            method {string} -- clutering method ('all'/'first'/'best'/'max'/'min')

        """

        def insert_one(phrase, method):

            try:
                self.single_pass(phrase, method=method)
            except IndexError: # TODO need more examples
                print('Invalid candidates, maybe wrong input?\nphrase {}\n'.format(phrase.label))

            return

        def find_extrema(triangle, method):

            exts, exts_index = [], []
            for row_index, row in enumerate(triangle):
                if np.isnan(row).all():
                    exts.append(float("nan"))
                    exts_index.append(float("nan"))
                else:
                    if method == "max":
                        exts.append(np.nanmin(row))
                        exts_index.append(np.nanargmin(row))
                    elif method == "min":
                        exts.append(np.nanmax(row))
                        exts_index.append(np.nanargmax(row))
            # print(exts, "\n", exts_index, "\n")

            if np.isnan(exts).all():
                return [float("nan"), float("nan")]
            else:
                if method == "max":
                    ext = np.nanmin(exts)
                    ext_index = np.nanargmin(exts)
                elif method == "min":
                    ext = np.nanmax(exts)
                    ext_index = np.nanargmax(exts)
                result = [ext_index, exts_index[ext_index]]
                # print(result, triangle[result[0],result[1]])

            return result

        def main_cluster_map(phrases):

            """
            generate a map between main-clusters and phrases
            mapping [list] = [
                [cluster.order, [phrase.label, ...]], 
                ...
            ]
            """

            # if self.clustered == True:
                # self.uncluster()

            mapping = []
            for p in phrases:
                for i, j in enumerate(mapping):
                    if p.order == j[0]:
                        p.main_cluster_label = i
                        j[1].append(int(p.label))
                        break
                else:
                    p.main_cluster_label = len(mapping)
                    mapping.append([p.order, [int(p.label), ]])

            return mapping


        if self.clustered == True:
            print('\nThis model has already has {} cluster(s). Make sure parameters are constant.'.format(self.num_of_clusters))
            self.uncluster()

        if method is None:
            method = self.clustering_method

        print(f'Clustering {self.file_name}.\n')
        count = 0
        # sequencial single-pass algorithm (SPA)
        if method == "best" or method == "all" or method == "first":
            for phrase in self.phrases:
                insert_one(phrase, method=method)
                count += 1

        # ordered single-pass (SPA) (similarity triangle)
        elif method == "max" or method == "min":
            mapping = main_cluster_map(self.phrases)
            for main_label, [order, labels] in enumerate(mapping): # within a main-cluster
                if len(labels) == 1: # skip triangle if only one phrase per main-cluster
                    insert_one(self.phrases[labels[0]], method="best")
                    count += 1
                else: # calculate triangle
                    triangle = np.zeros((len(labels), len(labels)))
                    for i, pi in enumerate(labels):
                        for j, pj in enumerate(labels):
                            if i >= j:
                                triangle[i, j] = None
                            else:
                                temp_cluster = Cluster(label=self.phrases[pi].label, training=False)
                                temp_cluster.add_phrase(self.phrases[pi], self.prefix_weight, self.middle_weight, self.suffix_weight)
                                sim = match(self.phrases[pj], temp_cluster, self.prefix_weight, self.middle_weight, self.suffix_weight)
                                triangle[i, j] = sim
                    # print(order, "\n")
                    # for k in triangle:
                        # print(["%.2f" % j for j in k])
                    # print("\n")
                
                    # assign to clusters in pairs to maximize num_of_clusters
                    phrase_list = list(np.arange(len(labels)))
                    while phrase_list:
                        if len(phrase_list) >= 2:
                            one = find_extrema(triangle, method)
                            insert_one(self.phrases[labels[one[0]]], method="best")
                            insert_one(self.phrases[labels[one[1]]], method="best")
                            count += 2
                            triangle[one[0], :] = float("nan")
                            triangle[one[1], :] = float("nan")
                            triangle[:, one[0]] = float("nan")
                            triangle[:, one[1]] = float("nan")
                            phrase_list.remove(one[0])
                            phrase_list.remove(one[1])
                        elif len(phrase_list) == 1:
                            insert_one(self.phrases[labels[phrase_list[0]]], method="best")
                            phrase_list.pop(0)
                            count += 1
                            triangle[one[0], :] = float("nan")
                        # for k in triangle:
                            # print(["%.2f" % j for j in k])
                        # print("\n")
                    
        else:
            raise ValueError("Invalid clustering method, it can be best / all / first / max / min.")

        self.clustered = True
        self.phrases = []
        self.save()
        print(f'Clustered {count} sentences; now have {len(self.clusters)} clusters.\n')

        return


    def uncluster(self, training_only=False):

        """
        transfer phrases from clusters into the phrase cache and delete all clusters

        Arguments:
            training_only {boolean} -- whether to ignore phrases curated during data extraction
        """

        for c in self.clusters:
            for p in c.phrases:
                if not training_only:
                    p.cluster_label = None
                    self.phrases.append(p)
                elif p.training == True:
                    p.cluster_label = None
                    self.phrases.append(p)

        self.clusters = []
        self.clustered = False
        self.save()

        return


    def single_pass(self, phrase, method):

        """
        the original single pass algorithm

        Arguments:
            phrase {Phrase object} -- the Phrase to cluster
            method {string} -- clustering method
        """

        if self.num_of_clusters == 0:
            # print('create first cluster')
            cluster0 = Cluster(label=str(self.num_of_clusters), training=True)
            cluster0.add_phrase(phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)
            self.clusters.append(cluster0)

        else:
            if method == "best":
                self.classify_best(phrase)
            elif method == "all":
                self.classify_all(phrase)
            elif method == "first":
                self.classify_first(phrase)
            else:
                raise ValueError("Invalid clustering method, it can be best / all / first.")

        return


    def classify_all(self, phrase):

        """
        Assign a phrase to all matching clusters

        Arguments:
            phrase {Phrase object} -- the Phrase to cluster
        """

        phrase_added = False
        for cluster in self.clusters:
            if phrase.order == cluster.order: # only compare clusters that have the same ordering of entities
                similarity = match(phrase, cluster, self.prefix_weight, self.middle_weight, self.suffix_weight)
                if similarity >= self.tsim_h:
                    cluster.add_phrase(phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)
                    phrase_added = True
                    
        if phrase_added is False:
            # print('creating a new cluster')
            new_cluster = Cluster(label=str(self.num_of_clusters), training=True)
            new_cluster.add_phrase(phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)
            self.clusters.append(new_cluster)

        return


    def classify_first(self, phrase):

        """
        Assign a phrase to the first matching cluster

        Arguments:
            phrase {Phrase object} -- the Phrase to cluster
        """

        for cluster in self.clusters:
            if phrase.order == cluster.order: # only compare clusters that have the same ordering of entities
                similarity = match(phrase, cluster, self.prefix_weight, self.middle_weight, self.suffix_weight)
                if similarity >= self.tsim_h:
                    cluster.add_phrase(phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)
                    break

        else:
            # print('creating a new cluster')
            new_cluster = Cluster(label=str(self.num_of_clusters), training=True)
            new_cluster.add_phrase(phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)
            self.clusters.append(new_cluster)

        return


    def classify_best(self, phrase):

        """
        Assign a phrase to the best matching cluster

        Arguments:
            phrase {Phrase object} -- the Phrase to cluster
        """

        sim_dict = {}
        for i, cluster in enumerate(self.clusters):
            if phrase.order == cluster.order: # only compare clusters that have the same ordering of entities
                similarity = match(phrase, cluster, self.prefix_weight, self.middle_weight, self.suffix_weight)
                if similarity >= self.tsim_h:
                    sim_dict[str(i)] = similarity
        
        if sim_dict:
            best_idx = int(max(sim_dict, key=sim_dict.get))
            self.clusters[best_idx].add_phrase(phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)
        else:
            # print('creating a new cluster')
            new_cluster = Cluster(label=str(self.num_of_clusters), training=True)
            new_cluster.add_phrase(phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)
            self.clusters.append(new_cluster)

        return


    def find_best_cluster(self, tokens):

        """
        find the highest similarity / match between a cluster and a possible candidate relationship extracted from the sentence

        Arguments:
            tokens {list} -- a list of tagged tokens of the unseen sentence

        Returns:
            best_candidate_phrase {Phrase object} -- a candidate phrase from the sentence which has the highest similarity score
            best_candidate_cluster {Cluster object} -- a cluster from the model that matches to the candidate phrase
            best_candidate_phrase_score {float} -- the confidence score of the candidate phrase
            best_candidate_phrase_sim {float} -- the similarity score of the candidate phrase vs. best matching cluster
        """

        # print("\n\nParsing sentece", ' '.join([t[0] for t in tokens]))

        # Use the default tagger to find candidate relationships
        candidate_relations = self.find_candidates(tokens)
        # print("Candidates", candidate_relations)
        unique_names = set() # TODO may have two compound_names with same text but different start&end
        reduced_candidate_relations = []
        for i in candidate_relations:
            for j in i.entities:
                if j.tag == 'compound__names':
                    if self.compound_filter(j.text):
                        unique_names.add(j.text)
                        reduced_candidate_relations.append(i)
        # print("unique_names: ", unique_names, '; num of candidate relations:', len(reduced_candidate_relations))

        # a sentence can contain multiple relationships (different compound names)
        all_combs = [i for r in range(1, len(unique_names) + 1) for i in islice(combinations(reduced_candidate_relations, r), self.max_candidate_combinations)]
        # print(f'num of all_combs: {len(all_combs)}. ')
        # pprint(all_combs)

        # Create a candidate phrase for each possible combination
        all_candidate_phrases = []
        for combination in all_combs:
            rels = [r for r in combination]
            new_rels = copy.copy(rels)
            
            candidate_phrase = Phrase([t[0] for t in tokens], new_rels, self.prefix_length, self.suffix_length, confidence=0.0, label=str(self.num_of_phrases), training=False, root_name=self.model.__name__.lower(), all_names=list(self.detailed_model.keys()))
            if self.phrase_filter(candidate_phrase):
                all_candidate_phrases.append(candidate_phrase)
        # print('num of all_candidate_phrases:', len(all_candidate_phrases))

        # Only pick the phrase with the best confidence score
        best_candidate_phrase = None
        best_candidate_cluster = None
        best_candidate_phrase_score = 0
        best_candidate_phrase_sim = 0

        for candidate_phrase in all_candidate_phrases:
            # print("Evaluating candidate", candidate_phrase)
            # For each cluster, compare the candidate phrase to the cluster extraction pattern
            best_match_score = 0
            best_match_cluster = None
            confidence_term = 1.0
            count = 0
            for cluster in self.clusters:
                if candidate_phrase.order != cluster.order: # only use one main-cluster
                    # print("not same order")
                    continue
                match_score = match(candidate_phrase, cluster, self.prefix_weight, self.middle_weight, self.suffix_weight)
                # print(cluster.label, match_score)
                
                if match_score >= self.tsim_l:
                    # print(f"\nsimilarity = {match_score:.5f}\n\t{candidate_phrase}\n\t{cluster.pattern}\n")
                    confidence_term *= (1.0 - (match_score * cluster.pattern.confidence))
                    count += 1
                    # print(match_score, cluster.pattern.confidence)
                    if match_score > best_match_score:
                        best_match_cluster = cluster
                        best_match_score = match_score

            if count != 0:
                coefficient = count * self.normalization + 1 * (not self.normalization)
                phrase_confidence_score = 1.0 - confidence_term**(1/coefficient)
            else:
                phrase_confidence_score = 0.0
            # print(f"confidenc_term = {confidence_term}")
            # print(f"Confidence = {phrase_confidence_score}\n")

            if phrase_confidence_score > best_candidate_phrase_score:
                best_candidate_phrase_sim = best_match_score
                candidate_phrase.confidence = phrase_confidence_score
                best_candidate_phrase = candidate_phrase
                best_candidate_phrase_score = phrase_confidence_score
                best_candidate_cluster = best_match_cluster
                
        if best_candidate_phrase:
            # print(best_candidate_phrase_sim)
            for candidate_relation in best_candidate_phrase.relations:
                candidate_relation.confidence = best_candidate_phrase_score

        del all_candidate_phrases
        gc.collect()
        # print(best_candidate_phrase)
        # print(best_candidate_cluster.pattern)

        return best_candidate_phrase, best_candidate_cluster, best_candidate_phrase_score, best_candidate_phrase_sim


    def find_best_phrase(self, best_candidate_phrase, best_candidate_cluster):

        """
        re-calculate the confidence score of a phrase within a cluster if the similarity is higher than self.tsim_h

        Arguments:
            best_candidate_phrase {Phrase object} -- the candidate phrase from the sentence which has the highest similarity score
            best_candidate_cluster {Cluster object} -- the cluster from the model that matches to the candidate phrase

        Returns:
            phrase_confidence_score {float} -- the new confidence score of the candidate phrase
        """

        confidence_term = 1.0
        count = 0
        for phrase in best_candidate_cluster.phrases:
            temp_cluster = Cluster(label=best_candidate_cluster.label, training=False)
            temp_cluster.add_phrase(phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)
            match_score = match(best_candidate_phrase, temp_cluster, self.prefix_weight, self.middle_weight, self.suffix_weight)
            del temp_cluster
            gc.collect()

            if match_score >= self.tsim_l:
                count += 1
                confidence_term *= 1.0 - (match_score * phrase.confidence)
        
        if count == 0:
            phrase_confidence_score = 0.0
        elif count == 1:
            phrase_confidence_score = 1.0 - confidence_term
        else:
            phrase_confidence_score = 1.0 - confidence_term**(1/(count))
        # print(phrase_confidence_score)

        return phrase_confidence_score


    #: Override from BaseSentenceParser
    def parse_sentence(self, tokens, testing=False):

        """
        Parse a sentence with the Snowball parser

        Arguments:
            tokens {list} -- The tokens to parse
        Returns:
            The matching records
        """
        # from CDE2.1, input for this function in BaseSentenceParser changed from tagged_tokens to Sentence object!
        if isinstance(tokens, Sentence):
            tokens = tokens.tagged_tokens

        best_candidate_phrase, best_candidate_cluster, best_candidate_phrase_score, best_candidate_phrase_sim = self.find_best_cluster(tokens)
        # print(best_candidate_phrase)

        # only for testing purpose
        # if testing and best_candidate_phrase is None:
            # return None, 0, 0
        
        if best_candidate_phrase:
            # second round of confidence calculation if the candidate_phrase is similar enough
            # print(best_candidate_phrase_sim)
            if self.second_round and best_candidate_phrase_sim >= self.tsim_h:
                best_candidate_phrase_score = self.find_best_phrase(best_candidate_phrase, best_candidate_cluster)
                best_candidate_phrase.confidence = best_candidate_phrase_score
                for relation in best_candidate_phrase.relations:
                    relation.confidence = best_candidate_phrase_score
            
            # update the knowlegde base if the match is very good
            if best_candidate_phrase_sim >= self.tsim_h:
                best_candidate_cluster.add_phrase(best_candidate_phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)

            # generate a new cluster if phrase has high enough confidence but not enough similarity (should already satisfy best_sim>=tsim_l)
            elif best_candidate_phrase_score >= self.tc:
                new_cluster = Cluster(label=str(self.num_of_clusters), training=False)
                new_cluster.add_phrase(best_candidate_phrase, self.prefix_weight, self.middle_weight, self.suffix_weight)
                self.clusters.append(new_cluster)

            self.save()
            # print(best_candidate_phrase.relations)

            # only for testing purpose, also need to comment out the next 3 lines
            # if testing:
                # return best_candidate_phrase, best_candidate_phrase_score, best_candidate_phrase_sim

            for relation in best_candidate_phrase.relations:
                for model in self.interpret(relation, best_candidate_phrase.full_sentence):
                    yield model


    def _get_data(self, field_name, field, relation_data):
        
        if hasattr(field, 'model_class'): # temperature, compound
            field_result = relation_data[field_name]  # {spec. val. units}

            if field_result is None and field.required and not field.contextual:
                raise TypeError('Could not find element for ' + str(field_name))
            elif field_result is None:
                return None

            field_data = {}
            for subfield_name, subfield in six.iteritems(field.model_class.fields):  # compound, names
                data = self._get_data(subfield_name, subfield, field_result)
                if data is not None:
                    field_data.update(data)
            field_object = field.model_class(**field_data)
            log.debug('Created for' + field_name)
            log.debug(field_object)
            return {field_name: field_object}
        elif hasattr(field, 'field'):
            # Case that we have listtype
            # Always only takes the first found one though
            field = field.field
            field_data = self._get_data(field_name, field, relation_data)
            if field_data is not None:
                if field_name not in field_data.keys() or field_data[field_name] is None:
                    return None
                field_data = [field_data[field_name]]
            elif field_data is None and field.required and not field.contextual:
                raise TypeError('Could not find element for ' + str(field_name))
            elif field_data is None:
                return None
            return {field_name: field_data}
        else:
            try:
                field_result = relation_data[field_name]
            except KeyError:
                return {}
            if field_result is None and field.required and not field.contextual:
                raise TypeError('Could not find element for ' + str(field_name))
                
            return {field_name: field_result}


    def interpret(self, relation, sentence):

        """
        Convert a detected relation to a ModelType Record

        Arguments:
            relation {Relation object} -- an extracted relationship from the sentence
            sentence {str} -- the text version of the sentence
        """

        # print("\n\n", "Interpreting")
        # Set the confidence field if not already set
        if not 'confidence' in self.model.fields.keys():
            setattr(self.model, 'confidence', FloatType())
        if not 'sentence' in self.model.fields.keys():
            setattr(self.model, 'sentence', StringType())

        # Get the serialized relation data
        relation_data = relation.serialize()
        # print(relation_data)
        # Do conversions etc
        models = list(self.model.flatten())
        model_instances = []

        for model in models:
            model_name = model.__name__.lower()
            if model_name == self.model.__name__.lower():
                is_root_instance=True
            else:
                is_root_instance=False

            # print(model_name)
            model_data = {}
            if model_name in relation_data.keys() and 'specifier' in relation_data[model_name].keys():
                model_data['specifier'] = relation_data[model_name]['specifier']
            model_data['confidence'] = relation_data['confidence']
            model_data['sentence'] = sentence

            if hasattr(model, 'dimensions') and not model.dimensions:
                # the specific entities of a DimensionlessModel are retrieved explicitly and packed into a dictionary
                raw_value = relation_data[model_name]['raw_value']
                value = self.extract_value(raw_value)
                error = self.extract_error(raw_value)

                model_data.update({"raw_value": raw_value,
                                        "value": value,
                                        "error": error})

            elif hasattr(model, 'dimensions') and model.dimensions and model_name in relation_data.keys():
                # the specific entities of a QuantityModel are retrieved explicitly and packed into a dictionary
                # print(etree.tostring(result))
                raw_value = relation_data[model_name]['raw_value']
                raw_units = relation_data[model_name]['raw_units']
                value = self.extract_value(raw_value)
                error = self.extract_error(raw_value)
                units = None
                try:
                    units = self.extract_units(raw_units, strict=True)
                except TypeError as e:
                    log.debug(e)
                model_data.update({"raw_value": raw_value,
                                        "raw_units": raw_units,
                                        "value": value,
                                        "error": error,
                                        "units": units})
            elif hasattr(model, 'category') and model.category and model_name in relation_data.keys():
                raw_value = relation_data[model_name]['raw_value']
                model_data.update({"raw_value": raw_value})

            for field_name, field in six.iteritems(model.fields):
                if field_name not in ['raw_value', 'raw_units', 'value', 'units', 'error', 'specifier', 'confidence', 'sentence']:
                    try:
                        data = self._get_data(field_name, field, relation_data)
                        if data is not None:
                            model_data.update(data)
                    # if field is required, but empty, the requirements have not been met
                    except (TypeError, KeyError) as e:
                        log.debug(self.model)
                        log.debug(e)
            model_instance = model(**model_data)
            model_instances.append((model_instance, is_root_instance))
        
        root_model_instance = [i[0] for i in model_instances if i[1]][0]
        for m in model_instances:
            if m[1]:
                continue
            root_model_instance.merge_all(m[0])
        

        # records the parser that was used to generate this record, can be used for evaluation
        root_model_instance.record_method = self.__class__.__name__

        yield root_model_instance
        

