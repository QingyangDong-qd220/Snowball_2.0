# -*- coding: utf-8 -*-
"""
Classes for defining new chemical relationships
"""
import copy
from .entity import Entity


class Relation(object):
    """Relation class

    Essentially a placeholder for related entities
    """

    def __init__(self, entities, confidence):

        """Init

        Arguments:
            entities {list} -- List of Entity objects that are present in this relationship
            confidence {float} -- The confidence of the relation
        """
        
        self.entities = copy.copy(entities)
        self.confidence = confidence
    

    def __len__(self):

        return len(self.entities)


    def __getitem__(self, idx):

        return self.entities[idx]


    def __setitem__(self, idx, value):
        
        self.entities[idx] = value
    

    def __eq__(self, other):
        if len(self.entities) != len(other.entities):
            return False
            
        for i, entity in enumerate(self.entities):
            if entity != other.entities[i]:
                return False
            else:
                continue
        return True


    def __repr__(self):

        return '<' + ', '.join([str(i) for i in self.entities]) + '>'


    def __str__(self):

        return self.__repr__()
    

    def serialize(self):

        output = {}
        for entity in self.entities:
            entity_data = entity.serialize()
            entity_root = list(entity_data.keys())[0]
            if entity_root not in output.keys():
                output[entity_root] = {}
            output[entity_root].update(entity_data[entity_root])

        output['confidence'] = self.confidence

        return output
    

    def is_valid(self, value_before_unit=False):

        # Returns False if relationship contains entities with different tags at the same location
        for i in range(len(self.entities)):
            e1 = self.entities[i]
            for j in range(i+1, len(self.entities)):
                e2 = self.entities[j]
                if e1.tag != e2.tag and e1.start == e2.start and e1.end == e2.end:
                    return False
                elif value_before_unit:
                    tag1 = e1.tag.split("__")
                    tag2 = e2.tag.split("__")
                    if tag1[1] == "raw_units" and tag2[1] == "raw_value" and tag1[0] == tag2[0]:
                        return False

        return True


