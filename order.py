"""
Phrase order object to generalize the first level of clustering
"""


class Order:

    def __init__(self, tags, root_name=None, all_names=None):

        self.tags = tags
        self.root_name = root_name
        self.all_names = all_names

        self.root_positions = []
        self.compound_positions = []
        self.labeled_positions = {}
        # for name in all_names:
            # if name == root_name or name == "compound":
                # continue
            # else:
                # self.labeled_positions[name] = []
        # print(self.labeled_positions)

        for i, tag in enumerate(self.tags):
            name, role = tag.split("__")
            if name == root_name:
                self.root_positions.append([i, role])
            elif name == "compound":
                self.compound_positions.append([i, role])
            elif name in self.all_names:
                if name in self.labeled_positions.keys():
                    self.labeled_positions[name].append([i, role])
                else:
                    self.labeled_positions[name] = [[i, role], ]
        self.nested_positions = list(self.labeled_positions.values())
        # print(all_names)
        # print("root:", self.root_positions, "\n")
        # print("compound:", self.compound_positions, "\n")
        # from pprint import pprint
        # pprint(self.nested_positions)
        # print(" ")


    def __eq__(self, other):

        if self.root_positions == other.root_positions and self.compound_positions == other.compound_positions:
            if len(self.nested_positions) != len(other.nested_positions):
                # not same number of nested properties
                # print("not same length")
                # print(self.nested_positions, self.root_name)
                # print(other.nested_positions, other.root_name)
                return False
            elif not self.nested_positions and not other.nested_positions:
                # no nested properties for both
                # print("no nested properties")
                return True
            else:
                # compare positions of nested properties
                count = 0
                for i in self.nested_positions:
                    if i in other.nested_positions:
                        count += 1
                # print(count)
                if count == len(self.nested_positions):
                    return True
                else:
                    return False
        else:
            # print("root or compound not right")
            return False


    def __repr__(self):

        return "<" + ", ".join([str(i) for i in self.tags]) + ">"


    def __str__(self):

        return self.__repr__()


    def __len__(self):

        return len(self.tags)

    
    def __getitem__(self, idx):

        return self.tags[idx]


    def __setitem__(self, idx, value):

        self.tags[idx] = value