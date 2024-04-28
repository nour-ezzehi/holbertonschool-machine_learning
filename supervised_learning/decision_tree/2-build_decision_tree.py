#!/usr/bin/env python3
"""
decison tree
"""
import numpy as np


class Node:
    """Node class"""

    def __init__(
            self,
            feature=None,
            threshold=None,
            left_child=None,
            right_child=None,
            is_root=False,
            depth=0):
        self.feature = feature
        self.threshold = threshold
        self.left_child = left_child
        self.right_child = right_child
        self.is_leaf = False
        self.is_root = is_root
        self.sub_population = None
        self.depth = depth

    def max_depth_below(self):
        """returns the depth """
        if self is Leaf:
            return self.depth
        if self.left_child is None and self.right_child is None:
            return self.depth
        left_depth = self.left_child.max_depth_below()
        right_depth = self.right_child.max_depth_below()

        return max(left_depth, right_depth)

    def count_nodes_below(self, only_leaves=False):
        """count leaves"""
        if self.is_leaf:
            return 1
        if self.left_child:
            left_count = self.left_child.count_nodes_below(only_leaves)
        else:
            left_count = 0

        if self.right_child:
            right_count = self.right_child.count_nodes_below(only_leaves)
        else:
            right_count = 0

        if only_leaves:
            return left_count + right_count
        else:
            return 1 + left_count + right_count

    def left_child_add_prefix(self, text):
        """left child string"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:-1]:
            new_text += ("    |  " + x) + "\n"
        return new_text

    def right_child_add_prefix(self, text):
        """right child string"""
        lines = text.split("\n")
        new_text = "    +--" + lines[0] + "\n"
        for x in lines[1:]:
            if x:
                new_text += ("       " + x) + "\n"
        return new_text

    def __str__(self):
        """Return the Node and its children"""
        result = (
            f"root [feature={self.feature}, threshold={self.threshold}]"
            if self.is_root
            else f"-> node [feature={self.feature}, threshold={self.threshold}]"
        )
        if self.left_child:
            result += "\n" + self.left_child_add_prefix(str(self.left_child))
        if self.right_child:
            result += self.right_child_add_prefix(str(self.right_child))
        return result


class Leaf(Node):
    """Leaf class"""

    def __init__(self, value, depth=None):
        super().__init__()
        self.value = value
        self.is_leaf = True
        self.depth = depth

    def max_depth_below(self):
        """depth of a leaf"""
        return self.depth

    def count_nodes_below(self, only_leaves=False):
        """one leaf counted"""
        return 1

    def __str__(self):
        """print leaf"""
        return (f"-> leaf [value={self.value}]")


class Decision_Tree():
    """Decision tree class"""

    def __init__(
            self,
            max_depth=10,
            min_pop=1,
            seed=0,
            split_criterion="random",
            root=None):
        self.rng = np.random.default_rng(seed)
        if root:
            self.root = root
        else:
            self.root = Node(is_root=True)
        self.explanatory = None
        self.target = None
        self.max_depth = max_depth
        self.min_pop = min_pop
        self.split_criterion = split_criterion
        self.predict = None

    def depth(self):
        """depth of the tree"""
        return self.root.max_depth_below()

    def count_nodes(self, only_leaves=False):
        """count leaves"""
        return self.root.count_nodes_below(only_leaves=only_leaves)

    def __str__(self):
        """print tree"""
        return self.root.__str__()
