def is_link(obj):
    if not isinstance(obj, list):
        return False
    if len(obj) != 2:
        return False
    if not isinstance(obj[0], str):
        return False
    if not isinstance(obj[1], int) and not isinstance(obj[1], float):
        return False
    return True

# The GraphBuilder is just a utility class that outputs graphs in the form expected by the ComfyUI back-end
class GraphBuilder:
    _default_prefix_root = ""
    _default_prefix_call_index = 0
    _default_prefix_graph_index = 0

    def __init__(self, prefix = None):
        if prefix is None:
            self.prefix = GraphBuilder.alloc_prefix()
        else:
            self.prefix = prefix
        self.nodes = {}
        self.id_gen = 1

    @classmethod
    def set_default_prefix(cls, prefix_root, call_index, graph_index = 0):
        cls._default_prefix_root = prefix_root
        cls._default_prefix_call_index = call_index
        cls._default_prefix_graph_index = graph_index

    @classmethod
    def alloc_prefix(cls, root=None, call_index=None, graph_index=None):
        if root is None:
            root = GraphBuilder._default_prefix_root
        if call_index is None:
            call_index = GraphBuilder._default_prefix_call_index
        if graph_index is None:
            graph_index = GraphBuilder._default_prefix_graph_index
        result = f"{root}.{call_index}.{graph_index}."
        GraphBuilder._default_prefix_graph_index += 1
        return result

    def node(self, class_type, id=None, **kwargs):
        if id is None:
            id = str(self.id_gen)
            self.id_gen += 1
        id = self.prefix + id
        if id in self.nodes:
            return self.nodes[id]

        node = Node(id, class_type, kwargs)
        self.nodes[id] = node
        return node

    def lookup_node(self, id):
        id = self.prefix + id
        return self.nodes.get(id)

    def finalize(self):
        output = {}
        for node_id, node in self.nodes.items():
            output[node_id] = node.serialize()
        return output

    def replace_node_output(self, node_id, index, new_value):
        node_id = self.prefix + node_id
        to_remove = []
        for node in self.nodes.values():
            for key, value in node.inputs.items():
                if is_link(value) and value[0] == node_id and value[1] == index:
                    if new_value is None:
                        to_remove.append((node, key))
                    else:
                        node.inputs[key] = new_value
        for node, key in to_remove:
            del node.inputs[key]

    def remove_node(self, id):
        id = self.prefix + id
        del self.nodes[id]

class Node:
    def __init__(self, id, class_type, inputs):
        self.id = id
        self.class_type = class_type
        self.inputs = inputs
        self.override_display_id = None

    def out(self, index):
        return [self.id, index]

    def set_input(self, key, value):
        if value is None:
            if key in self.inputs:
                del self.inputs[key]
        else:
            self.inputs[key] = value

    def get_input(self, key):
        return self.inputs.get(key)

    def set_override_display_id(self, override_display_id):
        self.override_display_id = override_display_id

    def serialize(self):
        serialized = {
            "class_type": self.class_type,
            "inputs": self.inputs
        }
        if self.override_display_id is not None:
            serialized["override_display_id"] = self.override_display_id
        return serialized

def add_graph_prefix(graph, outputs, prefix):
    # Change the node IDs and any internal links
    new_graph = {}
    for node_id, node_info in graph.items():
        # Make sure the added nodes have unique IDs
        new_node_id = prefix + node_id
        new_node = { "class_type": node_info["class_type"], "inputs": {} }
        for input_name, input_value in node_info.get("inputs", {}).items():
            if is_link(input_value):
                new_node["inputs"][input_name] = [prefix + input_value[0], input_value[1]]
            else:
                new_node["inputs"][input_name] = input_value
        new_graph[new_node_id] = new_node

    # Change the node IDs in the outputs
    new_outputs = []
    for n in range(len(outputs)):
        output = outputs[n]
        if is_link(output):
            new_outputs.append([prefix + output[0], output[1]])
        else:
            new_outputs.append(output)

    return new_graph, tuple(new_outputs)

class ExecutionBlocker:
    """
    Return this from a node and any users will be blocked with the given error message.
    If the message is None, execution will be blocked silently instead.
    Generally, you should avoid using this functionality unless absolutely necessary. Whenever it's
    possible, a lazy input will be more efficient and have a better user experience.
    This functionality is useful in two cases:
    1. You want to conditionally prevent an output node from executing. (Particularly a built-in node
       like SaveImage. For your own output nodes, I would recommend just adding a BOOL input and using
       lazy evaluation to let it conditionally disable itself.)
    2. You have a node with multiple possible outputs, some of which are invalid and should not be used.
       (I would recommend not making nodes like this in the future -- instead, make multiple nodes with
       different outputs. Unfortunately, there are several popular existing nodes using this pattern.)
    """
    def __init__(self, message):
        self.message = message


class ExecutionFailureBlocker(ExecutionBlocker):
    def __init__(self, node_id):
        super().__init__(None)
        self.node_id = node_id
