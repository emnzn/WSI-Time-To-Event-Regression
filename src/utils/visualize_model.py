import os
from typing import Union

from torchview import draw_graph
from graphviz.graphs import Digraph

from .model import (
    ResNet,
    SwinTransformer,
    AttentionBasedMIL
)

def generate_graph(
    model: Union[ResNet, SwinTransformer, AttentionBasedMIL], 
    destination: str,
    filename: str,
    expand_nested: bool,
    mil: bool
    ) -> Digraph:

    """
    Creates a graph to visualize the architecture of a given model.

    Parameters
    ----------
    model: Union[ResNet, SwinTransformer, AttentionBasedMIL]
        The model to be visualized.

    destination: str
        The path where the generated graph will be saved.

    model_name: str
        The filename to use when saving the graph.

    expand_nested: bool
        Wether to expand each node in the graph.

    mil: bool
        Whether a Multiple Instance-based model is being used.

    Returns
    -------
    graph: Digraph
        A digraph object that visualizes the model.
    """

    os.makedirs(destination, exist_ok=True)

    input_size = (1, 224 * 224, 1024) if mil else (1, 1024, 224, 224)

    model_graph = draw_graph(
        model, 
        input_size=input_size,
        graph_name=filename,
        expand_nested=expand_nested,
        save_graph=True, 
        directory=destination,
        filename=f"{filename}-architecture"
    )

    graph = model_graph.visual_graph
    
    return graph

