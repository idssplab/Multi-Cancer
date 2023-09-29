import torch.nn as nn
import torch
from dgl import DGLGraph
from dgl.nn.pytorch import GraphConv, Sequential, EdgeConv

from base import BaseModel
from .dnn import Clinical_Feature_Extractor


class Graph_Feature_Extractor(BaseModel):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, n_genes: int, genomic_embedding_dim: int):
        super().__init__()
        self.n_genes = n_genes
        # The model that reproduces extremely high AUPRC for BRCA with the Task Classifier.
        # self.graph_conv_0 = GraphConv(input_dim, hidden_dim, allow_zero_in_degree=True)
        # self.graph_conv_1 = GraphConv(hidden_dim, output_dim, allow_zero_in_degree=True)
        # self.bn_0 = nn.BatchNorm1d(hidden_dim)
        # self.bn_1 = nn.BatchNorm1d(output_dim)
        # self.activation = nn.ReLU()
        self.edge_conv = Sequential(
            EdgeConv(input_dim, hidden_dim, batch_norm=False, allow_zero_in_degree=True),
            EdgeConv(hidden_dim, output_dim, batch_norm=True, allow_zero_in_degree=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(output_dim * n_genes, genomic_embedding_dim),
            nn.BatchNorm1d(genomic_embedding_dim),
        )

    def forward(self, graph: DGLGraph):
        # h = self.graph_conv_0(graph, graph.ndata['feat'])
        # h = self.bn_0(h)
        # h = self.activation(h)
        # h = self.graph_conv_1(graph, h)
        # h = self.bn_1(h)
        # h: torch.Tensor = self.activation(h)
        h: torch.Tensor = self.edge_conv(graph, graph.ndata['feat'])
        return self.fc(h.reshape(h.shape[0] // self.n_genes, -1))


class Graph_And_Clinical_Feature_Extractor(BaseModel):
    def __init__(self, graph_input_dim, graph_hidden_dim, graph_output_dim, n_genes,
                 clinical_numerical_dim, clinical_categorical_dim,
                 genomic_embedding_dim=8, clinical_embedding_dim=8):
        super().__init__()

        # If genomic_embedding_dim is 0, then the genomic features will not be used.
        self.genomic_feature_extractor = Graph_Feature_Extractor(
            graph_input_dim,
            graph_hidden_dim,
            graph_output_dim,
            n_genes,
            genomic_embedding_dim
        ) if genomic_embedding_dim > 0 else None

        # If clinical_embedding_dim is 0, then the clinical features will not be used.
        self.clinical_feature_extractor = Clinical_Feature_Extractor(
            clinical_numerical_dim,
            clinical_categorical_dim,
            clinical_embedding_dim
        ) if clinical_embedding_dim > 0 else None

        # If both genomic_embedding_dim and clinical_embedding_dim are 0, then the configuration is invalid.
        if self.genomic_feature_extractor is None and self.clinical_feature_extractor is None:
            raise ValueError("Both genomic and clinical feature extractor cannot be None.")

        # Initialize linear layers and batch normalization layers. TODO: Other layers?
        self.initialization()

    def initialization(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, graph: DGLGraph, clinical, project_id):
        if self.genomic_feature_extractor is None:
            clinical_features: torch.Tensor = self.clinical_feature_extractor(clinical)
            genomic_features = torch.zeros(clinical_features.shape, device=clinical_features.device)

        elif self.clinical_feature_extractor is None:
            genomic_features: torch.Tensor = self.genomic_feature_extractor(graph)
            clinical_features = torch.zeros(genomic_features.shape, device=genomic_features.device)

        else:
            genomic_features: torch.Tensor = self.genomic_feature_extractor(graph)
            clinical_features: torch.Tensor = self.clinical_feature_extractor(clinical)

        return torch.hstack([genomic_features, clinical_features])
