import torch
import torch.nn as nn
from base import BaseModel


class Genomic_Feature_Extractor(BaseModel):
    def __init__(self, genomic_dim=20, genomic_embedding_dim=8):
        super().__init__()
        self.genomic_dim = genomic_dim
        self.genomic_embedding_dim = genomic_embedding_dim

        self.genomic_feature_extractor = nn.Sequential(
            nn.Linear(self.genomic_dim, self.genomic_embedding_dim),
            nn.BatchNorm1d(self.genomic_embedding_dim),
            nn.Linear(self.genomic_embedding_dim, self.genomic_embedding_dim),
            nn.BatchNorm1d(self.genomic_embedding_dim),
            nn.Linear(self.genomic_embedding_dim, self.genomic_embedding_dim),
            nn.BatchNorm1d(self.genomic_embedding_dim)
        )

        self.initialization()

    def initialization(self):
        '''
        Initiate parameters in the model.
        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, genomic):
        return self.genomic_feature_extractor(genomic)


class Clinical_Feature_Extractor(BaseModel):
    def __init__(self, clinical_numerical_dim, clinical_categorical_dim, clinical_embedding_dim=8):
        super().__init__()
        self.clinical_numerical_dim = clinical_numerical_dim
        self.clinical_categorical_dim = clinical_categorical_dim
        self.clinical_embedding_dim = clinical_embedding_dim

        if self.clinical_categorical_dim:
            self.clinical_categorical_embedding = nn.Embedding(
                self.clinical_categorical_dim,
                self.clinical_embedding_dim - clinical_numerical_dim
            )

        self.clinical_feature_encoder = nn.Sequential(
            nn.Linear(self.clinical_embedding_dim, self.clinical_embedding_dim),
            nn.BatchNorm1d(self.clinical_embedding_dim),
            nn.Linear(self.clinical_embedding_dim, self.clinical_embedding_dim),
            nn.BatchNorm1d(self.clinical_embedding_dim),
            nn.Linear(self.clinical_embedding_dim, self.clinical_embedding_dim),
            nn.BatchNorm1d(self.clinical_embedding_dim)
        )
        self.initialization()

    def initialization(self):
        '''
        Initiate parameters in the model.
        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, clinical):
        batch_size = clinical.size(0)

        clinical_numerical, clinical_categorical = torch.split(
            clinical,
            [self.clinical_numerical_dim, self.clinical_categorical_dim],
            dim=1
        )

        if self.clinical_categorical_dim:
            clinical_categorical_embeddings = self.clinical_categorical_embedding(
                clinical_categorical.nonzero(as_tuple=True)[1].view(
                    clinical_categorical.size(0), -1
                )
            ).mean(dim=1)
        else:
            clinical_categorical_embeddings = torch.zeros(
                batch_size,
                self.clinical_embedding_dim - self.clinical_numerical_dim,
                device=clinical.device
            )

        return self.clinical_feature_encoder(
            torch.hstack([clinical_numerical, clinical_categorical_embeddings])
        ).squeeze()


class Feature_Extractor(BaseModel):
    def __init__(self, genomic_dim, clinical_numerical_dim, clinical_categorical_dim,
                 genomic_embedding_dim=8, clinical_embedding_dim=8):
        super().__init__()
        self.clinical_numerical_dim = clinical_numerical_dim
        self.clinical_categorical_dim = clinical_categorical_dim
        self.genomic_dim = genomic_dim
        self.clinical_dim = self.clinical_numerical_dim + self.clinical_categorical_dim

        self.genomic_embedding_dim = genomic_embedding_dim
        self.clinical_embedding_dim = clinical_embedding_dim
        self.embedding_dim = self.genomic_embedding_dim + self.clinical_embedding_dim

        if self.genomic_dim and self.genomic_embedding_dim:
            self.genomic_feature_extractor = Genomic_Feature_Extractor(
                genomic_dim=self.genomic_dim,
                genomic_embedding_dim=self.genomic_embedding_dim
            )
        if self.clinical_dim and self.clinical_embedding_dim:
            self.clinical_feature_extractor = Clinical_Feature_Extractor(
                clinical_numerical_dim=self.clinical_numerical_dim,
                clinical_categorical_dim=self.clinical_categorical_dim,
                clinical_embedding_dim=self.clinical_embedding_dim
            )

        self.initialization()

    def initialization(self):
        '''
        Initiate parameters in the model.
        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, genomic, clinical, project_id):
        if self.genomic_dim and self.genomic_embedding_dim:
            genomic_features = self.genomic_feature_extractor(genomic)
        else:
            genomic_features = torch.zeros(genomic.size(0), self.genomic_embedding_dim, device=genomic.device)

        if self.clinical_dim and self.clinical_embedding_dim:
            clinical_features = self.clinical_feature_extractor(clinical)
        else:
            clinical_features = torch.zeros(clinical.size(0), self.clinical_embedding_dim, device=clinical.device)

        features = torch.hstack([genomic_features, clinical_features])
        return features


class Genomic_Separate_Feature_Extractor(BaseModel):
    def __init__(self, genomic_dim, genomic_feature_extractor_num, clinical_numerical_dim, clinical_categorical_dim,
                 genomic_embedding_dim=8, clinical_embedding_dim=8):
        super().__init__()
        self.clinical_numerical_dim = clinical_numerical_dim
        self.clinical_categorical_dim = clinical_categorical_dim
        self.genomic_dim = genomic_dim
        self.genomic_feature_extractor_num = genomic_feature_extractor_num
        self.clinical_dim = self.clinical_numerical_dim + self.clinical_categorical_dim

        self.genomic_embedding_dim = genomic_embedding_dim
        self.clinical_embedding_dim = clinical_embedding_dim
        self.embedding_dim = self.genomic_embedding_dim + self.clinical_embedding_dim

        if self.genomic_dim and self.genomic_embedding_dim:
            self.genomic_feature_extractors = {}
            for i in range(self.genomic_feature_extractor_num):
                self.genomic_feature_extractors[str(i)] = Genomic_Feature_Extractor(
                    genomic_dim=self.genomic_dim,
                    genomic_embedding_dim=self.genomic_embedding_dim
                )
            self.genomic_feature_extractors = nn.ModuleDict(self.genomic_feature_extractors)
        if self.clinical_dim and self.clinical_embedding_dim:
            self.clinical_feature_extractor = Clinical_Feature_Extractor(
                clinical_numerical_dim=self.clinical_numerical_dim,
                clinical_categorical_dim=self.clinical_categorical_dim,
                clinical_embedding_dim=self.clinical_embedding_dim
            )

        self.initialization()

    def initialization(self):
        '''
        Initiate parameters in the model.
        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, genomic, clinical, project_id):
        genomic_features = torch.zeros(genomic.size(0), self.genomic_embedding_dim, device=genomic.device)

        for p_id in self.genomic_feature_extractors:
            i = int(p_id)
            if (project_id == i).sum() > 1:
                genomic_features[project_id == i] = self.genomic_feature_extractors[p_id](genomic[project_id == i])

        if self.clinical_dim and self.clinical_embedding_dim:
            clinical_features = self.clinical_feature_extractor(clinical)
        else:
            clinical_features = torch.zeros(clinical.size(0), self.clinical_embedding_dim, device=clinical.device)

        features = torch.hstack([genomic_features, clinical_features])
        return features


class Classifier(BaseModel):
    def __init__(self, genomic_embedding_dim=8, clinical_embedding_dim=8, output_dim=1, skip=False):
        super().__init__()
        self.genomic_embedding_dim = genomic_embedding_dim
        self.clinical_embedding_dim = clinical_embedding_dim
        self.embedding_dim = int((self.genomic_embedding_dim + self.clinical_embedding_dim) / 2)
        self.output_dim = output_dim

        self.classifier = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.Softsign(),
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.Softsign(),
            nn.Linear(self.embedding_dim, self.output_dim)
        )

        if skip:
            self.classifier = nn.Linear(2 * self.embedding_dim, self.output_dim)

        self.initialization()

    def initialization(self):
        '''
        Initiate parameters in the model.
        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, embeddings, project_id):
        return self.classifier(embeddings).squeeze()


class Task_Classifier(BaseModel):
    def __init__(self, task_dim, genomic_embedding_dim=8, clinical_embedding_dim=8, output_dim=1):
        super().__init__()
        self.task_dim = task_dim
        self.genomic_embedding_dim = genomic_embedding_dim
        self.clinical_embedding_dim = clinical_embedding_dim
        self.embedding_dim = int((self.genomic_embedding_dim + self.clinical_embedding_dim) / 2)
        self.output_dim = output_dim

        self.task_embedding = nn.Embedding(self.task_dim, self.embedding_dim)

        self.combine_layer = nn.Sequential(
            nn.Linear(2 * self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.Softsign()
        )

        self.classifier = nn.Sequential(
            nn.Linear(self.embedding_dim, self.embedding_dim),
            nn.BatchNorm1d(self.embedding_dim),
            nn.Softsign(),
            nn.Linear(self.embedding_dim, self.output_dim)
        )

        self.initialization()

    def initialization(self):
        '''
        Initiate parameters in the model.
        '''
        for m in self.modules():
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)

    def forward(self, embedding, project_id):
        task_embedding = self.task_embedding(project_id)
        embeddings = torch.add(self.combine_layer(embedding), task_embedding)
        return self.classifier(embeddings).squeeze()
