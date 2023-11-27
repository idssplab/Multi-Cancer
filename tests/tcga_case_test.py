from preprocess import TCGA_Case
from utils.logger import setup_logging
from pathlib import Path
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

if __name__ == '__main__':
    setup_logging(Path('./Logs/Tests/'))

    Case_1 = TCGA_Case(case_id='f130f376-5801-40f9-975d-a7e2f7b5670d', directory='./Data/TCGA-BRCA/f130f376-5801-40f9-975d-a7e2f7b5670d')

    print(Case_1.genomic)
    print(Case_1.clinical)
    print(Case_1.overall_survival)
    print(Case_1.disease_specific_survival)
    print(Case_1.primary_site)
