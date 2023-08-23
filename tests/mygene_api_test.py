from utils.api.mygene_api import *
from utils import set_random_seed

SEED = 1126
set_random_seed(SEED)

if __name__ == '__main__':
    print(get_gene_annotation(ensembl_id='1017'))
    print(get_gene_annotation(ensembl_id='ENSG00000123374'))
    print(get_gene_annotation(ensembl_id='1017', fields=['name', 'symbol']))
    print(get_gene_annotation(ensembl_id='1017', fields=['name', 'symbol', 'summary']))
    print(get_gene_annotations(ensembl_ids=['1017', '695'], fields=['name', 'symbol', 'refseq.rna']))
    print(get_gene_annotations(ensembl_ids=['1017', '695'], fields=['name', 'symbol', 'refseq.rna']))
    print(get_gene_annotations(ensembl_ids=['ENSG00000123374', 'ENSG00000270112'], fields=['name', 'symbol', 'refseq.rna']))
